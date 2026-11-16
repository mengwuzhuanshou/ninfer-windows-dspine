#!/usr/bin/env python3
"""Frame the released Clef joint schema head for the C++ engine path.

The engine has no Python model route and no safetensors reader, so `ninfer-decide-serve` loads the
head from this flat sidecar instead of `joint_head.safetensors`. Values are copied **verbatim** in
the released BF16 -- the sidecar is a framing change, never a numerical one, and that is what makes
the C++/Python parity check meaningful.

Head sidecar layout (little-endian, no padding between index fields):

    magic        8 bytes  b"NINHHEAD"
    version      u32      1
    tensor_count u32
    reserved     u32      0
    index        tensor_count records, in name-sorted order, each:
                   u32       name_len
                   bytes     name              (UTF-8, key as in the safetensors file)
                   u32       ndim
                   i64[ndim] dims              (outermost first, PyTorch order)
                   u32       dtype             0 = BF16, 1 = FP32
                   u64       offset             from file start to the blob
                   u64       nbytes
    blobs        tensor data, each blob start aligned up to 64 bytes

`--lexical` writes the raw lm_head table that the head indexes by absolute vocabulary id
(`joint_schema_model.py:381`). The engine artifact stores that matrix after an FP8 round trip while
the released BF16 shards hold the exact table, and the served Python path deliberately used the
exact one -- so this file is the product source of lexical rows. Rows are fixed size and directly
seekable at `64 + row * columns * 2`.

    lexical layout: b"NINHLEX\\0" + u32 version + u32 rows + u32 columns + u32 dtype + u64 data_offset
                    + u64 data_bytes + zero pad to 64, then the BF16 rows.

Run:
    python tools/export_joint_head.py \
        --release path/to/clef_release_dir \
        --out     models/clef_joint_head.v1.bin \
        --lexical models/clef_lmhead_bf16.v1.bin
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import torch
from safetensors import safe_open

MAGIC = b"NINHHEAD"
VERSION = 1
LEXICAL_MAGIC = b"NINHLEX\x00"
LEXICAL_VERSION = 1
ALIGN = 64
DTYPE_CODE = {"BF16": 0, "BFLOAT16": 0, "FP32": 1, "FLOAT32": 1, "FLOAT": 1}
DTYPE_BYTES = {0: 2, 1: 4}


def dtype_code(value: object) -> int:
    """safetensors says "BF16" while torch says "torch.bfloat16"; both mean the same bits."""
    name = str(value).replace("torch.", "").replace("dtype(", "").upper()
    if name not in DTYPE_CODE:
        raise SystemExit(f"unsupported dtype in sidecar source: {value}")
    return DTYPE_CODE[name]


def align_up(value: int) -> int:
    return (value + ALIGN - 1) // ALIGN * ALIGN


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as handle:
        for name in handle.keys():
            out[name] = handle.get_tensor(name)
    return out


def build_index(tensors: dict[str, torch.Tensor]) -> list[tuple[str, list[int], int, int, int]]:
    header_size = 8 + 4 + 4 + 4
    index = []
    cursor = header_size
    for name in sorted(tensors):
        dims = list(tensors[name].shape)
        cursor += 4 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8 + 8
    cursor = align_up(cursor)
    for name in sorted(tensors):
        tensor = tensors[name]
        dims = list(tensor.shape)
        code = dtype_code(tensor.dtype)
        nbytes = 1
        for extent in dims:
            nbytes *= int(extent)
        nbytes *= DTYPE_BYTES[code]
        index.append((name, dims, code, cursor, nbytes))
        cursor = align_up(cursor + nbytes)
    return index


def raw_bytes(tensor: torch.Tensor, code: int) -> bytes:
    # BF16 has no native numpy dtype; keep the storage bits by viewing as uint16.
    contiguous = tensor.contiguous()
    if code == 0:
        return contiguous.view(torch.uint16).numpy().tobytes()
    return contiguous.numpy().tobytes()


def export_head(release: Path, out: Path) -> dict:
    tensors = load_tensors(release / "joint_head.safetensors")
    index = build_index(tensors)
    with out.open("wb") as sink:
        sink.write(MAGIC)
        sink.write(struct.pack("<III", VERSION, len(index), 0))
        for name, dims, code, offset, nbytes in index:
            encoded = name.encode()
            sink.write(struct.pack("<I", len(encoded)))
            sink.write(encoded)
            sink.write(struct.pack("<I", len(dims)))
            for extent in dims:
                sink.write(struct.pack("<q", int(extent)))
            sink.write(struct.pack("<I", code))
            sink.write(struct.pack("<QQ", offset, nbytes))
        # The index ends wherever it ends; every blob, including the first, starts 64-byte aligned.
        sink.write(b"\0" * (align_up(sink.tell()) - sink.tell()))
        for name, dims, code, offset, nbytes in index:
            payload = raw_bytes(tensors[name], code)
            assert len(payload) == nbytes, (name, len(payload), nbytes)
            assert sink.tell() == offset, (name, sink.tell(), offset)
            sink.write(payload)
            padding = align_up(sink.tell()) - sink.tell()
            sink.write(b"\0" * padding)
        sink.flush()
        os.fsync(sink.fileno())
    # Read the sidecar back and require every tensor's bytes to be exactly the released ones: a
    # framing tool that silently drops or shifts a blob would otherwise show up as a head mismatch
    # much later, in a parity number nobody can attribute.
    with out.open("rb") as source:
        for name, dims, code, offset, nbytes in index:
            source.seek(offset)
            stored = source.read(nbytes)
            if stored != raw_bytes(tensors[name], code):
                raise SystemExit(f"sidecar round-trip differs for {name} at offset {offset}")
    return {
        "tensors": len(index),
        "bytes": out.stat().st_size,
        "params": sum(nbytes // DTYPE_BYTES[code] for _, _, code, _, nbytes in index),
        "shapes": {name: dims for name, dims, _, _, _ in index},
    }


def export_lexical(release: Path, out: Path) -> dict:
    index = json.loads((release / "model.safetensors.index.json").read_text())
    shard = release / index["weight_map"]["lm_head.weight"]
    with safe_open(str(shard), framework="pt") as handle:
        slice_ = handle.get_slice("lm_head.weight")
        rows, columns = [int(v) for v in slice_.get_shape()]
        dtype_text = str(slice_.get_dtype())
    if dtype_text != "BF16":
        raise SystemExit(f"lm_head is {dtype_text}; this framer only handles the released BF16 table")
    data_bytes = rows * columns * 2
    with safe_open(str(shard), framework="pt") as handle:
        table = handle.get_slice("lm_head.weight")
        with out.open("wb") as sink:
            sink.write(LEXICAL_MAGIC)
            sink.write(struct.pack("<IIIIIQQ", LEXICAL_VERSION, rows, columns,
                                   DTYPE_CODE["BF16"], 0, 64, data_bytes))
            sink.write(b"\0" * (64 - sink.tell()))
            step = 8192
            for start in range(0, rows, step):
                stop = min(start + step, rows)
                block = table[start:stop].contiguous()
                sink.write(block.view(torch.uint16).numpy().tobytes())
                if (start // step) % 16 == 0:
                    print(f"  lexical rows {start}-{stop} / {rows}", flush=True)
            if sink.tell() != 64 + data_bytes:
                raise SystemExit(f"lexical table truncated: wrote {sink.tell()} bytes")
            sink.flush()
            os.fsync(sink.fileno())
    # The engine addresses a row by seek, not by scanning, so the thing that has to be true is
    # `row r lives at 64 + r * columns * 2`. Sample the ends and a deterministic spread.
    probe = sorted({0, rows - 1, rows // 2} | {i * (rows - 1) // 47 for i in range(48)})
    row_bytes = columns * 2
    with safe_open(str(shard), framework="pt") as handle, out.open("rb") as source:
        table = handle.get_slice("lm_head.weight")
        for row in probe:
            expected = table[row:row + 1].contiguous().view(torch.uint16).numpy().tobytes()
            source.seek(64 + row * row_bytes)
            if source.read(row_bytes) != expected:
                raise SystemExit(f"lexical row {row} does not round-trip at its seek offset")
    return {"rows": rows, "columns": columns, "bytes": out.stat().st_size,
            "rows_verified": len(probe)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="directory holding joint_head.safetensors")
    ap.add_argument("--out", help="head sidecar to write")
    ap.add_argument("--lexical", help="optional lm_head BF16 table to write")
    ap.add_argument("--report", help="optional json summary")
    args = ap.parse_args()

    release = Path(args.release)
    summary: dict = {}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary["head"] = export_head(release, out)
        print(f"head sidecar: {summary['head']['tensors']} tensors, "
              f"{summary['head']['bytes'] / 2**20:.1f} MiB -> {out}", flush=True)
    if args.lexical:
        lexical = Path(args.lexical)
        lexical.parent.mkdir(parents=True, exist_ok=True)
        summary["lexical"] = export_lexical(release, lexical)
        print(f"lexical table: {summary['lexical']['rows']} x {summary['lexical']['columns']}, "
              f"{summary['lexical']['bytes'] / 2**30:.2f} GiB -> {lexical}", flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
