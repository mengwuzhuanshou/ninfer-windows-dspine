#!/usr/bin/env python3
"""Attach a DSpine draft head to an existing NInfer v3 artifact.

    python -X utf8 tools/attach_dspine_head.py TARGET.ninfer DSPINE_DIR OUTPUT.ninfer

DSPINE_DIR is the HF checkpoint directory (config.json + model.safetensors).
Backbone linears are quantized to NVFP4 (weight-only, A16Only); the code bridge,
norms and codebook stay BF16. Weight bytes of the target are copied verbatim;
the DSpine payload region is appended after the target payload.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import sys
import uuid

_CONVERT_ROOT = Path(
    os.environ.get("NINFER_CONVERT_ROOT",
                   str(Path(__file__).resolve().parent.parent / "converter"))
)
sys.path.insert(0, str(_CONVERT_ROOT))

import torch  # noqa: E402

from tools.artifact.layouts import (  # noqa: E402
    dequantize_fp8_row_scaled,
    encode_direct,
)
from tools.convert.common.safetensors import ShardReader  # noqa: E402
from tools.convert.quantization.nvfp4_encode import (  # noqa: E402
    dequantize_nvfp4,
    encode_nvfp4_parent,
    quantize_nvfp4,
    relative_frobenius_error,
)
from tools.convert.qwen3_8_27b import dspine_recipe  # noqa: E402

HEADER = struct.Struct("<8sQ16s")
MAGIC = b"NINFER\x00\x03"
ALIGN = 4096
OBJ_ALIGN = 256
CHUNK = 64 * 1024 * 1024
PREFIX = dspine_recipe.PREFIX
# Opt-in: store the codebook and the TG/TSV serial tables as real NVFP4 (weight-only) instead of
# BF16. The engine gather kernels dequantize rows on access; saves ~870 MiB of draft VRAM.
TABLES_NVFP4 = os.environ.get("NINFER_DSPINE_TABLES_NVFP4") == "1"


def align(value: int, amount: int = ALIGN) -> int:
    return (value + amount - 1) // amount * amount


def read_directory(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        magic, json_bytes, _ = HEADER.unpack(handle.read(32))
        if magic != MAGIC:
            raise SystemExit(f"{path}: not a NInfer v3 artifact (magic {magic!r})")
        directory = json.loads(handle.read(json_bytes))
    payload = directory["files"][0]["payload_bytes"]
    if align(32 + json_bytes) + payload != os.path.getsize(path):
        raise SystemExit(f"{path}: entry length differs from directory")
    return directory, 32 + json_bytes


def base_summary(target: dict) -> dict:
    text = target["components"]["text"]["config"]
    return {
        "text": {
            name: text[name]
            for name in (
                "hidden_size",
                "vocab_size",
                "num_hidden_layers",
                "max_position_embeddings",
            )
        },
        "rope": {"rope_theta": text["rope_parameters"]["rope_theta"]},
    }


def serialize(directory: dict) -> tuple[bytes, int]:
    reserve = ALIGN - 32
    while True:
        data = json.dumps(
            directory, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
        if len(data) <= reserve:
            return data + b" " * (reserve - len(data)), reserve
        reserve = align(32 + len(data)) - 32


def build_payloads(dspine_dir: Path, device: str) -> list[tuple[dict, bytes]]:
    entries: list[tuple[dict, bytes]] = []
    with ShardReader.from_file(dspine_dir / "model.safetensors") as reader:
        for recipe in dspine_recipe.DSPINE_RECIPE_SPECS:
            name = recipe.object_name
            tensor = dspine_recipe.materialize_tensor(name, reader)
            shape = dspine_recipe.object_shape(name)
            if tuple(tensor.shape) != shape:
                raise SystemExit(f"{name}: shape {tuple(tensor.shape)} != {shape}")
            fmt = dspine_recipe.object_format(name)
            if TABLES_NVFP4 and name == "dspine/bridge/codebook":
                fmt = dspine_recipe.NVFP4
            if fmt == dspine_recipe.NVFP4:
                words = quantize_nvfp4(tensor, device=device)
                payload = encode_nvfp4_parent(tensor, device=device)
                restored = dequantize_nvfp4(words, device=device)
                error = relative_frobenius_error(tensor, restored)
                print(
                    f"  {name}  NVFP4 {shape}  rel_err={error:.5f}",
                    flush=True,
                )
                layout = "block_scale_k16_m128x4_v1"
                fmt_name = "nvfp4"
                del words, restored
            else:
                payload = encode_direct(tensor, "BF16")
                layout = "contiguous_le_v1"
                fmt_name = "bf16"
            entries.append(
                (
                    {
                        "id": name,
                        "kind": "tensor",
                        "shape": list(shape),
                        "format": fmt_name,
                        "layout": layout,
                        "bytes": len(payload),
                    },
                    payload,
                )
            )
            del tensor, payload
    return entries


def compute_serial_tables(
    target_path: str,
    target_start: int,
    target: dict,
    dspine_dir: Path,
    device: str,
) -> list[tuple[dict, bytes]]:
    """Precompute the vocabulary-wide greedy refinement tables (BF16).

    Mirrors dspine_mlx.bridge.build_serial_tables rounding point by rounding
    point: TG rows are bf16(fp32(lm_head) * final_norm) @ write_w[-1], TSV rows
    are bf16(sqrt(C) * codebook) @ [predecessor_gate; prev].T. The target
    lm_head is stored FP8 row-scaled, so it is dequantized to dense BF16 first.
    """
    import math

    head_id = target["bindings"]["text/output_head"]["object"]
    head_obj = next(obj for obj in target["objects"] if obj["id"] == head_id)
    if head_obj["format"] != "fp8_e4m3fn_row_bf16":
        raise SystemExit(f"text/output_head: unexpected format {head_obj['format']}")
    vocab, hidden = head_obj["shape"]
    with open(target_path, "rb") as handle:
        handle.seek(target_start + head_obj["offset"])
        head_payload = handle.read(head_obj["bytes"])
    if len(head_payload) != head_obj["bytes"]:
        raise SystemExit("text/output_head payload truncated")
    head = dequantize_fp8_row_scaled(head_payload, (vocab, hidden)).to(device)
    del head_payload

    with ShardReader.from_file(dspine_dir / "model.safetensors") as reader:
        norm_w = dspine_recipe.materialize_tensor("dspine/final_norm", reader).to(device)
        write_w = dspine_recipe.materialize_tensor(
            "dspine/bridge/feature_write_weights", reader
        )[-1].to(device)
        pred = dspine_recipe.materialize_tensor(
            "dspine/bridge/predecessor_gate", reader
        ).to(device)
        prev = dspine_recipe.materialize_tensor("dspine/bridge/prev", reader).to(device)
        codebook = dspine_recipe.materialize_tensor(
            "dspine/bridge/codebook", reader
        ).to(device)
    rank = write_w.shape[1]
    code_size = codebook.shape[1]
    pv = torch.cat([pred, prev], dim=0)
    sqrt_c = math.sqrt(code_size)
    norm_f = norm_w.float()

    chunk = 8192
    tg_parts: list[torch.Tensor] = []
    tsv_parts: list[torch.Tensor] = []
    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        tg_parts.append(
            (head[start:end].float() * norm_f).to(torch.bfloat16) @ write_w
        )
        tsv_parts.append(
            (sqrt_c * codebook[start:end].float()).to(torch.bfloat16) @ pv.T
        )
    tg = torch.cat(tg_parts, dim=0).contiguous()
    tsv = torch.cat(tsv_parts, dim=0).contiguous()
    del head, tg_parts, tsv_parts
    print(
        f"  serial tables  TG {tuple(tg.shape)}  TSV {tuple(tsv.shape)}  BF16",
        flush=True,
    )
    entries = []
    for name, tensor in (("dspine/bridge/tg", tg), ("dspine/bridge/tsv", tsv)):
        shape = dspine_recipe.SERIAL_TABLE_OBJECTS[name]
        if tuple(tensor.shape) != shape:
            raise SystemExit(f"{name}: shape {tuple(tensor.shape)} != {shape}")
        if TABLES_NVFP4:
            payload = encode_nvfp4_parent(tensor, device=device)
            fmt_name, layout = "nvfp4", "block_scale_k16_m128x4_v1"
            words = quantize_nvfp4(tensor, device=device)
            error = relative_frobenius_error(tensor, dequantize_nvfp4(words, device=device))
            print(f"  {name}  NVFP4 {shape}  rel_err={error:.5f}", flush=True)
            del words
        else:
            payload = encode_direct(tensor, "BF16")
            fmt_name, layout = "bf16", "contiguous_le_v1"
        entries.append(
            (
                {
                    "id": name,
                    "kind": "tensor",
                    "shape": list(shape),
                    "format": fmt_name,
                    "layout": layout,
                    "bytes": len(payload),
                },
                payload,
            )
        )
    return entries


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    target_path, dspine_dir, output_path = sys.argv[1:4]
    dspine_dir = Path(dspine_dir)
    if os.path.exists(output_path):
        raise SystemExit(f"refusing to overwrite {output_path}")

    target, target_start = read_directory(target_path)
    if any(obj["id"].startswith(PREFIX) for obj in target["objects"]):
        raise SystemExit("target artifact already carries a DSpine head")

    config = json.loads((dspine_dir / "config.json").read_text(encoding="utf-8"))
    summary = dspine_recipe.validate_config(config)
    dspine_recipe.validate_base_compatibility(base_summary(target), summary)
    preflight = dspine_recipe.preflight_sources(dspine_dir)
    print(
        f"preflight: {preflight.recipe_count} recipes, "
        f"{preflight.source_tensor_count} source tensors",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    entries = build_payloads(dspine_dir, device)
    entries += compute_serial_tables(target_path, target_start, target, dspine_dir, device)

    target_payload = target["files"][0]["payload_bytes"]
    last_end = max(obj["offset"] + obj["bytes"] for obj in target["objects"])
    if last_end > target_payload:
        raise SystemExit("target payload does not cover its own objects")
    new_base = align(max(target_payload, last_end))

    cursor = new_base
    attached = []
    for obj, payload in entries:
        cursor = align(cursor, OBJ_ALIGN)
        attached.append(dict(obj, offset=cursor))
        cursor += len(payload)
    payload_total = cursor

    merged = dict(target)
    merged["components"] = dict(target["components"])
    merged["components"]["dspine"] = dspine_recipe.build_component(summary)
    merged["objects"] = target["objects"] + attached
    merged["bindings"] = dict(target["bindings"])
    merged["bindings"].update(dspine_recipe.build_bindings())
    merged["uses"] = target["uses"] + dspine_recipe.build_uses()
    merged["files"] = [{"path": None, "payload_bytes": payload_total}]

    data, reserve = serialize(merged)
    print(
        f"target payload {target_payload:,} | dspine region "
        f"{payload_total - new_base:,} in {len(attached)} objects | "
        f"json reserve {reserve:,} | entry bytes {32 + reserve + payload_total:,}",
        flush=True,
    )

    with open(target_path, "rb") as source, open(output_path, "wb") as out:
        out.write(HEADER.pack(MAGIC, reserve, uuid.uuid4().bytes))
        out.write(data)
        remaining = target_payload
        source.seek(target_start)
        while remaining:
            block = source.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit("target payload truncated")
            out.write(block)
            remaining -= len(block)
        payload_start = 32 + reserve
        for obj, (_, payload) in zip(attached, entries):
            gap = payload_start + obj["offset"] - out.tell()
            if gap < 0:
                raise SystemExit(f"{obj['id']}: overlapping payload layout")
            out.write(b"\0" * gap)
            out.write(payload)
        if out.tell() != 32 + reserve + payload_total:
            raise SystemExit("output layout does not match directory")
        out.flush()
        os.fsync(out.fileno())

    if os.path.getsize(output_path) != 32 + reserve + payload_total:
        raise SystemExit("output size mismatch")
    check, _ = read_directory(output_path)
    if len(check["objects"]) != len(merged["objects"]):
        raise SystemExit("output directory object count mismatch")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()