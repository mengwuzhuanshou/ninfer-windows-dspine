#!/usr/bin/env python3
"""Extract the DSpine draft suffix from a full NInfer v3 artifact into a standalone file.

    python -X utf8 tools/extract_dspine_draft.py FULL.ninfer DRAFT_SUFFIX.ninfer

The result holds only the dspine/* objects (backbone, code bridge, and the tower-specific
TG/TSV serial tables), with offsets rebased to zero. It is not loadable on its own; fuse it
onto a compatible target tower with fuse_dspine_draft.py. Pure byte copy: no requantization,
the draft payload is preserved verbatim from the validated full artifact.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import uuid

HEADER = struct.Struct("<8sQ16s")
MAGIC = b"NINFER\x00\x03"
ALIGN = 4096
OBJ_ALIGN = 256
CHUNK = 64 * 1024 * 1024
PREFIX = "dspine/"


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


def serialize(directory: dict) -> tuple[bytes, int]:
    reserve = ALIGN - 32
    while True:
        data = json.dumps(directory, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode()
        if len(data) <= reserve:
            return data + b" " * (reserve - len(data)), reserve
        reserve = align(32 + len(data)) - 32


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    full_path, out_path = sys.argv[1:3]
    if os.path.exists(out_path):
        raise SystemExit(f"refusing to overwrite {out_path}")
    full, full_start = read_directory(full_path)

    objs = [o for o in full["objects"] if o["id"].startswith(PREFIX)]
    if not objs:
        raise SystemExit("source artifact carries no DSpine head")
    region_begin = min(o["offset"] for o in objs)
    region_end = max(o["offset"] + o["bytes"] for o in objs)
    if region_begin % OBJ_ALIGN:
        raise SystemExit("dspine region is not object-aligned")

    rebased = []
    for o in objs:
        c = dict(o)
        c["offset"] = o["offset"] - region_begin
        rebased.append(c)
    payload_bytes = region_end - region_begin

    bindings = {k: v for k, v in full["bindings"].items() if k.startswith(PREFIX)}
    uses = [u for u in full["uses"]
            if u.get("parameter", "").startswith(PREFIX) or u.get("input", "").startswith(PREFIX)]
    merged = {
        "components": {"dspine": full["components"]["dspine"]},
        "objects": rebased,
        "bindings": bindings,
        "uses": uses,
        "files": [{"path": None, "payload_bytes": payload_bytes}],
    }
    data, reserve = serialize(merged)
    print(f"dspine region {payload_bytes:,} bytes in {len(rebased)} objects | "
          f"json reserve {reserve:,}", flush=True)

    with open(full_path, "rb") as src, open(out_path, "wb") as out:
        out.write(HEADER.pack(MAGIC, reserve, uuid.uuid4().bytes))
        out.write(data)
        src.seek(full_start + region_begin)
        remaining = payload_bytes
        while remaining:
            block = src.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit("dspine payload truncated")
            out.write(block)
            remaining -= len(block)
        out.flush()
        os.fsync(out.fileno())
    if os.path.getsize(out_path) != 32 + reserve + payload_bytes:
        raise SystemExit("output size mismatch")
    check, _ = read_directory(out_path)
    if len(check["objects"]) != len(rebased):
        raise SystemExit("output object count mismatch")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()