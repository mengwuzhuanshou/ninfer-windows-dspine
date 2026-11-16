#!/usr/bin/env python3
"""Fuse a standalone DSpine draft suffix onto a target tower NInfer v3 artifact.

    python -X utf8 tools/fuse_dspine_draft.py TARGET.ninfer DRAFT_SUFFIX.ninfer OUTPUT.ninfer

TARGET is a base tower artifact without a DSpine head (e.g. the official or Coder390
DFlash2 .ninfer). DRAFT_SUFFIX is produced by extract_dspine_draft.py and already carries
the tower-specific TG/TSV serial tables, so it must match the target tower it was extracted
for. The target payload is copied verbatim; the draft region is appended after it. This is
the same layout attach_dspine_head.py produces, but reuses pre-built draft bytes instead of
requantizing from the checkpoint.
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


def check_compat(target: dict, draft: dict) -> None:
    text = target["components"]["text"]["config"]
    dc = draft["components"]["dspine"]["config"]
    for key in ("hidden_size", "vocab_size"):
        if text.get(key) != dc.get(key):
            raise SystemExit(f"draft/target {key} mismatch: {dc.get(key)} vs {text.get(key)}")
    if text["rope_parameters"]["rope_theta"] != dc["rope_parameters"]["rope_theta"]:
        raise SystemExit("draft/target rope_theta mismatch")


def copy_payload(src_path: str, src_start: int, out, begin: int, nbytes: int) -> None:
    with open(src_path, "rb") as src:
        src.seek(src_start + begin)
        remaining = nbytes
        while remaining:
            block = src.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit("payload truncated")
            out.write(block)
            remaining -= len(block)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    target_path, draft_path, out_path = sys.argv[1:4]
    if os.path.exists(out_path):
        raise SystemExit(f"refusing to overwrite {out_path}")
    target, target_start = read_directory(target_path)
    draft, draft_start = read_directory(draft_path)
    if any(o["id"].startswith(PREFIX) for o in target["objects"]):
        raise SystemExit("target artifact already carries a DSpine head")
    if not all(o["id"].startswith(PREFIX) for o in draft["objects"]):
        raise SystemExit("draft suffix contains non-dspine objects")
    check_compat(target, draft)

    target_payload = target["files"][0]["payload_bytes"]
    last_end = max(o["offset"] + o["bytes"] for o in target["objects"])
    if last_end > target_payload:
        raise SystemExit("target payload does not cover its own objects")
    new_base = align(max(target_payload, last_end))

    draft_payload = draft["files"][0]["payload_bytes"]
    attached = []
    for o in draft["objects"]:
        c = dict(o)
        c["offset"] = new_base + o["offset"]
        attached.append(c)
    payload_total = new_base + draft_payload

    merged = dict(target)
    merged["components"] = dict(target["components"])
    merged["components"]["dspine"] = draft["components"]["dspine"]
    merged["objects"] = target["objects"] + attached
    merged["bindings"] = dict(target["bindings"])
    merged["bindings"].update(draft["bindings"])
    merged["uses"] = target["uses"] + draft["uses"]
    merged["files"] = [{"path": None, "payload_bytes": payload_total}]

    data, reserve = serialize(merged)
    print(f"target payload {target_payload:,} | dspine region {draft_payload:,} "
          f"in {len(attached)} objects | json reserve {reserve:,} | "
          f"entry bytes {32 + reserve + payload_total:,}", flush=True)

    with open(out_path, "wb") as out:
        out.write(HEADER.pack(MAGIC, reserve, uuid.uuid4().bytes))
        out.write(data)
        copy_payload(target_path, target_start, out, 0, target_payload)
        gap = 32 + reserve + new_base - out.tell()
        if gap < 0:
            raise SystemExit("overlapping payload layout")
        out.write(b"\0" * gap)
        copy_payload(draft_path, draft_start, out, 0, draft_payload)
        if out.tell() != 32 + reserve + payload_total:
            raise SystemExit("output layout does not match directory")
        out.flush()
        os.fsync(out.fileno())
    if os.path.getsize(out_path) != 32 + reserve + payload_total:
        raise SystemExit("output size mismatch")
    check, _ = read_directory(out_path)
    if len(check["objects"]) != len(merged["objects"]):
        raise SystemExit("output directory object count mismatch")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()