#!/usr/bin/env python3
"""Sum the DSpine draft region bytes by format and compare against an all-BF16 draft."""
import json
import struct
import sys
from collections import defaultdict

HEADER = struct.Struct("<8sQ16s")
MAGIC = b"NINFER\x00\x03"


def read_directory(path):
    with open(path, "rb") as handle:
        magic, json_bytes, _ = HEADER.unpack(handle.read(32))
        if magic != MAGIC:
            raise SystemExit(f"{path}: not v3")
        return json.loads(handle.read(json_bytes))


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def main():
    path = sys.argv[1]
    d = read_directory(path)
    objs = [o for o in d["objects"] if o["id"].startswith("dspine/")]
    by_fmt = defaultdict(lambda: [0, 0])  # fmt -> [stored_bytes, bf16_equiv_bytes]
    rows = []
    for o in objs:
        fmt = o.get("format")
        stored = o["bytes"]
        bf16 = numel(o["shape"]) * 2
        by_fmt[fmt][0] += stored
        by_fmt[fmt][1] += bf16
        rows.append((o["id"], fmt, tuple(o["shape"]), stored, bf16))
    print(f"artifact: {path}")
    print(f"dspine objects: {len(objs)}")
    print(f"{'object':<46}{'fmt':<10}{'shape':<22}{'stored':>14}{'bf16_equiv':>14}")
    for name, fmt, shape, stored, bf16 in rows:
        print(f"{name:<46}{str(fmt):<10}{str(shape):<22}{stored:>14,}{bf16:>14,}")
    print("\n=== by format ===")
    tot_stored = tot_bf16 = 0
    for fmt, (stored, bf16) in sorted(by_fmt.items(), key=lambda x: str(x[0])):
        print(f"{str(fmt):<12} stored {stored:>14,} ({stored/2**20:8.1f} MiB)   "
              f"bf16_equiv {bf16:>14,} ({bf16/2**20:8.1f} MiB)")
        tot_stored += stored
        tot_bf16 += bf16
    print(f"\nCURRENT draft weights stored : {tot_stored:>14,} bytes = {tot_stored/2**20:.1f} MiB = {tot_stored/2**30:.3f} GiB")
    print(f"ALL-BF16 draft weights       : {tot_bf16:>14,} bytes = {tot_bf16/2**20:.1f} MiB = {tot_bf16/2**30:.3f} GiB")
    print(f"nvfp4 saving vs all-BF16     : {tot_bf16-tot_stored:>14,} bytes = {(tot_bf16-tot_stored)/2**20:.1f} MiB = {(tot_bf16-tot_stored)/2**30:.3f} GiB")


if __name__ == "__main__":
    main()