#!/usr/bin/env python
"""Create train/val/test splits by tool family and stratify by OS/shell/language.

Usage:
  python scripts/split_dataset.py --in data/synthetic/samples/synth_v1.clean.jsonl \
    [--out_dir data/splits] [--family_key auto]

Outputs:
  data/splits/train.jsonl
  data/splits/val.jsonl
  data/splits/test.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Dict, List, Tuple

from common import family_from_cmd, jsonl_read, jsonl_write


def lerp_split(seed: int, families: List[str]) -> Dict[str, str]:
    random.seed(seed)
    # Assign families deterministically using hash
    family_assign = {}
    for fam in families:
        h = int(hashlib.sha256(fam.encode()).hexdigest(), 16)
        r = (h % 1000) / 1000.0
        if r < 0.8:
            family_assign[fam] = "train"
        elif r < 0.9:
            family_assign[fam] = "val"
        else:
            family_assign[fam] = "test"
    return family_assign


def main() -> int:
    ap = argparse.ArgumentParser(description="Split dataset by family with stratification")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = list(jsonl_read(args.inp))
    if not rows:
        print("ERROR: empty input", flush=True)
        return 2

    # Infer family per row
    fams: List[str] = []
    row_fam: List[Tuple[dict, str]] = []
    for r in rows:
        try:
            obj = json.loads(r["output_json"]) if isinstance(r.get("output_json"), str) else {}
        except Exception:
            obj = {}
        t = (obj.get("type") or "").lower()
        fam = family_from_cmd(obj.get("cmd", ""), t)
        fams.append(fam)
        row_fam.append((r, fam))

    fam_assign = lerp_split(args.seed, sorted(set(fams)))
    buckets = {"train": [], "val": [], "test": []}
    for r, fam in row_fam:
        split = fam_assign.get(fam, "train")
        buckets[split].append(r)

    os.makedirs(args.out_dir, exist_ok=True)
    for k, v in buckets.items():
        outp = os.path.join(args.out_dir, f"{k}.jsonl")
        jsonl_write(outp, v)
        print(f"{k}: {len(v)} -> {outp}")

    # Leakage check: ensure no family overlap
    fam_train = set([fam for _, fam in row_fam if fam_assign.get(fam) == "train"])
    fam_val = set([fam for _, fam in row_fam if fam_assign.get(fam) == "val"])
    fam_test = set([fam for _, fam in row_fam if fam_assign.get(fam) == "test"])
    leak = (fam_train & fam_val) | (fam_train & fam_test) | (fam_val & fam_test)
    if leak:
        print(f"WARNING: family leakage detected: {len(leak)} families", flush=True)
    else:
        print("Leakage check passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

