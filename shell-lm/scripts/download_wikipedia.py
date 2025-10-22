#!/usr/bin/env python
"""Pre-download Hugging Face wikimedia/wikipedia and optionally export seed text files.

Examples:
  # Cache EN+ZH 20231101 and export seed .txt files under data/synthetic/raw_text
  python scripts/download_wikipedia.py --version 20231101 --langs en zh \
    --samples 5000 --export --per_file 200 --out_dir data/synthetic/raw_text

  # Only warm the HF cache (no export)
  python scripts/download_wikipedia.py --version 20231101 --langs en zh --samples 10000

The synthesis script (synth_generate.py) already loads from HF, but exporting seeds
can make runs more reproducible and allows offline use of local context.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

try:
    from datasets import load_dataset
except Exception as e:
    load_dataset = None


def export_language(version: str, lang: str, samples: int, out_dir: str, per_file: int, do_export: bool) -> int:
    if load_dataset is None:
        print("ERROR: datasets package is required. pip install datasets", file=sys.stderr)
        return 2
    name = f"{version}.{lang}"
    try:
        ds = load_dataset("wikimedia/wikipedia", name=name, split="train")
    except Exception as e:
        print(f"ERROR: failed to load wikimedia/wikipedia:{name}: {e}", file=sys.stderr)
        return 3
    n = min(samples, len(ds)) if samples > 0 else len(ds)
    # Warm cache
    ds_head = ds.select(range(n))

    if not do_export:
        print(f"Cached {n} rows for {name}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    buf: List[str] = []
    file_idx = 1
    written = 0
    for ex in ds_head:
        t = (ex.get("text") or "").strip()
        if len(t) < 80:
            continue
        buf.append(t)
        if len(buf) >= per_file:
            path = os.path.join(out_dir, f"wikipedia_{version}_{lang}_{file_idx}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(buf))
            written += len(buf)
            file_idx += 1
            buf = []
    if buf:
        path = os.path.join(out_dir, f"wikipedia_{version}_{lang}_{file_idx}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(buf))
        written += len(buf)

    print(f"Exported {written} paragraphs for {name} into {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Download HF wikimedia/wikipedia and export text seeds")
    ap.add_argument("--version", default="20231101", help="Wikipedia dump version, e.g., 20231101")
    ap.add_argument("--langs", nargs="+", default=["en", "zh"], help="Language codes to fetch (e.g., en zh)")
    ap.add_argument("--samples", type=int, default=5000, help="Rows per language to cache/export")
    ap.add_argument("--export", action="store_true", help="Export text files to out_dir")
    ap.add_argument("--per_file", type=int, default=200, help="Paragraphs per exported file")
    ap.add_argument("--out_dir", default="data/synthetic/raw_text", help="Export directory")
    args = ap.parse_args()

    rc = 0
    for lang in args.langs:
        rc = export_language(args.version, lang, args.samples, args.out_dir, args.per_file, args.export)
        if rc != 0:
            return rc
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

