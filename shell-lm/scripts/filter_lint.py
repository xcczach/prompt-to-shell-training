#!/usr/bin/env python
"""Filter, lint, rebalance, and deduplicate synthesized data.

Usage:
  python scripts/filter_lint.py --in data/synthetic/samples/synth_v1.jsonl \
    --out data/synthetic/samples/synth_v1.clean.jsonl

Behavior:
- Hard-ban risky patterns.
- Lint by shell (shellcheck; PowerShell .NET Parser; cmd heuristics).
- Canonicalize cmd, deduplicate.
- Rebalance to target OS/shell/language ratios if drifted.

Notes:
- PowerShell validation loads profile (no -NoProfile) in line with user instruction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from common import (
    ALLOWED_TYPES,
    OS_RATIOS,
    SHELL_MIX,
    canonicalize_cmd,
    family_from_cmd,
    jsonl_read,
    jsonl_write,
    looks_risky,
    stratified_quota_counts,
    validate_output_json,
    which_powershell,
)


def has_shellcheck() -> bool:
    return shutil.which("shellcheck") is not None


def lint_posix(cmd: str, shell: str) -> Tuple[bool, str]:
    if not has_shellcheck():
        return True, "shellcheck not found; skipping"
    # Feed command via stdin
    sc = shutil.which("shellcheck")
    # shellcheck -s bash/zsh can be set; fish has limited support
    sh_map = {"bash": "bash", "zsh": "bash", "fish": "bash"}
    sh = sh_map.get(shell, "bash")
    p = subprocess.run([sc, "-s", sh, "-S", "error", "-"], input=cmd.encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ok = p.returncode == 0
    msg = p.stdout.decode(errors="ignore")
    return ok, msg.strip()


def lint_powershell(cmd: str) -> Tuple[bool, str]:
    ps = which_powershell()
    if not ps:
        return True, "powershell not found; skipping"
    # Use .NET Parser to only parse, not execute the command
    script = (
        "$ErrorActionPreference='Stop';"
        "$null=[ref]$null;"
        f"[void][System.Management.Automation.Language.Parser]::ParseInput(\"{cmd.replace('\\', r'\\').replace('"', r'\"')}\", [ref]$null, [ref]$null);"
    )
    p = subprocess.run([ps, "-Command", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ok = p.returncode == 0
    return ok, p.stdout.decode(errors="ignore").strip()


def lint_cmd(cmd: str) -> Tuple[bool, str]:
    # Heuristic: reject multi-line; warn if dangerous meta chars
    if "\n" in cmd or "\r" in cmd:
        return False, "cmd contains newline"
    # Allow pipe and && but warn
    if re.search(r"\b(del|format|chkdsk|diskpart)\b", cmd.lower()):
        return False, "cmd potentially dangerous"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter + lint + dedup + rebalance JSONL dataset")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--target_counts", default=json.dumps(OS_RATIOS))
    args = ap.parse_args()

    try:
        target_counts = json.loads(args.target_counts)
    except Exception as e:
        print(f"ERROR: invalid --target_counts JSON: {e}", file=sys.stderr)
        return 2

    rows: List[dict] = list(jsonl_read(args.inp))
    n_in = len(rows)
    kept: List[dict] = []

    seen: set = set()  # canonicalized commands to dedup
    rejected: Dict[str, int] = {}

    def rej(reason: str):
        rejected[reason] = rejected.get(reason, 0) + 1

    for r in rows:
        # Parse output_json
        try:
            obj = json.loads(r["output_json"]) if isinstance(r.get("output_json"), str) else None
        except Exception:
            rej("invalid_json")
            continue
        ok, err = validate_output_json(obj or {})
        if not ok:
            rej(f"schema:{err}")
            continue
        t = obj["type"].lower()
        cmd = obj["cmd"].strip()
        # Hard-ban risky
        if looks_risky(cmd):
            rej("risky_pattern")
            continue
        # Lint per shell type
        if t in ("bash", "zsh", "fish"):
            ok, msg = lint_posix(cmd, t)
            if not ok:
                rej("shellcheck_error")
                continue
        elif t == "powershell":
            ok, msg = lint_powershell(cmd)
            if not ok:
                rej("powershell_syntax_error")
                continue
        elif t == "cmd":
            ok, msg = lint_cmd(cmd)
            if not ok:
                rej("cmd_invalid")
                continue
        # Canonicalize and dedup
        canon = canonicalize_cmd(cmd)
        key = (t, canon)
        if key in seen:
            rej("duplicate")
            continue
        seen.add(key)
        # Normalize object with canonical cmd
        obj["cmd"] = canon
        r["output_json"] = json.dumps(obj, ensure_ascii=False)
        kept.append(r)

    if not kept:
        print("ERROR: no valid rows after filtering", file=sys.stderr)
        return 3

    # Rebalance to OS/shell targets
    total = len(kept)
    quota = stratified_quota_counts(total, target_counts, SHELL_MIX)
    buckets: Dict[Tuple[str, str], List[dict]] = {}
    for r in kept:
        osn = r.get("context", {}).get("os")
        sh = r.get("context", {}).get("shell")
        if not osn or not sh:
            osn, sh = "linux", "bash"
        buckets.setdefault((osn, sh), []).append(r)

    # Sample per bucket respecting quotas
    out_rows: List[dict] = []
    for k, q in quota.items():
        pool = buckets.get(k, [])
        if not pool:
            continue
        if len(pool) <= q:
            out_rows.extend(pool)
        else:
            out_rows.extend(random_sample(pool, q))

    # Final write
    jsonl_write(args.out, out_rows)
    # Logs
    print(json.dumps({
        "input_rows": n_in,
        "kept": len(kept),
        "written": len(out_rows),
        "rejected": rejected,
    }, ensure_ascii=False))
    return 0


def random_sample(items: List[dict], k: int) -> List[dict]:
    import random
    if k <= 0:
        return []
    if k >= len(items):
        return items
    return random.sample(items, k)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

