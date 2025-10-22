#!/usr/bin/env python
"""Execute predicted commands in a Docker sandbox (read-only, no network).

Notes:
- If requires_confirm is true, skip execution and record status=unknown.
- Maps types to images:
    bash/zsh/fish -> ubuntu:22.04 (shell presence may vary)
    powershell    -> mcr.microsoft.com/powershell
    cmd           -> skipped (record not run)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List

from common import jsonl_read, jsonl_write


IMAGES = {
    "bash": "ubuntu:22.04",
    "zsh": "ubuntu:22.04",
    "fish": "ubuntu:22.04",
    "powershell": "mcr.microsoft.com/powershell:latest",
}


def docker_available() -> bool:
    return shutil.which("docker") is not None


def build_command(shell_type: str, cmd: str) -> List[str]:
    if shell_type in ("bash", "zsh", "fish"):
        sh = "/bin/bash" if shell_type == "bash" else (
            "/usr/bin/zsh" if shell_type == "zsh" else "/usr/bin/fish"
        )
        return [sh, "-lc", cmd]
    if shell_type == "powershell":
        # Load profile by default (no -NoProfile)
        return ["pwsh", "-Command", cmd]
    return ["/bin/sh", "-lc", cmd]


def run_in_container(shell_type: str, cmd: str, timeout_s: int, mem: str, cpus: str) -> Dict:
    if not docker_available():
        return {"skipped": True, "reason": "docker_missing"}
    if shell_type == "cmd":
        return {"skipped": True, "reason": "cmd_not_supported"}
    image = IMAGES.get(shell_type, "ubuntu:22.04")
    run_cmd = build_command(shell_type, cmd)
    # Create a temp dir and mount read-only
    with tempfile.TemporaryDirectory() as tmpd:
        docker_args = [
            "docker", "run", "--rm", "--network", "none",
            "--cpus", str(cpus), "--memory", mem,
            "-v", f"{tmpd}:/workspace:ro",
            image,
        ] + run_cmd
        t0 = time.monotonic()
        try:
            p = subprocess.run(docker_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
            dt = int((time.monotonic() - t0) * 1000)
            return {
                "skipped": False,
                "ok": p.returncode == 0,
                "exit_code": p.returncode,
                "stderr_head": p.stderr.decode(errors="ignore")[:400],
                "runtime_ms": dt,
            }
        except subprocess.TimeoutExpired:
            return {"skipped": False, "ok": False, "timeout": True}


def main() -> int:
    ap = argparse.ArgumentParser(description="Dry functional check in Docker sandbox")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--mem", default="512m")
    ap.add_argument("--cpus", default="1.0")
    args = ap.parse_args()

    preds = list(jsonl_read(args.preds))
    if not preds:
        print("ERROR: empty preds", file=sys.stderr)
        return 2

    results: List[dict] = []
    for p in preds:
        obj = p.get("obj") or {}
        t = (obj.get("type") or "").lower()
        cmd = (obj.get("cmd") or "").strip()
        req = bool(obj.get("requires_confirm") is True)
        if not t or not cmd:
            results.append({"i": p.get("i"), "skipped": True, "reason": "missing_fields"})
            continue
        if req:
            results.append({"i": p.get("i"), "skipped": True, "reason": "requires_confirm"})
            continue
        r = run_in_container(t, cmd, args.timeout, args.mem, args.cpus)
        r["i"] = p.get("i")
        results.append(r)

    jsonl_write(args.out, results)
    print(json.dumps({
        "total": len(results),
        "runnable": sum(1 for r in results if not r.get("skipped")),
        "ok": sum(1 for r in results if r.get("ok")),
        "out": args.out,
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

