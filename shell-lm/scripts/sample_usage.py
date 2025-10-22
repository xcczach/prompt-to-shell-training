#!/usr/bin/env python
"""Sample usage script for the trained (fine-tuned) model via vLLM.

This script sends a single natural-language instruction to an OpenAI-compatible
endpoint (e.g., vLLM serving the fine-tuned Qwen model) and prints the model's
JSON response along with a parsed summary.

Example:
  # Assuming finetuned endpoint on 8001
  python scripts/sample_usage.py \
    --url http://localhost:8001/v1 \
    --prompt "List only hidden files in the current directory (Linux bash)."

  # With context hints
  python scripts/sample_usage.py --url http://localhost:8001/v1 \
    --prompt "Show Homebrew packages" --os macos --shell zsh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

try:
    import httpx
except Exception:
    httpx = None


SYSTEM = "You convert natural language into safe, single-line shell commands in JSON."


def call_openai_chat(url_base: str, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    if httpx is None:
        raise RuntimeError("httpx is required. pip install httpx")
    payload = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
    }
    resp = httpx.post(url_base.rstrip("/") + "/chat/completions", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample usage against a served fine-tuned model")
    ap.add_argument("--url", default="http://localhost:8001/v1", help="OpenAI-compatible base URL (finetuned endpoint)")
    ap.add_argument("--prompt", default=None, help="Natural language instruction. If omitted, read from stdin.")
    ap.add_argument("--os", dest="os_name", default=None, choices=["linux", "macos", "windows"], help="Optional OS hint")
    ap.add_argument("--shell", dest="shell", default=None, choices=["bash", "zsh", "fish", "powershell", "cmd", "nushell"], help="Optional shell hint")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    if args.prompt is None:
        print("Enter instruction (Ctrl-D/Ctrl-Z to end):", file=sys.stderr)
        args.prompt = sys.stdin.read().strip()
    if not args.prompt:
        print("ERROR: empty prompt", file=sys.stderr)
        return 2

    prompt = args.prompt
    if args.os_name or args.shell:
        hint_os = args.os_name or "?"
        hint_sh = args.shell or "?"
        prompt = f"{prompt}\n\n(Hint: OS={hint_os} shell={hint_sh})"

    try:
        raw = call_openai_chat(args.url, prompt, args.temperature, args.top_p, args.max_new_tokens)
    except Exception as e:
        print(f"ERROR: request failed: {e}", file=sys.stderr)
        return 3

    print("=== Raw Model Output ===")
    print(raw)
    print()
    try:
        obj = json.loads(raw)
        # Basic schema check
        t = obj.get("type"); cmd = obj.get("cmd"); rc = obj.get("requires_confirm"); exp = obj.get("explain")
        ok = isinstance(t, str) and isinstance(cmd, str) and isinstance(rc, bool) and isinstance(exp, str)
        print("=== Parsed ===")
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        if ok:
            print()
            print(f"Type: {t}")
            print(f"Command: {cmd}")
            print(f"Requires Confirm: {rc}")
        else:
            print("WARNING: Parsed JSON missing required fields.")
    except Exception:
        print("WARNING: Output is not valid JSON.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

