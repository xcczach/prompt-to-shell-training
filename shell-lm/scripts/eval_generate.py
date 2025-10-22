#!/usr/bin/env python
"""Query both vLLM endpoints on test split and collect predictions.

Outputs:
  preds_base.jsonl and preds_ft.jsonl in the specified out_dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

try:
    import httpx
except Exception:
    httpx = None

from common import jsonl_read, jsonl_write


SYSTEM = "You convert natural language into safe, single-line shell commands in JSON."


def call_openai_chat(url: str, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    if httpx is None:
        return ""
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
    r = httpx.post(url.rstrip("/") + "/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content


def try_parse_json(s: str) -> (bool, dict | None):
    s = s.strip()
    try:
        return True, json.loads(s)
    except Exception:
        # attempt close brace fix (but mark invalid)
        if not s.endswith("}"):
            try:
                s2 = s + "}"
                obj = json.loads(s2)
                return False, obj
            except Exception:
                pass
        return False, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run eval generation against two endpoints")
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--ft_url", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_rows = list(jsonl_read(args.inp))
    if not test_rows:
        print("ERROR: test split empty", file=sys.stderr)
        return 2

    base_preds: List[dict] = []
    ft_preds: List[dict] = []

    for i, r in enumerate(test_rows):
        prompt = r.get("input", "")
        # include simple context hints if present
        ctx = r.get("context")
        if ctx and ctx.get("os") and ctx.get("shell"):
            prompt = f"{prompt}\n\n(Hint: OS={ctx['os']} shell={ctx['shell']})"
        try:
            b_out = call_openai_chat(args.base_url, prompt, args.temperature, args.top_p, args.max_new_tokens)
        except Exception as e:
            b_out = ""
        try:
            f_out = call_openai_chat(args.ft_url, prompt, args.temperature, args.top_p, args.max_new_tokens)
        except Exception as e:
            f_out = ""
        # Parse JSON strictly
        b_ok, b_obj = try_parse_json(b_out)
        f_ok, f_obj = try_parse_json(f_out)
        base_preds.append({
            "i": i,
            "input": r.get("input"),
            "raw": b_out,
            "json_valid": bool(b_ok and b_obj is not None),
            "obj": b_obj,
        })
        ft_preds.append({
            "i": i,
            "input": r.get("input"),
            "raw": f_out,
            "json_valid": bool(f_ok and f_obj is not None),
            "obj": f_obj,
        })

    base_path = os.path.join(args.out_dir, "preds_base.jsonl")
    ft_path = os.path.join(args.out_dir, "preds_ft.jsonl")
    jsonl_write(base_path, base_preds)
    jsonl_write(ft_path, ft_preds)

    print(json.dumps({
        "attempted": len(test_rows),
        "base_valid": sum(1 for p in base_preds if p["json_valid"]),
        "ft_valid": sum(1 for p in ft_preds if p["json_valid"]),
        "base_path": base_path,
        "ft_path": ft_path,
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

