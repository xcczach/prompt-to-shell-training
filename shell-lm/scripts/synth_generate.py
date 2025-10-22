#!/usr/bin/env python
"""Teacher-driven synthesis from Hugging Face wikimedia/wikipedia seeds.

Generates JSONL with rows: {input, context, output_json}, enforcing OS/shell/language ratios.

Usage:
  python scripts/synth_generate.py --seeds data/synthetic/raw_text \
    --out data/synthetic/samples/synth_v1.jsonl \
    --target_counts '{"linux":45,"macos":25,"windows":30}' \
    [--teacher_endpoint http://localhost:8000/v1/chat/completions] \
    [--total 10000]

Notes:
- Always uses profiles when calling PowerShell elsewhere in the pipeline.
- By default, seeds are augmented by sampling neutral paragraphs from the
  Hugging Face 'wikimedia/wikipedia' dataset (English and Chinese splits).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

try:
    import httpx
except Exception:
    httpx = None  # Optional

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from common import (
    ALLOWED_TYPES,
    OS_RATIOS,
    SHELL_MIX,
    canonicalize_cmd,
    jsonl_write,
    looks_risky,
    stratified_quota_counts,
    validate_output_json,
)


SYSTEM_PROMPT = (
    "You generate practical, safe terminal tasks. Output a single JSON object with fields: "
    "type, cmd, requires_confirm, explain. Prefer read-only commands. Vary tools and difficulty."
)

USER_TEMPLATE = textwrap.dedent(
    """
    Context:
    {context}

    Constraints:
    - OS distribution target: Linux 45%, macOS 25%, Windows 30%.
    - Shell types: bash/zsh/fish on Linux and macOS; PowerShell/cmd on Windows (see ratios).
    - One single-line command (pipes ok), no multi-line scripts.
    - If operation could be destructive, set requires_confirm=true and prefer a safe variant.
    - Produce Chinese for half of tasks, English for the other half.
    - Output ONLY the JSON object.
    """
).strip()


def load_wikipedia_seeds(n_en: int, n_zh: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    if load_dataset is None:
        raise RuntimeError("datasets package required. pip install datasets")
    random.seed(seed)
    en = load_dataset("wikimedia/wikipedia", name="20231101.en", split="train")
    zh = load_dataset("wikimedia/wikipedia", name="20231101.zh", split="train")
    en_texts = []
    zh_texts = []
    for ex in en.shuffle(seed=seed).select(range(min(n_en * 5, len(en)))):
        t = ex.get("text") or ""
        t = t.strip()
        if len(t) > 200:  # prefer substantial paragraphs
            en_texts.append(t[:800])
        if len(en_texts) >= n_en:
            break
    for ex in zh.shuffle(seed=seed).select(range(min(n_zh * 5, len(zh)))):
        t = ex.get("text") or ""
        t = t.strip()
        if len(t) > 100:
            zh_texts.append(t[:800])
        if len(zh_texts) >= n_zh:
            break
    return en_texts, zh_texts


def read_seed_files(path: str) -> List[str]:
    texts: List[str] = []
    if os.path.isdir(path):
        for name in os.listdir(path):
            if name.lower().endswith(".txt"):
                with open(os.path.join(path, name), "r", encoding="utf-8") as f:
                    texts.append(f.read().strip())
    elif os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
    return [t for t in texts if t]


def paraphrases(text: str, lang: str, k: int = 2) -> List[str]:
    res = [text]
    # Simple, deterministic paraphrasing to avoid external deps
    if lang == "en":
        res.append(text.replace(" for ", " to "))
        res.append(text.replace(" using ", " with "))
    else:
        res.append(text.replace("例如", "比如"))
        res.append(text.replace("请", "请你"))
    return res[: max(1, k + 1)]


def pick_os_and_shell(quota: Dict[Tuple[str, str], int]) -> Tuple[str, str]:
    # Choose next (os,shell) bucket with remaining quota
    for (os_name, shell), cnt in quota.items():
        if cnt > 0:
            quota[(os_name, shell)] = cnt - 1
            return os_name, shell
    # Should not happen if quota computed correctly
    return "linux", "bash"


def teacher_call(endpoint: str, context_text: str) -> Optional[str]:
    """Call a teacher endpoint with OpenAI-compatible chat payload.
    Returns the raw string (expected to be a JSON object string), or None.
    """
    if httpx is None:
        return None
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context_text)},
        ]
        # Try OpenAI-compatible /chat/completions
        payload = {
            "model": "teacher",
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 256,
        }
        r = httpx.post(endpoint.rstrip("/") + "/chat/completions", json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "").strip()
            return content or None
    except Exception:
        return None
    return None


def stub_teacher(context_text: str, os_name: str, shell: str, lang: str) -> str:
    # Very conservative, read-only commands tailored by OS/shell
    explain_en = "List files or query system safely."
    explain_zh = "安全地列出文件或查询系统。"
    if os_name == "windows":
        if shell == "powershell":
            cmd = "Get-ChildItem -Force | Select-Object -First 10"
        else:
            cmd = "dir"
        explain = explain_zh if lang == "zh" else explain_en
        out = {
            "type": shell,
            "cmd": cmd,
            "requires_confirm": False,
            "explain": explain,
        }
        return json.dumps(out, ensure_ascii=False)
    # POSIX shells
    if shell == "zsh" or shell == "bash":
        cmd = "ls -la | head -n 10"
    elif shell == "fish":
        cmd = "ls -la | head -n 10"
    else:
        cmd = "echo 'Unsupported shell'"
    explain = explain_zh if lang == "zh" else explain_en
    out = {
        "type": shell,
        "cmd": cmd,
        "requires_confirm": False,
        "explain": explain,
    }
    return json.dumps(out, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesize NL->Shell training data with teacher LLM.")
    ap.add_argument("--seeds", required=True, help="Path to seed .txt files directory or single file")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--target_counts", default=json.dumps(OS_RATIOS), help="OS ratio JSON e.g. {\"linux\":45,\"macos\":25,\"windows\":30}")
    ap.add_argument("--teacher_endpoint", default=None, help="OpenAI-compatible teacher endpoint base URL (optional)")
    ap.add_argument("--total", type=int, default=10000, help="Total examples to generate (default 10000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    try:
        target_counts = json.loads(args.target_counts)
    except Exception as e:
        print(f"ERROR: invalid --target_counts JSON: {e}", file=sys.stderr)
        return 2
    for k in ("linux", "macos", "windows"):
        if k not in target_counts:
            print("ERROR: target_counts must include linux/macos/windows", file=sys.stderr)
            return 2

    # Prepare seeds: local + HF wikipedia (en/zh)
    local_seeds = read_seed_files(args.seeds)
    need = max(0, args.total - len(local_seeds))
    add_en = need // 2 + (need % 2)
    add_zh = need // 2
    try:
        en_texts, zh_texts = load_wikipedia_seeds(add_en, add_zh, seed=args.seed)
    except Exception as e:
        print(f"WARNING: failed to load HF wikimedia/wikipedia seeds: {e}", file=sys.stderr)
        en_texts, zh_texts = [], []

    # Construct language-balanced contexts
    en_pool = [t for t in local_seeds if t and all(ord(c) < 128 for c in t)] + en_texts
    zh_pool = [t for t in local_seeds if t and any(ord(c) > 127 for c in t)] + zh_texts
    if not en_pool:
        en_pool = ["Describe a dataset and list files safely."]
    if not zh_pool:
        zh_pool = ["描述一个数据集，并安全地列出文件。"]

    # Allocate per (os, shell) quotas
    per_bucket = stratified_quota_counts(args.total, target_counts, SHELL_MIX)
    # Maintain 50/50 language split by alternating
    lang_cycle = ["en", "zh"] * (args.total // 2 + 2)

    out_rows: List[dict] = []
    idx = 0
    while len(out_rows) < args.total:
        os_name, shell = pick_os_and_shell(per_bucket)
        lang = lang_cycle[idx % len(lang_cycle)]
        idx += 1
        # Select context text
        if lang == "en":
            base = random.choice(en_pool)
        else:
            base = random.choice(zh_pool)
        # Create paraphrases (2-3 total variants per instruction)
        variants = paraphrases(base, lang, k=2)
        for v in variants:
            if len(out_rows) >= args.total:
                break
            # Input instruction: we provide short NL instruction consistent with context
            input_text = v if len(v) < 400 else v[:400]
            # Attempt teacher call
            output_raw = None
            if args.teacher_endpoint:
                output_raw = teacher_call(args.teacher_endpoint, input_text)
            if not output_raw:
                output_raw = stub_teacher(input_text, os_name, shell, lang if lang in ("en", "zh") else "en")
            # Validate structure
            try:
                obj = json.loads(output_raw)
            except Exception:
                # Skip invalid
                continue
            ok, err = validate_output_json(obj)
            if not ok:
                continue
            # Enforce language on explain field
            # (light heuristic; not enforced strictly)
            # Enforce safety preference
            if looks_risky(obj["cmd"]) and obj.get("requires_confirm") is False:
                obj["requires_confirm"] = True
                output_raw = json.dumps(obj, ensure_ascii=False)
            row = {
                "input": input_text,
                "context": {"os": os_name, "shell": shell},
                "output_json": output_raw,
            }
            out_rows.append(row)

    jsonl_write(args.out, out_rows)
    # Post-check distribution
    counts = {}
    for r in out_rows:
        osn = r["context"]["os"]
        sh = r["context"]["shell"]
        counts[(osn, sh)] = counts.get((osn, sh), 0) + 1
    total = len(out_rows)
    summary = {f"{k[0]}:{k[1]}": round(v * 100 / total, 1) for k, v in counts.items()}
    print("Wrote:", args.out)
    print("Distribution % by (os:shell):", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

