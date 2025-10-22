shell-lm: Natural Language to Safe Shell Command Training Pipeline

Overview
- End-to-end pipeline to fine-tune a small LLM (Qwen2.5-0.5B(-Instruct)) that converts natural language into safe, single-line shell commands in a strict JSON schema.
- Implements data synthesis (real + synthetic), filtering/linting, family-aware splits, Unsloth QLoRA training (FSDP multi-GPU), vLLM serving for base/finetuned, and evaluation (structure/text/functional/safety).

Key Requirements
- Output JSON schema: {type, cmd, requires_confirm, explain}.
- Language: 50% Chinese / 50% English.
- OS mix: Linux 45%, macOS 25%, Windows 30%.
- Shell mix per OS: Linux (bash 70%, zsh 20%, fish 10%), macOS (zsh 80%, bash 20%), Windows (PowerShell 80%, cmd 20%).
- Safety filters + linting (shellcheck, PowerShell parser, cmd heuristics). Deduplicate by canonicalized command.
- Always use profile when invoking PowerShell (no -NoProfile).

Dependencies (suggested)
- Python >= 3.10
- pip install: datasets, httpx, pyyaml, tqdm, numpy, pandas, scikit-learn, rapidfuzz, docker, jsonschema, rich, uvloop (optional), unsloth, transformers, accelerate, peft
- CLI tools (optional but recommended): shellcheck, docker, vllm, git, pwsh or Windows PowerShell

Quick Start
- 1) Generate synthetic data from Hugging Face wikimedia/wikipedia seeds, enforce ratios
  # (Optional) Pre-download and export Wikipedia seeds
  python scripts/download_wikipedia.py --version 20231101 --langs en zh \
    --samples 5000 --export --per_file 200 --out_dir data/synthetic/raw_text

  python scripts/synth_generate.py --seeds data/synthetic/raw_text \
    --out data/synthetic/samples/synth_v1.jsonl \
    --target_counts '{"linux":45,"macos":25,"windows":30}'

- 2) Filter, lint, rebalance, dedup
  python scripts/filter_lint.py --in data/synthetic/samples/synth_v1.jsonl \
    --out data/synthetic/samples/synth_v1.clean.jsonl

- 3) Split by task family
  python scripts/split_dataset.py --in data/synthetic/samples/synth_v1.clean.jsonl

- 4) Train with Unsloth (multi-GPU via FSDP)
  python scripts/train_unsloth.py --config configs/train_sft.yaml

- 5) Serve base & finetuned with vLLM (OpenAI-compatible API)
  bash scripts/serve_vllm_base.sh
  bash scripts/serve_vllm_ft.sh

- 6) Evaluate (generate predictions)
  python scripts/eval_generate.py --base_url http://localhost:8000 --ft_url http://localhost:8001 \
    --in data/splits/test.jsonl --out_dir outputs/eval_run1

- 7) Sandboxed functional checks (read-only; skips risky)
  python scripts/exec_sandbox.py --preds outputs/eval_run1/preds_ft.jsonl --out outputs/eval_run1/exec_ft.jsonl
  python scripts/exec_sandbox.py --preds outputs/eval_run1/preds_base.jsonl --out outputs/eval_run1/exec_base.jsonl

- 8) Score & report
  python scripts/eval_score.py --gold data/splits/test.jsonl \
    --base outputs/eval_run1 --ft outputs/eval_run1

- 7) Sample usage (interactive prompt → JSON command)
  python scripts/sample_usage.py --url http://localhost:8001/v1 \
    --prompt "List only hidden files in the current directory (Linux bash)."
  # With hints
  python scripts/sample_usage.py --url http://localhost:8001/v1 \
    --prompt "Show Homebrew packages" --os macos --shell zsh

Notes
- Synthesis pulls neutral paragraphs from the Hugging Face wikimedia/wikipedia dataset for both English and Chinese seeds.
- PowerShell linting/validation uses the .NET Parser via pwsh/powershell with profiles loaded (no -NoProfile).
- Docker sandbox is read-only and no network by default; set images per shell type.
