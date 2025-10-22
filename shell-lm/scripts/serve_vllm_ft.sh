#!/usr/bin/env bash
set -euo pipefail

# Serve finetuned model via vLLM. Load base + LoRA adapter from outputs dir.
# Env:
#   FT_PORT=8001
#   BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
#   ADAPTER_DIR=outputs/qwen0.5b-shell
#   TP=1

: "${FT_PORT:=8001}"
: "${BASE_MODEL:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${ADAPTER_DIR:=outputs/qwen0.5b-shell}"
: "${TP:=1}"

# vLLM LoRA argument: --lora-modules "default=/path/to/adapter"
exec python -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --port "${FT_PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len 4096 \
  --trust-remote-code \
  --lora-modules "default=${ADAPTER_DIR}"

