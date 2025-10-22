#!/usr/bin/env bash
set -euo pipefail

# Serve base model via vLLM OpenAI-compatible API.
# Env:
#   BASE_PORT=8000
#   BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
#   TP=1  # tensor parallel degree

: "${BASE_PORT:=8000}"
: "${BASE_MODEL:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${TP:=1}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --port "${BASE_PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len 4096 \
  --trust-remote-code

