#!/usr/bin/env bash
# Start llama-server with an OpenAI-compatible API on Apple Silicon (Metal).
set -euo pipefail

MODEL="${MODEL_PATH:-models/Qwen2.5-7B-Instruct-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"          # context window
NGL="${NGL:-999}"          # offload all layers to GPU (Metal)
PARALLEL="${PARALLEL:-8}"  # max concurrent request slots (continuous batching)

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL"
  echo "Run ./scripts/download_model.sh first."
  exit 1
fi

echo "Serving $MODEL on http://localhost:${PORT}"
echo "  ctx=${CTX}  n-gpu-layers=${NGL}  parallel-slots=${PARALLEL}"
echo

exec llama-server \
  --model "$MODEL" \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" \
  --parallel "$PARALLEL" \
  --cont-batching \
  --metrics            # expose Prometheus /metrics (useful for M5 dashboard)
