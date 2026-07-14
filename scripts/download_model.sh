#!/usr/bin/env bash
# Download a GGUF model into ./models.
# Default: Qwen2.5-7B-Instruct Q4_K_M (~4.7 GB) — good quality/size for 24 GB.
set -euo pipefail

# bartowski's quants are reliable single-file GGUFs (the official Qwen repo
# splits some quants into multiple parts, which complicates loading).
REPO="${MODEL_REPO:-bartowski/Qwen2.5-7B-Instruct-GGUF}"
FILE="${MODEL_FILE:-Qwen2.5-7B-Instruct-Q4_K_M.gguf}"
DEST="models/${FILE}"

if [[ -f "$DEST" ]]; then
  echo "Model already present: $DEST"
  exit 0
fi

URL="https://huggingface.co/${REPO}/resolve/main/${FILE}?download=true"
echo "Downloading ${FILE} from ${REPO} ..."
echo "  -> ${DEST}"
curl -L --fail -o "$DEST" "$URL"
echo "Done. Size:"
du -h "$DEST"
