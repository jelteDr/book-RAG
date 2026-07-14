#!/usr/bin/env bash
# Install llama.cpp (Metal-accelerated on Apple Silicon) via Homebrew.
set -euo pipefail

if command -v llama-server >/dev/null 2>&1; then
  echo "llama-server already installed: $(command -v llama-server)"
else
  echo "Installing llama.cpp via Homebrew..."
  brew install llama.cpp
fi

echo
echo "llama-server version:"
llama-server --version 2>&1 | head -n 3 || true
