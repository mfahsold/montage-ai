#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh or pipx install uv"
  exit 1
fi

echo "Generating uv.lock..."
uv lock

echo "Done. Please review and commit 'uv.lock' to repository if satisfied."