#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh or pipx install uv"
  exit 1
fi

echo "Generating uv.lock..."
if uv lock; then
  echo "Done. 'uv.lock' generated. Please review and commit 'uv.lock' to repository if satisfied."
else
  echo "Warning: 'uv lock' failed. This is often due to optional extras being unavailable in the index (e.g., 'cgpu')."
  echo "Options to proceed:"
  echo "  - Generate lock locally after installing optional/private dependencies and commit 'uv.lock'."
  echo "  - Run 'uv lock --index <private-index>' if using private registries."
  echo "  - Edit pyproject.toml to temporarily remove problematic extras and retry 'uv lock'."
  exit 1
fi