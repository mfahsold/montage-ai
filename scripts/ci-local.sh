#!/usr/bin/env bash
set -euo pipefail

# Local CI script: installs uv (if missing), sets up Python and dependencies via uv, then runs tests.
# Designed to run on developer machines or self-hosted runners to avoid GitHub Actions costs.

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ]; then
  echo "uv not found; installing via pipx (preferred) or pip..."
  if command -v pipx >/dev/null 2>&1; then
    pipx install uv || { echo "pipx install failed; try pip install uv"; exit 1; }
  else
    pip install --user uv || { echo "pip install uv failed; please install uv manually"; exit 1; }
  fi
fi

# Use UV_VERSION from deploy/config.env if set
if [ -f deploy/config.env ]; then
  source deploy/config.env || true
fi

echo "Using uv (version: $(uv --version 2>/dev/null || echo 'unknown'))"

echo "Installing Python runtime (uv python install)"
uv python install || true

# Prefer locked sync if uv.lock exists
if [ -f uv.lock ]; then
  echo "Found uv.lock — performing locked sync"
  uv sync --locked --all-extras --dev
else
  echo "No uv.lock found — performing best-effort sync"
  uv sync --all-extras --dev
fi

# Run test suite
echo "Running tests via uv"
uv run pytest -q

echo "Local CI complete"
