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
  if ! uv sync --locked --all-extras --dev; then
    echo "Warning: locked sync with extras failed. Falling back to locked sync without extras."
    uv sync --locked --dev || { echo "uv sync --locked --dev failed; aborting."; exit 1; }
  fi
else
  echo "No uv.lock found — performing best-effort sync"
  if ! uv sync --all-extras --dev; then
    echo "Warning: full sync with extras failed (likely optional/private extras). Falling back to sync without extras."
    if ! uv sync --dev; then
      echo "Warning: uv sync --dev failed as well. Proceeding to run tests without syncing dependencies (best-effort)."
      SKIP_SYNC=1
    fi
  fi
fi

# Run test suite
if [ "${SKIP_SYNC:-0}" = "1" ]; then
  echo "Running tests directly (no uv environment available)"
  PYTHONPATH=src pytest -q
else
  echo "Running tests via uv"
  uv run pytest -q
fi

echo "Local CI complete"
