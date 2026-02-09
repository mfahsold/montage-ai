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

# If the project declares private extras (e.g., `cloud-private`), temporarily remove them
# to avoid failures when a private index is not available. Developers can opt-in
# by setting INCLUDE_PRIVATE_EXTRAS=1 in their environment.
PYPROJECT_BACKUP=""
if grep -q "^cloud-private = " pyproject.toml && [ "${INCLUDE_PRIVATE_EXTRAS:-}" != "1" ]; then
  echo "⚠️  Detected private extras (cloud-private) in pyproject.toml — temporarily removing for public CI."
  PYPROJECT_BACKUP="$(mktemp)"
  cp pyproject.toml "$PYPROJECT_BACKUP"
  # Remove cloud-private section using sed (works cross-platform)
  sed '/^cloud-private = \[/,/^]/d' "$PYPROJECT_BACKUP" > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml
  echo "   (will restore after sync)"
fi

# Prefer locked sync if uv.lock exists
SKIP_SYNC=0
if [ -f uv.lock ]; then
  echo "Found uv.lock — performing locked sync"
  if ! uv sync --locked --all-extras --dev; then
    echo "Warning: locked sync with extras failed. Falling back to locked sync without extras."
    uv sync --locked --dev || { echo "uv sync --locked --dev failed; aborting."; exit 1; }
  fi
else
  echo "No uv.lock found — performing best-effort sync"
  if ! uv sync --dev --extra test 2>&1 | tee /tmp/uv-sync.log; then
    echo "⚠️  uv sync failed; see /tmp/uv-sync.log for details"
    SKIP_SYNC=1
  fi
fi

# Restore pyproject.toml if we backed it up
if [ -n "$PYPROJECT_BACKUP" ] && [ -f "$PYPROJECT_BACKUP" ]; then
  mv "$PYPROJECT_BACKUP" pyproject.toml
  echo "✅ Restored pyproject.toml"
fi

# Run test suite
if [ "${SKIP_SYNC:-0}" = "1" ]; then
  echo "Running tests directly (no uv environment available)"
  PYTHONPATH=src pytest -q
else
  echo "Running tests via uv"
  # If cloud-private was temporarily removed, we need to remove it again before uv run
  # (since uv run will also try to resolve the full pyproject.toml)
  PYPROJECT_BACKUP=""
  if grep -q "^cloud-private = " pyproject.toml && [ "${INCLUDE_PRIVATE_EXTRAS:-}" != "1" ]; then
    PYPROJECT_BACKUP="$(mktemp)"
    cp pyproject.toml "$PYPROJECT_BACKUP"
    sed '/^cloud-private = \[/,/^]/d' "$PYPROJECT_BACKUP" > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml
  fi

  PYTEST_EXIT=0
  uv run pytest -q || PYTEST_EXIT=$?

  # Restore if needed
  if [ -n "$PYPROJECT_BACKUP" ] && [ -f "$PYPROJECT_BACKUP" ]; then
    mv "$PYPROJECT_BACKUP" pyproject.toml
  fi

  if [ "$PYTEST_EXIT" -ne 0 ]; then
    echo "Local CI complete (pytest exited with $PYTEST_EXIT)"
    exit "$PYTEST_EXIT"
  fi
fi

echo "Local CI complete"
