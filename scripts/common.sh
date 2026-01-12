#!/usr/bin/env bash
# Common helper to load deploy/config.env and expose canonical variables for scripts
set -euo pipefail

CONFIG_ROOT="$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f "${CONFIG_ROOT}/deploy/config.env" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_ROOT}/deploy/config.env"
fi

# Derive a canonical REGISTRY_URL if not present
if [ -n "${REGISTRY_URL:-}" ]; then
  : # already set
else
  if [ -n "${REGISTRY_HOST:-}" ] && [ -n "${REGISTRY_PORT:-}" ]; then
    REGISTRY_URL="${REGISTRY_HOST}:${REGISTRY_PORT}"
  elif [ -n "${REGISTRY_HOST:-}" ]; then
    REGISTRY_URL="${REGISTRY_HOST}"
  else
    REGISTRY_URL="ghcr.io/mfahsold"
  fi
fi

export REGISTRY_URL
export UV_VERSION UV_IMAGE UV_CACHE_DIR 2>/dev/null || true
