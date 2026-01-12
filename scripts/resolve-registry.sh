#!/usr/bin/env bash
# Print resolved REGISTRY_URL (uses deploy/config.env when present)
set -euo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
CONFIG_ROOT="${SCRIPT_DIR}/.."
REGISTRY_URL=""
if [ -f "${CONFIG_ROOT}/deploy/config.env" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_ROOT}/deploy/config.env"
fi
if [ -n "${REGISTRY_URL:-}" ]; then
  echo "${REGISTRY_URL}"
  exit 0
fi
# Fallback to host:port or ghcr
if [ -n "${REGISTRY_HOST:-}" ] && [ -n "${REGISTRY_PORT:-}" ]; then
  echo "${REGISTRY_HOST}:${REGISTRY_PORT}"
elif [ -n "${REGISTRY_HOST:-}" ]; then
  echo "${REGISTRY_HOST}"
else
  echo "ghcr.io/mfahsold"
fi