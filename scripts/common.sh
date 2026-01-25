#!/usr/bin/env bash
# Common helper to load deploy/k3s/config-global.yaml and expose canonical variables for scripts
set -euo pipefail

CONFIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_GLOBAL="${CONFIG_GLOBAL:-${CONFIG_ROOT}/deploy/k3s/config-global.yaml}"
CONFIG_ENV="${CONFIG_ENV:-${CONFIG_ROOT}/deploy/k3s/base/cluster-config.env}"
CONFIG_LEGACY="${CONFIG_LEGACY:-${CONFIG_ROOT}/deploy/config.env}"
CONFIG_LIB="${CONFIG_ROOT}/scripts/ops/lib/config_global.sh"

if [ -f "${CONFIG_LIB}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_LIB}"
fi

if command -v config_global_export >/dev/null 2>&1; then
  eval "$(config_global_export "${CONFIG_GLOBAL}")"
fi

if [ -f "${CONFIG_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_ENV}"
fi

if [ -f "${CONFIG_LEGACY}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_LEGACY}"
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
    # Default to internal registry service
    REGISTRY_URL="registry.registry.svc.cluster.local:5000"
  fi
fi

if [ -z "${IMAGE_FULL:-}" ] && [ -n "${REGISTRY_URL:-}" ]; then
  IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
  IMAGE_TAG="${IMAGE_TAG:-latest-amd64}"
  IMAGE_FULL="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

export REGISTRY_URL
export UV_VERSION UV_IMAGE UV_CACHE_DIR 2>/dev/null || true
