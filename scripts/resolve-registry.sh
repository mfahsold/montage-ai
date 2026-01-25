#!/usr/bin/env bash
# Print resolved REGISTRY_URL (uses deploy/k3s/config-global.yaml when present)
set -euo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
CONFIG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_GLOBAL="${CONFIG_GLOBAL:-${CONFIG_ROOT}/deploy/k3s/config-global.yaml}"
CONFIG_ENV="${CONFIG_ENV:-${CONFIG_ROOT}/deploy/k3s/base/cluster-config.env}"
CONFIG_LEGACY="${CONFIG_LEGACY:-${CONFIG_ROOT}/deploy/config.env}"
CONFIG_LIB="${CONFIG_ROOT}/scripts/ops/lib/config_global.sh"
REGISTRY_URL=""
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
  # Default to internal cluster registry
  echo "registry.registry.svc.cluster.local:5000"
fi
