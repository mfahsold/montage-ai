#!/usr/bin/env bash
set -euo pipefail

# Resolve image name from deploy/k3s/config-global.yaml
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_GLOBAL="${CONFIG_GLOBAL:-${ROOT_DIR}/deploy/k3s/config-global.yaml}"
CONFIG_ENV="${CONFIG_ENV:-${ROOT_DIR}/deploy/k3s/base/cluster-config.env}"
CONFIG_LIB="${ROOT_DIR}/scripts/ops/lib/config_global.sh"

if [ -f "$CONFIG_LIB" ]; then
  # shellcheck source=/dev/null
  source "$CONFIG_LIB"
fi
if command -v config_global_export >/dev/null 2>&1; then
  eval "$(config_global_export "${CONFIG_GLOBAL}")"
fi
if [ -f "$CONFIG_ENV" ]; then
  # shellcheck source=/dev/null
  source "$CONFIG_ENV"
fi

REGISTRY_URL="${REGISTRY_URL:-registry.registry.svc.cluster.local:5000}"
IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_FULL="${IMAGE_FULL:-${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}}"

echo "${IMAGE_FULL}"
