#!/usr/bin/env bash
set -euo pipefail

# Resolve image name from shared config
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/scripts/common.sh"

IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_FULL="${IMAGE_FULL:-}"
if [ -z "${IMAGE_FULL}" ]; then
  if [ -n "${REGISTRY_URL:-}" ]; then
    IMAGE_FULL="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
  else
    IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"
  fi
fi

echo "${IMAGE_FULL}"
