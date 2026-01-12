#!/usr/bin/env bash
set -euo pipefail

# Resolve image name from deploy/config.env
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$ROOT_DIR/deploy/config.env" ]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/deploy/config.env"
fi

# IMAGE_FULL expected in deploy/config.env
echo "${IMAGE_FULL:-${REGISTRY_URL:-ghcr.io/mfahsold}/${IMAGE_NAME:-montage-ai}:${IMAGE_TAG:-latest}}"