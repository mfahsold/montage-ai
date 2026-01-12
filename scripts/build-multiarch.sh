#!/bin/bash
# DEPRECATED wrapper. Use `./scripts/build-distributed.sh` instead; this script delegates to it.
# Kept for convenience for legacy workflows.
set -euo pipefail

echo "WARNING: scripts/build-multiarch.sh is deprecated. Delegating to scripts/build-distributed.sh"

# Load centralized deployment config if present
if [ -f "deploy/config.env" ]; then
  # shellcheck disable=SC1091
  source "deploy/config.env"
fi

REGISTRY="${REGISTRY_URL:-${REGISTRY:-ghcr.io/mfahsold}}"
TAG="${TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

# Delegate to build-distributed.sh (preserves env vars)
REGISTRY="$REGISTRY" PLATFORMS="$PLATFORMS" TAG="$TAG" ./scripts/build-distributed.sh "$TAG"

echo "Done (delegated)."