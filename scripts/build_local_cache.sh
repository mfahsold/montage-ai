#!/bin/bash
# =============================================================================
# Local build with filesystem cache - Fast iteration (DRY/KISS)
# =============================================================================
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-/tmp/buildx-cache}"
TAG="${TAG:-dev}"
PLATFORM="${PLATFORM:-linux/amd64}"
BUILDER="${BUILDER:-multiarch-builder}"
LOAD="${LOAD:-false}"
GIT_COMMIT=$(git rev-parse --short=8 HEAD 2>/dev/null || echo "dev")

echo "=== Local Build ==="
echo "Cache: $CACHE_DIR | Tag: $TAG | Platform: $PLATFORM"
echo ""

mkdir -p "$CACHE_DIR"

BUILD_CMD="docker buildx build \
  --builder $BUILDER \
  --platform $PLATFORM \
  --cache-from type=local,src=$CACHE_DIR \
  --cache-to type=local,dest=$CACHE_DIR,mode=max \
  --build-arg GIT_COMMIT=$GIT_COMMIT \
  -t montage-ai:$TAG"

[ "$LOAD" = "true" ] && BUILD_CMD="$BUILD_CMD --load"
BUILD_CMD="$BUILD_CMD ."

eval "$BUILD_CMD"

echo ""
echo "✓ montage-ai:$TAG ($(du -sh $CACHE_DIR 2>/dev/null | cut -f1) cached)"
