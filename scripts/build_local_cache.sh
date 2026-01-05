#!/bin/bash
# =============================================================================
# Montage AI - Multi-Arch Build mit LOCAL Cache
# =============================================================================
# Nutzt lokales Filesystem für Build-Cache (schneller, aber nicht geteilt)
# =============================================================================

set -euo pipefail

CACHE_DIR="${CACHE_DIR:-/tmp/buildx-cache}"
IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILDER="${BUILDER:-multiarch-builder}"
TAG="${TAG:-latest}"
REGISTRY="${REGISTRY:-localhost:5000}"

# Farben
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Montage AI Multi-Arch Build (Local Cache) ===${NC}"
echo "Cache: $CACHE_DIR"
echo "Image: $IMAGE_NAME:$TAG"
echo "Platforms: $PLATFORMS"
echo

# Erstelle Cache-Dir
mkdir -p "$CACHE_DIR"

# Build
docker buildx build \
  --builder "$BUILDER" \
  --platform "$PLATFORMS" \
  --cache-from "type=local,src=$CACHE_DIR" \
  --cache-to "type=local,dest=$CACHE_DIR,mode=max" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg "GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo 'dev')" \
  -t "$REGISTRY/$IMAGE_NAME:$TAG" \
  --push \
  .

echo
echo -e "${GREEN}✓ Build complete!${NC}"
echo "Cache: $CACHE_DIR ($(du -sh $CACHE_DIR | cut -f1))"
