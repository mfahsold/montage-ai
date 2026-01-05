#!/bin/bash
# =============================================================================
# Montage AI - Multi-Arch Build mit Registry Cache
# =============================================================================
# Nutzt die lokale Registry für Build-Cache (montage-ai-registry)
# =============================================================================

set -euo pipefail

REGISTRY="${REGISTRY:-registry:5000}"
IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILDER="${BUILDER:-multiarch-builder}"
PUSH="${PUSH:-false}"
TAG="${TAG:-latest}"

# Farben
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Montage AI Multi-Arch Build ===${NC}"
echo "Registry: $REGISTRY"
echo "Image: $IMAGE_NAME:$TAG"
echo "Platforms: $PLATFORMS"
echo "Builder: $BUILDER"
echo

# Prüfe Builder
if ! docker buildx inspect "$BUILDER" &>/dev/null; then
    echo -e "${YELLOW}Builder '$BUILDER' nicht gefunden, erstelle...${NC}"
    docker buildx create --name "$BUILDER" --driver docker-container --bootstrap
fi

# Build mit Cache
echo -e "${GREEN}Building mit Registry Cache...${NC}"

# Für insecure registry (HTTP statt HTTPS)
export BUILDKIT_INSECURE_REGISTRIES="${REGISTRY}"

BUILD_ARGS=(
    "build"
    "--builder" "$BUILDER"
    "--platform" "$PLATFORMS"
    "--cache-from" "type=registry,ref=${REGISTRY}/${IMAGE_NAME}:buildcache"
    "--cache-to" "type=registry,ref=${REGISTRY}/${IMAGE_NAME}:buildcache,mode=max"
    "--build-arg" "BUILDKIT_INLINE_CACHE=1"
    "--build-arg" "GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo 'dev')"
    "-t" "${REGISTRY}/${IMAGE_NAME}:${TAG}"
)

# Tags hinzufügen
if [ "$TAG" == "latest" ]; then
    BUILD_ARGS+=("-t" "${REGISTRY}/${IMAGE_NAME}:$(date +%Y%m%d)")
fi

# Push oder Load
if [ "$PUSH" == "true" ]; then
    BUILD_ARGS+=("--push")
    echo -e "${GREEN}Pushing to registry after build${NC}"
else
    # Multi-arch requires --push, can't use --load
    if [[ "$PLATFORMS" == *","* ]]; then
        echo -e "${YELLOW}Multi-platform build detected, forcing --push (--load not supported)${NC}"
        BUILD_ARGS+=("--push")
    else
        BUILD_ARGS+=("--load")
        echo -e "${YELLOW}Single-platform build, loading to local Docker${NC}"
    fi
fi

BUILD_ARGS+=(".")

echo
echo -e "${BLUE}Running: docker buildx ${BUILD_ARGS[*]}${NC}"
echo

docker buildx "${BUILD_ARGS[@]}"

echo
echo -e "${GREEN}✓ Build complete!${NC}"
echo
echo "Cache location: ${REGISTRY}/${IMAGE_NAME}:buildcache"
echo "Image: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
