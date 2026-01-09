#!/bin/bash
# =============================================================================
# Montage-AI Distributed Multi-Arch Build Script
# =============================================================================
# Follows fluxibri_core patterns:
# - Uses distributed builder (ARM64 + AMD64 native nodes)
# - Git SHA tagging for reproducible builds
# - Registry cache for fast rebuilds
# - Canonical environment variables
#
# Usage:
#   ./scripts/build-distributed.sh              # Build with Git SHA tag
#   ./scripts/build-distributed.sh latest       # Build with specific tag
#   TAG=v1.0.0 ./scripts/build-distributed.sh   # Override via env
#
# Environment Variables:
#   REGISTRY            - Registry host (default: ghcr.io/mfahsold)
#   CACHE_REF           - Cache reference (default: $REGISTRY/montage-ai:buildcache)
#   BUILDER             - Buildx builder name (default: default)
#   PLATFORMS           - Target platforms (default: linux/amd64,linux/arm64)
#   TAG                 - Image tag (default: git SHA)
#   PUSH                - Push to registry (default: true)
#
# For self-hosted registries:
#   REGISTRY=your-registry:5000 ./scripts/build-distributed.sh
# =============================================================================

set -euo pipefail

# Build configuration - ghcr.io as default for public builds
REGISTRY="${REGISTRY:-ghcr.io/mfahsold}"
CACHE_REF="${CACHE_REF:-${REGISTRY}/montage-ai:buildcache}"
IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
BUILDER="${BUILDER:-default}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-true}"

# Git SHA for reproducible builds
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "dev")
GIT_SHA_FULL=$(git rev-parse HEAD 2>/dev/null || echo "dev")
TAG="${TAG:-${1:-$GIT_SHA}}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Montage-AI Distributed Multi-Arch Build                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Registry:   ${GREEN}$REGISTRY${NC}"
echo -e "Image:      ${GREEN}$IMAGE_NAME:$TAG${NC}"
echo -e "Git SHA:    ${GREEN}$GIT_SHA${NC}"
echo -e "Platforms:  ${GREEN}$PLATFORMS${NC}"
echo -e "Builder:    ${GREEN}$BUILDER${NC}"
echo -e "Cache:      ${GREEN}$CACHE_REF${NC}"
echo ""

# Preflight checks
echo -e "${YELLOW}[1/4] Preflight checks...${NC}"

# Check builder exists and is running
if ! docker buildx inspect "$BUILDER" &>/dev/null; then
    echo -e "${RED}ERROR: Builder '$BUILDER' not found${NC}"
    echo "Available builders:"
    docker buildx ls
    echo ""
    echo "Create builder with:"
    echo "  docker buildx create --name $BUILDER --driver docker-container"
    exit 1
fi

# Check registry is accessible
if ! curl -s --connect-timeout 5 "http://$REGISTRY/v2/" &>/dev/null; then
    echo -e "${YELLOW}WARNING: Registry $REGISTRY not accessible (continuing anyway)${NC}"
fi

echo -e "  ✅ Builder '$BUILDER' ready"
echo -e "  ✅ Registry '$REGISTRY' configured"

# Build arguments
echo ""
echo -e "${YELLOW}[2/4] Preparing build...${NC}"

BUILD_ARGS=(
    "build"
    "--builder" "$BUILDER"
    "--platform" "$PLATFORMS"
    "--cache-from" "type=registry,ref=${CACHE_REF}"
    "--cache-to" "type=registry,ref=${CACHE_REF},mode=max"
    "--build-arg" "BUILDKIT_INLINE_CACHE=1"
    "--build-arg" "GIT_COMMIT=$GIT_SHA_FULL"
    "--build-arg" "BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "-t" "${REGISTRY}/${IMAGE_NAME}:${TAG}"
)

# Add latest tag if building with Git SHA
if [ "$TAG" = "$GIT_SHA" ]; then
    BUILD_ARGS+=("-t" "${REGISTRY}/${IMAGE_NAME}:latest")
    echo -e "  Tags: ${GREEN}:$TAG${NC} + ${GREEN}:latest${NC}"
else
    echo -e "  Tag: ${GREEN}:$TAG${NC}"
fi

# Push or load
if [ "$PUSH" = "true" ]; then
    BUILD_ARGS+=("--push")
    echo -e "  Mode: ${GREEN}push to registry${NC}"
else
    if [[ "$PLATFORMS" == *","* ]]; then
        echo -e "${YELLOW}  Multi-platform build requires --push, enabling...${NC}"
        BUILD_ARGS+=("--push")
    else
        BUILD_ARGS+=("--load")
        echo -e "  Mode: ${GREEN}load to local Docker${NC}"
    fi
fi

BUILD_ARGS+=(".")

# Execute build
echo ""
echo -e "${YELLOW}[3/4] Building...${NC}"
echo -e "${BLUE}docker buildx ${BUILD_ARGS[*]}${NC}"
echo ""

time docker buildx "${BUILD_ARGS[@]}"

# Summary
echo ""
echo -e "${YELLOW}[4/4] Build complete!${NC}"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    ✅ Build Successful!                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Image:     ${REGISTRY}/${IMAGE_NAME}:${TAG}"
if [ "$TAG" = "$GIT_SHA" ]; then
    echo "Also:      ${REGISTRY}/${IMAGE_NAME}:latest"
fi
echo "Git SHA:   $GIT_SHA_FULL"
echo "Cache:     $CACHE_REF"
echo ""
echo "Next steps:"
echo "  # Verify image"
echo "  docker buildx imagetools inspect ${REGISTRY}/${IMAGE_NAME}:${TAG}"
echo ""
echo "  # Deploy to cluster"
echo "  kubectl rollout restart deployment/montage-ai-web -n montage-ai"
echo ""
echo "  # Or update image tag in kustomization"
echo "  # deploy/k3s/base/kustomization.yaml → newTag: $TAG"
