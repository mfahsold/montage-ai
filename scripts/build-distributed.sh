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
#   REGISTRY            - Registry host (default: ${REGISTRY_URL} from config)
#   CACHE_REF           - Cache reference (default: $REGISTRY/montage-ai:buildcache)
#   BUILDER             - Buildx builder name (default: distributed-builder)
#   PLATFORMS           - Target platforms (default: linux/amd64,linux/arm64)
#   TAG                 - Image tag (default: git SHA)
#   PUSH                - Push to registry (default: true)
#   PARALLEL_JOBS       - Parallel build jobs (default: 4)
#   BUILDKIT_SSH        - If set to 1, enables BuildKit SSH forwarding (--ssh default)
#   GOPRIVATE           - If set, passed as a build-arg (useful for Go private modules)
#
# For public releases (GitHub Container Registry):
#   REGISTRY=ghcr.io/your-org ./scripts/build-distributed.sh
#
# Distributed Builder Setup (parallel native ARM64 + AMD64):
#   docker buildx create --name distributed-builder --driver docker-container
#   docker buildx create --append --name distributed-builder ssh://user@arm64-node
#   docker buildx create --append --name distributed-builder ssh://user@amd64-node
# =============================================================================

set -euo pipefail

# Build configuration
# Default: GHCR for reliable public registry
# Cluster: set REGISTRY to your local cluster registry (e.g., REGISTRY=your-registry:30500)
# Note: Cluster registry may use self-signed certs; configure trust accordingly
# Load canonical settings
# shellcheck disable=SC1090
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
REGISTRY="${REGISTRY:-${REGISTRY_URL}}"
CACHE_REF="${CACHE_REF:-${REGISTRY}/montage-ai:buildcache}"
IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
# simple-builder has parallel ARM64 (Pi) + AMD64 (Server) nodes running natively
BUILDER="${BUILDER:-simple-builder}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-true}"
# Parallel build jobs per platform (uses native nodes if distributed-builder is configured)
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

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

# Preflight checks - run first to select best builder
echo -e "${YELLOW}[1/4] Preflight checks...${NC}"

# Check builder exists and supports required platforms
# Note: Multi-node builders have platforms spread across nodes, so we aggregate all
check_builder_platforms() {
    local builder=$1
    local all_platforms
    # Get ALL platforms from ALL nodes (multi-line grep)
    all_platforms=$(docker buildx inspect "$builder" --bootstrap 2>/dev/null | grep -E "Platforms:" | tr '\n' ' ' || echo "")
    # Check if builder supports both amd64 and arm64 (across any nodes)
    if [[ "$all_platforms" == *"linux/amd64"* ]] && [[ "$all_platforms" == *"linux/arm64"* ]]; then
        return 0
    fi
    return 1
}

find_best_builder() {
    # Prefer builders with both ARM64 and AMD64 support
    for b in "simple-builder" "multiarch-builder" "distributed-builder" "local-dist"; do
        if docker buildx inspect "$b" &>/dev/null && check_builder_platforms "$b"; then
            echo "$b"
            return 0
        fi
    done
    # Fallback to any available builder
    for b in "simple-builder" "multiarch-builder" "default"; do
        if docker buildx inspect "$b" &>/dev/null; then
            echo "$b"
            return 0
        fi
    done
    return 1
}

# Check builder exists and has required platforms
if ! docker buildx inspect "$BUILDER" &>/dev/null || ! check_builder_platforms "$BUILDER"; then
    echo -e "${YELLOW}  ⚠ Builder '$BUILDER' not found or missing platforms, finding alternative...${NC}"

    BEST_BUILDER=$(find_best_builder)
    if [ -n "$BEST_BUILDER" ]; then
        BUILDER="$BEST_BUILDER"
        if check_builder_platforms "$BUILDER"; then
            echo -e "${GREEN}  → Using '$BUILDER' (multi-arch capable)${NC}"
        else
            echo -e "${YELLOW}  → Using '$BUILDER' (limited platforms)${NC}"
            # Warn about platform limitations
            if [[ "$PLATFORMS" == *","* ]]; then
                echo -e "${YELLOW}  ⚠ Multi-platform may not be fully supported${NC}"
            fi
        fi
    else
        echo -e "${RED}ERROR: No suitable builder found${NC}"
        echo "Create distributed builder with parallel ARM64+AMD64 nodes:"
        echo "  docker buildx create --name simple-builder --driver docker-container"
        echo "  docker buildx create --append --name simple-builder ssh://user@arm64-host"
        echo "  docker buildx create --append --name simple-builder ssh://user@amd64-host"
        exit 1
    fi
fi

# Check registry is accessible (HTTP for insecure registry)
REGISTRY_HOST="${REGISTRY%%/*}"
if curl -s --connect-timeout 3 "http://${REGISTRY_HOST}/v2/" &>/dev/null; then
    echo -e "  ✅ Registry '${REGISTRY_HOST}' accessible (HTTP)"
elif curl -s --connect-timeout 3 "https://${REGISTRY_HOST}/v2/" &>/dev/null; then
    echo -e "  ✅ Registry '${REGISTRY_HOST}' accessible (HTTPS)"
else
    echo -e "${YELLOW}  ⚠ Registry ${REGISTRY_HOST} not reachable (build may fail on push)${NC}"
fi

# Show builder info (aggregate all nodes' platforms)
BUILDER_PLATFORMS=$(docker buildx inspect "$BUILDER" --bootstrap 2>/dev/null | grep -E "Platforms:" | sed 's/Platforms:[[:space:]]*//' | tr '\n' '+' | sed 's/+$//' || echo "unknown")
echo -e "  ✅ Builder '$BUILDER' ready"
echo -e "     Platforms: $BUILDER_PLATFORMS"

# Build arguments
echo ""
echo -e "${YELLOW}[2/4] Preparing build...${NC}"

# Note: For insecure registries (HTTP), builders need to be created with:
#   docker buildx create --name builder --driver docker-container \
#     --buildkitd-flags '--allow-insecure-entitlement=network.host' \
#     --config /path/to/buildkitd.toml
# GHCR uses HTTPS so no special config needed

BUILD_ARGS=(
    "build"
    "--builder" "$BUILDER"
    "--platform" "$PLATFORMS"
    "--cache-from" "type=registry,ref=${CACHE_REF}"
    "--cache-to" "type=registry,ref=${CACHE_REF},mode=max"
    "--build-arg" "BUILDKIT_INLINE_CACHE=1"
    "--build-arg" "GIT_COMMIT=$GIT_SHA_FULL"
    "--build-arg" "BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "--progress" "plain"
    "-t" "${REGISTRY}/${IMAGE_NAME}:${TAG}"
)

# Parallel jobs for faster builds (especially with distributed builder)
if [ -n "${PARALLEL_JOBS:-}" ] && [ "$PARALLEL_JOBS" -gt 1 ]; then
    # BuildKit parallelism via build-arg
    BUILD_ARGS+=("--build-arg" "BUILDKIT_MULTI_PLATFORM=1")
fi

# Optional: enable BuildKit SSH forwarding to allow the build container to use
# your local SSH agent for fetching private git repositories (e.g., private Go modules).
# Usage: export BUILDKIT_SSH=1 && eval "$(ssh-agent -s)" && ssh-add
if [ "${BUILDKIT_SSH:-0}" = "1" ]; then
    echo -e "  ${YELLOW}→ Using BuildKit SSH forwarding (--ssh default). Ensure ssh-agent is running and keys are added.${NC}"
    BUILD_ARGS+=("--ssh" "default")
fi

# Pass GOPRIVATE to the build if set (useful for Go builds that require private module access)
if [ -n "${GOPRIVATE:-}" ]; then
    echo -e "  ${YELLOW}→ Passing GOPRIVATE to build args${NC}"
    BUILD_ARGS+=("--build-arg" "GOPRIVATE=${GOPRIVATE}")
fi

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
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
    echo -e "${RED}Build failed with exit code $BUILD_EXIT${NC}"
    exit $BUILD_EXIT
fi

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
