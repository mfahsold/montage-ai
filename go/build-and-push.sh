#!/bin/bash
# Build script for Go Worker (multi-platform)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REGISTRY="${REGISTRY:-registry.example.com}"
IMAGE_NAME="${IMAGE_NAME:-montage-ai-worker}"
IMAGE_TAG="${IMAGE_TAG:-go-v1-canary}"
IMAGE_FULL="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[info]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
log_error() { echo -e "${RED}[error]${NC} $*"; }

# Step 1: Verify prerequisites
log_info "Checking prerequisites..."

if ! command -v go &> /dev/null; then
    log_error "Go not found. Install from https://golang.org/doc/install"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Install from https://docs.docker.com/get-docker/"
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}')
log_info "Go version: $GO_VERSION"
log_info "Docker version: $(docker --version)"

# Step 2: Build locally (test)
log_info "Building Go Worker locally (test)..."
cd "${PROJECT_ROOT}/go"
go mod tidy
go build -o montage-worker ./cmd/worker
log_info "✅ Local build succeeded: montage-worker"

# Step 3: Run tests (future: add actual tests)
log_info "Running tests..."
# go test ./pkg/... -v

# Step 4: Build Docker image (multi-arch)
log_info "Building Docker image: $IMAGE_FULL"
log_info "Using buildx for multi-platform (amd64, arm64)..."

if ! docker buildx create --name montage-builder 2>/dev/null; then
    log_warn "Buildx builder 'montage-builder' already exists, using it"
fi

docker buildx build \
    --builder montage-builder \
    --platform linux/amd64,linux/arm64 \
    --tag "${IMAGE_FULL}" \
    --push \
    -f "${PROJECT_ROOT}/go/Dockerfile" \
    "${PROJECT_ROOT}"

log_info "✅ Docker image built and pushed: $IMAGE_FULL"

# Step 5: Summary
cat << EOF

${GREEN}✅ Build Complete!${NC}

To deploy the canary:

  kubectl apply -f deploy/k3s/overlays/cluster/worker-go-canary.yaml

To scale up after testing:

  kubectl scale deployment montage-ai-worker-go -n montage-ai --replicas=3

To monitor:

  kubectl logs -n montage-ai -l app.kubernetes.io/component=worker-go -f

To rollback to Python worker:

  kubectl delete deployment montage-ai-worker-go -n montage-ai

EOF

log_info "Build complete."
