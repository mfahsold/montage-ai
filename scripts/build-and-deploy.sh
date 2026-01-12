#!/bin/bash
################################################################################
# Build and Deploy Script for Montage AI
# 
# Solves Docker layer caching issues by:
# 1. Building with BuildKit (better cache invalidation)
# 2. Tagging for local registry (192.168.1.12:30500)
# 3. Pushing to local registry
# 4. k3s pulls automatically with imagePullPolicy: Always
#
# Usage:
#   ./scripts/build-and-deploy.sh [quality=preview|standard|high]
#   ./scripts/build-and-deploy.sh quality=standard  # Full rebuild
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
# Load centralized deployment config if present
if [ -f "deploy/config.env" ]; then
  # shellcheck disable=SC1091
  source "deploy/config.env"
fi

# REGISTRY can come from deploy/config.env (REGISTRY_URL) or environment overrides
REGISTRY="${REGISTRY_URL:-${REGISTRY:-192.168.1.12:30500}}"
IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LOCAL_TAG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
BUILD_QUALITY="${1:-standard}"
NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"

# Build metadata
BUILD_VERSION=$(git describe --tags --always 2>/dev/null || echo "dev")
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
BUILD_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🐳 Montage AI Build & Deploy Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  Version: $BUILD_VERSION"
echo "  Commit:  ${BUILD_COMMIT:0:7}"
echo "  Date:    $BUILD_DATE"
echo "  Quality: $BUILD_QUALITY"
echo "  Registry: $REGISTRY"
echo "  Image:   $LOCAL_TAG"
echo ""

# Step 1: Verify registry connectivity
echo -e "${YELLOW}1️⃣  Checking registry connectivity...${NC}"
if ! curl -s "http://${REGISTRY}/v2/" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot reach registry at ${REGISTRY}${NC}"
    echo -e "${RED}   Please ensure local registry is running${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Registry reachable${NC}"
echo ""

# Step 2: Build image with BuildKit
echo -e "${YELLOW}2️⃣  Building Docker image with BuildKit...${NC}"
export DOCKER_BUILDKIT=1
docker build \
    --build-arg BUILD_VERSION="$BUILD_VERSION" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg BUILD_COMMIT="$BUILD_COMMIT" \
    --build-arg QUALITY_PROFILE="$BUILD_QUALITY" \
    -t "${LOCAL_TAG}" \
    -t "${REGISTRY}/${IMAGE_NAME}:${BUILD_VERSION}" \
    -f Dockerfile \
    . 2>&1 | tail -20

BUILD_STATUS=$?
if [ $BUILD_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed (exit code: $BUILD_STATUS)${NC}"
    exit 1
fi
echo ""

# Step 3: Verify image
echo -e "${YELLOW}3️⃣  Verifying image integrity...${NC}"
docker run --rm "${LOCAL_TAG}" python3 -c "from montage_ai import worker; print('✅ Worker module OK')" 2>&1 | grep -q "Worker module OK"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image verification passed${NC}"
else
    echo -e "${RED}❌ Image verification failed${NC}"
    exit 1
fi
echo ""

# Step 4: Push to local registry
echo -e "${YELLOW}4️⃣  Pushing to local registry...${NC}"
docker push "${LOCAL_TAG}" 2>&1 | tail -10
docker push "${REGISTRY}/${IMAGE_NAME}:${BUILD_VERSION}" 2>&1 | tail -3
echo -e "${GREEN}✅ Pushed to registry${NC}"
echo ""

# Step 5: Trigger Kubernetes deployment
echo -e "${YELLOW}5️⃣  Triggering Kubernetes deployment...${NC}"
if ! kubectl get ns "$NAMESPACE" > /dev/null 2>&1; then
    echo -e "${RED}❌ Namespace $NAMESPACE not found${NC}"
    exit 1
fi

# Delete old pods to force image pull
echo "   Removing old worker pods..."
kubectl delete pods -n "$NAMESPACE" -l component=worker --grace-period=30 2>/dev/null || true
sleep 2

# Wait for new pod to start
echo "   Waiting for new pod..."
sleep 5

# Get pod status
POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l component=worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "$POD_NAME" ]; then
    echo -e "${RED}❌ No worker pod found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pod: $POD_NAME${NC}"
echo ""

# Step 6: Check pod logs
echo -e "${YELLOW}6️⃣  Checking pod status...${NC}"
sleep 3
POD_STATUS=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null)
if [ "$POD_STATUS" = "Running" ]; then
    echo -e "${GREEN}✅ Pod is Running${NC}"
    
    # Show logs
    echo ""
    echo -e "${BLUE}📜 Pod logs (last 30 lines):${NC}"
    kubectl logs "$POD_NAME" -n "$NAMESPACE" --tail=30 2>/dev/null || echo "   (logs not yet available)"
else
    echo -e "${YELLOW}⚠️  Pod status: $POD_STATUS${NC}"
    echo "   Full pod description:"
    kubectl describe pod "$POD_NAME" -n "$NAMESPACE" 2>/dev/null | tail -20
fi
echo ""

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Build and Deploy Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "  • Monitor pod: kubectl logs -f $POD_NAME -n $NAMESPACE"
echo "  • Check image in registry: curl http://${REGISTRY}/v2/_catalog"
echo ""
