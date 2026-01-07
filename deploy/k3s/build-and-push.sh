#!/bin/bash
# Build and push montage-ai Docker image to registry
# Sources centralized configuration from deploy/config.env
# Supports local registry (192.168.1.12:5000) or remote registries

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEPLOY_ROOT")"

# Source centralized configuration
if [ -f "${DEPLOY_ROOT}/config.env" ]; then
  source "${DEPLOY_ROOT}/config.env"
else
  echo "❌ ERROR: Configuration file not found at ${DEPLOY_ROOT}/config.env"
  exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "Building and Pushing ${APP_NAME} Docker Image"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Registry: ${REGISTRY_URL}"
echo "  Image: ${IMAGE_FULL}"
echo "  Tag: ${IMAGE_TAG}"
echo ""

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
  echo "❌ ERROR: Docker daemon not running"
  exit 1
fi

# Check if registry is accessible (for local registries)
if [[ ${REGISTRY_URL} == "192.168.1.12:5000" ]]; then
  echo "🔍 Checking registry connectivity..."
  if ! curl -s http://${REGISTRY_URL}/v2/ > /dev/null 2>&1; then
    echo "⚠️  WARNING: Registry ${REGISTRY_URL} may not be accessible"
    echo "    Continuing with build. Fix registry if push fails."
  else
    echo "✅ Registry is accessible"
  fi
fi
echo ""

# Build Docker image
echo "📦 Building Docker image '${IMAGE_NAME}:${IMAGE_TAG}'..."
cd "${PROJECT_ROOT}"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .

echo ""
echo "🏷️  Tagging image for registry..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_FULL}"

echo ""
echo "📤 Pushing to registry ${REGISTRY_URL}..."
if docker push "${IMAGE_FULL}"; then
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "✅ Image successfully pushed!"
  echo "════════════════════════════════════════════════════════════"
  echo ""
  echo "Image: ${IMAGE_FULL}"
  echo ""
  echo "Next steps:"
  echo "  1. Deploy: ./deploy.sh [dev|staging|production]"
  echo "  2. Check: docker images | grep montage-ai"
  echo "  3. Test: kubectl get pods -n montage-ai"
else
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "❌ Push failed! Troubleshooting:"
  echo "════════════════════════════════════════════════════════════"
  echo ""
  echo "1. Check Docker daemon:"
  echo "   docker info"
  echo ""
  echo "2. Check registry is running:"
  echo "   curl http://${REGISTRY_URL}/v2/"
  echo ""
  echo "3. Check image was built:"
  echo "   docker images | grep montage-ai"
  echo ""
  echo "4. Try logging in (if auth required):"
  echo "   docker login ${REGISTRY_HOST}"
  echo ""
  exit 1
fi
