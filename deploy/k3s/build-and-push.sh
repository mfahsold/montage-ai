#!/bin/bash
# Build and push montage-ai Docker image to registry
# Sources centralized configuration from deploy/k3s/config-global.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEPLOY_ROOT")"

# Source centralized configuration
CONFIG_GLOBAL="${CONFIG_GLOBAL:-${SCRIPT_DIR}/config-global.yaml}"
CONFIG_ENV_SCRIPT="${PROJECT_ROOT}/scripts/ops/render_cluster_config_env.sh"
CONFIG_ENV_OUT="${SCRIPT_DIR}/base/cluster-config.env"

if [ -x "${CONFIG_ENV_SCRIPT}" ]; then
  CONFIG_GLOBAL="${CONFIG_GLOBAL}" ENV_OUT="${CONFIG_ENV_OUT}" bash "${CONFIG_ENV_SCRIPT}"
fi

if [ -f "${CONFIG_ENV_OUT}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_ENV_OUT}"
else
  echo "❌ ERROR: Configuration file not found at ${CONFIG_ENV_OUT}"
  exit 1
fi

APP_NAME="${APP_NAME:-montage-ai}"

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

# Check if registry is accessible (best-effort)
echo "🔍 Checking registry connectivity..."
if ! curl -s "http://${REGISTRY_URL}/v2/" > /dev/null 2>&1; then
  echo "⚠️  WARNING: Registry ${REGISTRY_URL} may not be accessible"
  echo "    Continuing with build. Fix registry if push fails."
else
  echo "✅ Registry is accessible"
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
