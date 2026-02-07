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

BUILD_MULTIARCH="${BUILD_MULTIARCH:-true}"
BUILD_PLATFORMS="${BUILD_PLATFORMS:-linux/amd64,linux/arm64}"
BUILDER_NAME="${BUILDER_NAME:-montage-multiarch}"
BUILDKIT_CONFIG="${BUILDKIT_CONFIG:-${PROJECT_ROOT}/buildkitd.toml}"
FORCE_BUILDER_RECREATE="${FORCE_BUILDER_RECREATE:-false}"
USE_REGISTRY_CACHE="${USE_REGISTRY_CACHE:-true}"
CACHE_REF="${CACHE_REF:-${REGISTRY_URL}/${APP_NAME}:buildcache}"
BUILDER_CANDIDATES="${BUILDER_CANDIDATES:-montage-multiarch,simple-builder,fluxibri,multiarch-builder,montage-ai-builder}"
BUILD_PROGRESS="${BUILD_PROGRESS:-plain}"
BUILDKIT_CONFIG_ARG=""

if [ -f "${BUILDKIT_CONFIG}" ]; then
  BUILDKIT_CONFIG_ARG="--config ${BUILDKIT_CONFIG}"
fi

supports_required_platforms() {
  local builder="$1"
  local required_csv="$2"
  local inspect_out required

  if ! inspect_out="$(docker buildx inspect "${builder}" --bootstrap 2>/dev/null | tr '\n' ' ')"; then
    return 1
  fi

  for required in ${required_csv//,/ }; do
    if [[ "${inspect_out}" != *"${required}"* ]]; then
      return 1
    fi
  done

  return 0
}

select_builder() {
  local requested="$1"
  local platforms="$2"
  local candidate

  if supports_required_platforms "${requested}" "${platforms}"; then
    echo "${requested}"
    return 0
  fi

  IFS=',' read -r -a candidates <<< "${BUILDER_CANDIDATES}"
  for candidate in "${candidates[@]}"; do
    candidate="${candidate// /}"
    [ -z "${candidate}" ] && continue
    if supports_required_platforms "${candidate}" "${platforms}"; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

cd "${PROJECT_ROOT}"

if [ "${BUILD_MULTIARCH}" = "true" ]; then
  echo "📦 Building multi-arch image (${BUILD_PLATFORMS})..."

  SELECTED_BUILDER="$(select_builder "${BUILDER_NAME}" "${BUILD_PLATFORMS}" || true)"

  # Ensure buildx builder exists and is active
  if [ -n "${SELECTED_BUILDER}" ]; then
    echo "✅ Using existing multi-arch builder: ${SELECTED_BUILDER}"
    docker buildx use "${SELECTED_BUILDER}" >/dev/null
  else
    if [ "${FORCE_BUILDER_RECREATE}" = "true" ]; then
      docker buildx rm "${BUILDER_NAME}" >/dev/null 2>&1 || true
    fi

    if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
      docker buildx create --name "${BUILDER_NAME}" \
        --driver docker-container \
        ${BUILDKIT_CONFIG_ARG} \
        --use >/dev/null
    else
      docker buildx use "${BUILDER_NAME}" >/dev/null
    fi
    SELECTED_BUILDER="${BUILDER_NAME}"
    echo "⚠️  Falling back to local builder: ${SELECTED_BUILDER}"
  fi

  # Ensure binfmt/qemu is available (best-effort)
  docker run --privileged --rm tonistiigi/binfmt --install all >/dev/null 2>&1 || true

  echo ""
  echo "📤 Building and pushing ${IMAGE_FULL}..."
  BUILD_ARGS=(
    --builder "${SELECTED_BUILDER}"
    --platform "${BUILD_PLATFORMS}"
    --progress "${BUILD_PROGRESS}"
    --build-arg "BUILDKIT_INLINE_CACHE=1"
    -t "${IMAGE_FULL}"
    -f Dockerfile
    --push
  )
  if [ "${USE_REGISTRY_CACHE}" = "true" ]; then
    BUILD_ARGS+=(
      --cache-from "type=registry,ref=${CACHE_REF}"
      --cache-to "type=registry,ref=${CACHE_REF},mode=max"
    )
    echo "🧠 Registry cache enabled: ${CACHE_REF}"
  else
    echo "ℹ️  Registry cache disabled (USE_REGISTRY_CACHE=false)"
  fi

  if docker buildx build "${BUILD_ARGS[@]}" .; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "✅ Multi-arch image successfully pushed!"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Image: ${IMAGE_FULL}"
    echo ""
    echo "Next steps:"
    echo "  1. Deploy: ./deploy.sh cluster"
    echo "  2. Verify: docker buildx imagetools inspect ${IMAGE_FULL}"
    echo "  3. Test: kubectl get pods -n montage-ai"
  else
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "❌ Multi-arch build failed! Troubleshooting:"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "1. Check buildx:"
    echo "   docker buildx ls"
    echo ""
    echo "2. Ensure binfmt is installed:"
    echo "   docker run --privileged --rm tonistiigi/binfmt --install all"
    echo ""
    echo "3. Retry single-arch:"
    echo "   BUILD_MULTIARCH=false ./build-and-push.sh"
    echo ""
    exit 1
  fi
  exit 0
fi

# Single-arch fallback
echo "📦 Building Docker image '${IMAGE_NAME}:${IMAGE_TAG}'..."
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
  echo "  1. Deploy: ./deploy.sh cluster"
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
