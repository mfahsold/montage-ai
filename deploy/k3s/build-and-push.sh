#!/bin/bash
# Build and push montage-ai Docker image to local registry
# Sources centralized configuration from deploy/config.env

set -e

# Source centralized configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${SCRIPT_DIR}/config.env" ]; then
  source "${SCRIPT_DIR}/config.env"
else
  echo "❌ ERROR: Configuration file not found at ${SCRIPT_DIR}/config.env"
  exit 1
fi

echo "Building montage-ai Docker image..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .

echo "Tagging image for registry ${REGISTRY_URL}..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_FULL}"

echo "Pushing to registry..."
docker push "${IMAGE_FULL}"

echo "✅ Image pushed: ${IMAGE_FULL}"
