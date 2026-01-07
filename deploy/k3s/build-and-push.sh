#!/bin/bash
# Build and push montage-ai Docker image to local registry

set -e

REGISTRY="192.168.1.12:5000"
IMAGE="montage-ai"
TAG="latest"

echo "Building montage-ai Docker image..."
docker build -t "${IMAGE}:${TAG}" -f Dockerfile .

echo "Tagging image for registry..."
docker tag "${IMAGE}:${TAG}" "${REGISTRY}/${IMAGE}:${TAG}"

echo "Pushing to registry ${REGISTRY}..."
docker push "${REGISTRY}/${IMAGE}:${TAG}"

echo "✓ Image pushed to ${REGISTRY}/${IMAGE}:${TAG}"
