#!/bin/bash
set -e

# Configuration
REGISTRY="${REGISTRY:-10.43.17.166:5000}"
IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
TAG="${TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILDER_NAME="distributed-builder"

echo "🔧 Configuring build environment..."
echo "   Registry: $REGISTRY"
echo "   Image:    $IMAGE_NAME:$TAG"
echo "   Platforms: $PLATFORMS"

# Ensure builder exists
if ! docker buildx inspect "$BUILDER_NAME" > /dev/null 2>&1; then
    echo "Creating new buildx builder: $BUILDER_NAME"
    # Create with access to host network for registry access
    docker buildx create --name "$BUILDER_NAME" --driver docker-container --driver-opt network=host --use --bootstrap
else
    echo "Using existing builder: $BUILDER_NAME"
    docker buildx use "$BUILDER_NAME"
fi

echo "🚀 Starting multi-arch build..."
docker buildx build \
  --platform "$PLATFORMS" \
  -t "$REGISTRY/$IMAGE_NAME:$TAG" \
  --cache-from "type=registry,ref=$REGISTRY/$IMAGE_NAME:buildcache" \
  --cache-to "type=registry,ref=$REGISTRY/$IMAGE_NAME:buildcache,mode=max" \
  --push \
  --progress=plain \
  .

echo "✅ Build and push completed successfully!"
