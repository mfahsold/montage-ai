#!/bin/bash
# Multi-Arch Docker Build Script (Optimized)
# Usage: REGISTRY=your-registry:5000 ./scripts/build-multiarch.sh
set -e

REGISTRY="${REGISTRY:-ghcr.io/mfahsold}"
IMAGE_NAME="${IMAGE_NAME:-montage-ai}"
TAG="${TAG:-latest}"

echo "🔨 Multi-Arch Build (AMD64 + ARM64)"
echo "⏳ Building... (parallel, with cache)"

docker buildx build \
  --builder multiarch-builder \
  --platform linux/amd64,linux/arm64 \
  --cache-from type=registry,ref=$REGISTRY/$IMAGE_NAME:buildcache \
  --cache-to type=registry,ref=$REGISTRY/$IMAGE_NAME:buildcache,mode=max \
  -t $REGISTRY/$IMAGE_NAME:$TAG \
  --push \
  . 2>&1 | tail -30

echo "✅ Done! Restarting pods..."
kubectl rollout restart deployment montage-ai-web -n montage-ai 2>/dev/null || true
