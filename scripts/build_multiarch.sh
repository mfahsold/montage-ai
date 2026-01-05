#!/bin/bash
set -e

# Ensure we are using the distributed builder
docker buildx use distributed-builder

echo "Starting multi-arch build for montage-ai..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t 192.168.1.12:5000/montage-ai:latest \
  --cache-from type=registry,ref=192.168.1.12:5000/montage-ai:buildcache \
  --cache-to type=registry,ref=192.168.1.12:5000/montage-ai:buildcache,mode=max \
  --push \
  --progress=plain \
  .

echo "Build completed successfully!"
