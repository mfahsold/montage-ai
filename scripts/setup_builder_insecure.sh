#!/bin/bash
set -e

echo "Removing old builder..."
docker buildx rm distributed-builder || true

echo "Creating new distributed-builder with insecure registry config..."
docker buildx create --name distributed-builder \
  --driver docker-container \
  --config buildkitd.toml \
  --platform linux/amd64 \
  --node amd64-builder \
  ssh://codeai@192.168.1.16

docker buildx create --name distributed-builder \
  --append \
  --driver docker-container \
  --config buildkitd.toml \
  --platform linux/arm64 \
  --node arm64-builder \
  ssh://codeaijetson@192.168.1.15

echo "Bootstrapping builder..."
docker buildx use distributed-builder
docker buildx inspect --bootstrap

echo "Builder ready!"
