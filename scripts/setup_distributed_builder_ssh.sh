#!/bin/bash
set -e

BUILDER_NAME="distributed-builder"

# Remove existing builder if it exists
if docker buildx ls | grep -q "$BUILDER_NAME"; then
    echo "Removing existing builder $BUILDER_NAME..."
    docker buildx rm "$BUILDER_NAME"
fi

echo "Creating distributed builder $BUILDER_NAME..."

# Add AMD64 node (FluxibriServer)
echo "Adding AMD64 node (192.168.1.16)..."
docker buildx create --name "$BUILDER_NAME" \
  --node amd64-builder \
  --platform linux/amd64 \
  ssh://codeai@192.168.1.16

# Add ARM64 node (Jetson)
echo "Adding ARM64 node (192.168.1.15)..."
docker buildx create --append --name "$BUILDER_NAME" \
  --node arm64-builder \
  --platform linux/arm64 \
  ssh://codeaijetson@192.168.1.15

echo "Bootstrapping builder..."
docker buildx use "$BUILDER_NAME"
docker buildx inspect --bootstrap

echo "Done! Distributed builder '$BUILDER_NAME' is ready."
