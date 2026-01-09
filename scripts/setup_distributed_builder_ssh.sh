#!/bin/bash
# =============================================================================
# Setup Distributed Builder via SSH
# =============================================================================
# Creates a multi-arch buildx builder using SSH to remote nodes.
#
# Prerequisites:
# - SSH key-based auth to both nodes
# - Docker installed on both nodes
#
# Configuration:
#   AMD64_HOST=user@10.0.0.10 ARM64_HOST=user@10.0.0.20 ./scripts/setup_distributed_builder_ssh.sh
# =============================================================================
set -e

BUILDER_NAME="${BUILDER_NAME:-distributed-builder}"
AMD64_HOST="${AMD64_HOST:-user@amd64-node}"
ARM64_HOST="${ARM64_HOST:-user@arm64-node}"

echo "Setting up distributed builder: $BUILDER_NAME"
echo "  AMD64 node: $AMD64_HOST"
echo "  ARM64 node: $ARM64_HOST"
echo ""

if [ "$AMD64_HOST" = "user@amd64-node" ] || [ "$ARM64_HOST" = "user@arm64-node" ]; then
    echo "WARNING: Using placeholder hosts. Set AMD64_HOST and ARM64_HOST environment variables."
    echo "Example: AMD64_HOST=user@10.0.0.10 ARM64_HOST=user@10.0.0.20 $0"
    exit 1
fi

# Remove existing builder if it exists
if docker buildx ls | grep -q "$BUILDER_NAME"; then
    echo "Removing existing builder $BUILDER_NAME..."
    docker buildx rm "$BUILDER_NAME"
fi

echo "Creating distributed builder $BUILDER_NAME..."

# Add AMD64 node
echo "Adding AMD64 node ($AMD64_HOST)..."
docker buildx create --name "$BUILDER_NAME" \
  --node amd64-builder \
  --platform linux/amd64 \
  "ssh://$AMD64_HOST"

# Add ARM64 node
echo "Adding ARM64 node ($ARM64_HOST)..."
docker buildx create --append --name "$BUILDER_NAME" \
  --node arm64-builder \
  --platform linux/arm64 \
  "ssh://$ARM64_HOST"

echo "Bootstrapping builder..."
docker buildx use "$BUILDER_NAME"
docker buildx inspect --bootstrap

echo "Done! Distributed builder '$BUILDER_NAME' is ready."
