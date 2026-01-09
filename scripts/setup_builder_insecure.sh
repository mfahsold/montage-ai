#!/bin/bash
# =============================================================================
# Setup Distributed Builder with Insecure Registry
# =============================================================================
# Configure your build nodes via environment variables:
#
#   AMD64_HOST=user@10.0.0.10 ARM64_HOST=user@10.0.0.20 ./scripts/setup_builder_insecure.sh
#
# Requirements:
# - SSH key-based auth to both nodes
# - Docker installed on both nodes
# - buildkitd.toml in repo root (for insecure registry config)
# =============================================================================
set -e

# Build node configuration (customize for your cluster)
AMD64_HOST="${AMD64_HOST:-user@amd64-node}"
ARM64_HOST="${ARM64_HOST:-user@arm64-node}"
BUILDER_NAME="${BUILDER_NAME:-distributed-builder}"

echo "Setting up distributed builder: $BUILDER_NAME"
echo "  AMD64 node: $AMD64_HOST"
echo "  ARM64 node: $ARM64_HOST"
echo ""

if [ "$AMD64_HOST" = "user@amd64-node" ] || [ "$ARM64_HOST" = "user@arm64-node" ]; then
    echo "WARNING: Using placeholder hosts. Set AMD64_HOST and ARM64_HOST environment variables."
    echo "Example: AMD64_HOST=user@10.0.0.10 ARM64_HOST=user@10.0.0.20 $0"
    exit 1
fi

echo "Removing old builder..."
docker buildx rm "$BUILDER_NAME" || true

echo "Creating new $BUILDER_NAME with insecure registry config..."
docker buildx create --name "$BUILDER_NAME" \
  --driver docker-container \
  --config buildkitd.toml \
  --platform linux/amd64 \
  --node amd64-builder \
  "ssh://$AMD64_HOST"

docker buildx create --name "$BUILDER_NAME" \
  --append \
  --driver docker-container \
  --config buildkitd.toml \
  --platform linux/arm64 \
  --node arm64-builder \
  "ssh://$ARM64_HOST"

echo "Bootstrapping builder..."
docker buildx use "$BUILDER_NAME"
docker buildx inspect --bootstrap

echo "Builder ready!"
