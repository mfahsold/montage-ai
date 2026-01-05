#!/bin/bash
set -e

# Configuration
CONFIG_FILE="$(pwd)/buildkitd.toml"

echo "Using config file: $CONFIG_FILE"

# Define nodes to use (Verified working)
declare -A NODES=(
  ["192.168.1.12"]="control-plane-builder:codeai:arm64"
  ["192.168.1.16"]="amd-gpu-builder:codeai:amd64"
  ["192.168.1.37"]="x86-builder:codeai:amd64"
  ["192.168.1.15"]="jetson-builder:codeaijetson:arm64"
)

echo "removing old distributed-builder..."
docker buildx rm distributed-builder 2>/dev/null || true

echo "Creating new distributed-builder with config..."
FIRST_NODE=true

for ip in "${!NODES[@]}"; do
  info="${NODES[$ip]}"
  name="${info%%:*}"
  rest="${info#*:}"
  user="${rest%%:*}"
  platform="${rest#*:}"

  echo "Adding $name ($ip) [$platform] user=$user"

  if [ "$FIRST_NODE" = true ]; then
    docker buildx create \
      --name distributed-builder \
      --node "$name" \
      --driver docker-container \
      --driver-opt network=host \
      --config "$CONFIG_FILE" \
      ssh://${user}@${ip} \
      --platform "linux/$platform" \
      --bootstrap
    FIRST_NODE=false
  else
    docker buildx create \
      --name distributed-builder \
      --append \
      --node "$name" \
      --driver-opt network=host \
      --config "$CONFIG_FILE" \
      ssh://${user}@${ip} \
      --platform "linux/$platform"
  fi
done

echo "Bootstrapping..."
docker buildx inspect distributed-builder --bootstrap
echo "Done."
