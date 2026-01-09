#!/bin/bash
# =============================================================================
# Setup Distributed Builder (Local Configuration)
# =============================================================================
# This script sets up a multi-node buildx builder using SSH.
# Configure your nodes via the NODES associative array below or via env vars.
#
# Configuration:
#   Edit the NODES array below, or set environment variables:
#   NODE_1="10.0.0.10:builder-1:user:arm64"
#   NODE_2="10.0.0.11:builder-2:user:amd64"
#
# Format: "ip:name:user:platform"
# =============================================================================
set -e

# Configuration
CONFIG_FILE="${CONFIG_FILE:-$(pwd)/buildkitd.toml}"
BUILDER_NAME="${BUILDER_NAME:-distributed-builder}"

echo "Using config file: $CONFIG_FILE"
echo "Builder name: $BUILDER_NAME"

# =============================================================================
# NODE CONFIGURATION
# =============================================================================
# Edit this section to match your cluster.
# Format: ["IP"]="name:ssh_user:platform"
#
# Example for a 4-node cluster:
#   ["10.0.0.10"]="control-plane:user:arm64"
#   ["10.0.0.11"]="gpu-node:user:amd64"
#   ["10.0.0.12"]="worker-1:user:amd64"
#   ["10.0.0.13"]="jetson:user:arm64"
# =============================================================================
declare -A NODES=(
  # Add your nodes here:
  # ["10.0.0.10"]="builder-arm64:user:arm64"
  # ["10.0.0.11"]="builder-amd64:user:amd64"
)

# Allow environment variable override
if [ -n "$NODE_1" ]; then
  IFS=':' read -r ip name user platform <<< "$NODE_1"
  NODES["$ip"]="$name:$user:$platform"
fi
if [ -n "$NODE_2" ]; then
  IFS=':' read -r ip name user platform <<< "$NODE_2"
  NODES["$ip"]="$name:$user:$platform"
fi
if [ -n "$NODE_3" ]; then
  IFS=':' read -r ip name user platform <<< "$NODE_3"
  NODES["$ip"]="$name:$user:$platform"
fi
if [ -n "$NODE_4" ]; then
  IFS=':' read -r ip name user platform <<< "$NODE_4"
  NODES["$ip"]="$name:$user:$platform"
fi

# Check if any nodes are configured
if [ ${#NODES[@]} -eq 0 ]; then
  echo "ERROR: No nodes configured!"
  echo ""
  echo "Edit this script and add nodes to the NODES array, or set environment variables:"
  echo "  NODE_1=\"10.0.0.10:builder-arm64:user:arm64\" NODE_2=\"10.0.0.11:builder-amd64:user:amd64\" $0"
  exit 1
fi

echo "Removing old $BUILDER_NAME..."
docker buildx rm "$BUILDER_NAME" 2>/dev/null || true

echo "Creating new $BUILDER_NAME with config..."
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
      --name "$BUILDER_NAME" \
      --node "$name" \
      --driver docker-container \
      --driver-opt network=host \
      --config "$CONFIG_FILE" \
      "ssh://${user}@${ip}" \
      --platform "linux/$platform" \
      --bootstrap
    FIRST_NODE=false
  else
    docker buildx create \
      --name "$BUILDER_NAME" \
      --append \
      --node "$name" \
      --driver-opt network=host \
      --config "$CONFIG_FILE" \
      "ssh://${user}@${ip}" \
      --platform "linux/$platform"
  fi
done

echo "Bootstrapping..."
docker buildx inspect "$BUILDER_NAME" --bootstrap
echo "Done."
