#!/bin/bash
# =============================================================================
# Setup Distributed BuildKit Builders in Kubernetes
# =============================================================================
# This script configures BuildKit deployments in your K8s cluster.
#
# Prerequisites:
# - BuildKit deployments (cluster-builder0, cluster-builder1) in montage-ai namespace
# - Local Docker with buildx installed
#
# Configuration:
#   BUILDER0_HOST=10.0.0.10 BUILDER1_HOST=10.0.0.11 ./scripts/setup_distributed_builder.sh
# =============================================================================
set -e

BUILDER0_HOST="${BUILDER0_HOST:-builder0-node}"
BUILDER1_HOST="${BUILDER1_HOST:-builder1-node}"
BUILDER0_PORT="${BUILDER0_PORT:-1234}"
BUILDER1_PORT="${BUILDER1_PORT:-1234}"

echo "Setting up distributed BuildKit builders..."
echo "  Builder 0: $BUILDER0_HOST:$BUILDER0_PORT"
echo "  Builder 1: $BUILDER1_HOST:$BUILDER1_PORT"
echo ""

if [ "$BUILDER0_HOST" = "builder0-node" ] || [ "$BUILDER1_HOST" = "builder1-node" ]; then
    echo "WARNING: Using placeholder hosts. Set BUILDER0_HOST and BUILDER1_HOST environment variables."
    echo "Example: BUILDER0_HOST=10.0.0.10 BUILDER1_HOST=10.0.0.11 $0"
    exit 1
fi

echo "Applying BuildKit config..."
kubectl apply -f deploy/k3s/base/buildkit-config.yaml

echo "Patching cluster-builder0..."
kubectl patch deployment cluster-builder0 -n montage-ai --type='json' -p='[
  {"op": "add", "path": "/spec/template/spec/containers/0/volumeMounts", "value": [{"name": "config", "mountPath": "/etc/buildkit/buildkitd.toml", "subPath": "buildkitd.toml"}]},
  {"op": "add", "path": "/spec/template/spec/volumes", "value": [{"name": "config", "configMap": {"name": "buildkit-config"}}]},
  {"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": [
    "--addr", "tcp://0.0.0.0:1234",
    "--addr", "unix:///run/buildkit/buildkitd.sock",
    "--allow-insecure-entitlement=network.host",
    "--config", "/etc/buildkit/buildkitd.toml"
  ]}
]'

echo "Patching cluster-builder1..."
kubectl patch deployment cluster-builder1 -n montage-ai --type='json' -p='[
  {"op": "add", "path": "/spec/template/spec/containers/0/volumeMounts", "value": [{"name": "config", "mountPath": "/etc/buildkit/buildkitd.toml", "subPath": "buildkitd.toml"}]},
  {"op": "add", "path": "/spec/template/spec/volumes", "value": [{"name": "config", "configMap": {"name": "buildkit-config"}}]},
  {"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": [
    "--addr", "tcp://0.0.0.0:1234",
    "--addr", "unix:///run/buildkit/buildkitd.sock",
    "--allow-insecure-entitlement=network.host",
    "--config", "/etc/buildkit/buildkitd.toml"
  ]}
]'

echo "Waiting for rollouts..."
kubectl rollout status deployment/cluster-builder0 -n montage-ai
kubectl rollout status deployment/cluster-builder1 -n montage-ai

echo "Configuring local docker buildx..."
# Remove existing builder if it exists
if docker buildx ls | grep -q "mybuilder"; then
    docker buildx rm mybuilder
fi

# Create new builder with first node
docker buildx create --name mybuilder --driver remote "tcp://${BUILDER0_HOST}:${BUILDER0_PORT}" --use

# Append second node
docker buildx create --append --name mybuilder --driver remote "tcp://${BUILDER1_HOST}:${BUILDER1_PORT}"

echo "Bootstraping builder..."
docker buildx inspect --bootstrap

echo "Done! Distributed builder 'mybuilder' is ready."
