#!/bin/bash
set -e

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
docker buildx create --name mybuilder --driver remote tcp://192.168.1.16:1234 --use

# Append second node
docker buildx create --append --name mybuilder --driver remote tcp://192.168.1.17:1234

echo "Bootstraping builder..."
docker buildx inspect --bootstrap

echo "Done! Distributed builder 'mybuilder' is ready."
