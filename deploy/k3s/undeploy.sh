#!/bin/bash
# Remove montage-ai from K3s cluster

set -e

NAMESPACE="montage-ai"

echo "Removing montage-ai from K3s..."

# Delete resources
kubectl delete -f montage-ai.yaml --ignore-not-found=true

# Wait for namespace deletion
echo "Waiting for namespace cleanup..."
kubectl wait --for=delete namespace/${NAMESPACE} --timeout=60s || true

echo "✓ Montage AI removed from cluster"
