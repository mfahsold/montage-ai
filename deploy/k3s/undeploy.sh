#!/bin/bash
# Remove montage-ai from K3s cluster
# Sources centralized configuration from deploy/config.env

set -e

# Source centralized configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${SCRIPT_DIR}/config.env" ]; then
  source "${SCRIPT_DIR}/config.env"
else
  echo "❌ ERROR: Configuration file not found at ${SCRIPT_DIR}/config.env"
  exit 1
fi

echo "Removing ${APP_NAME} from K3s cluster..."

# Delete resources
kubectl delete -f montage-ai.yaml --ignore-not-found=true

# Wait for namespace deletion
echo "Waiting for namespace cleanup..."
kubectl wait --for=delete namespace/${CLUSTER_NAMESPACE} --timeout=60s || true

echo "✅ ${APP_NAME} removed from cluster"
