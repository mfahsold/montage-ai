#!/bin/bash
# Deploy Montage AI to Kubernetes Cluster
# Sources centralized configuration from deploy/config.env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(dirname "$SCRIPT_DIR")"

# Source centralized configuration
if [ -f "${DEPLOY_ROOT}/config.env" ]; then
  source "${DEPLOY_ROOT}/config.env"
else
  echo "❌ ERROR: Configuration file not found at ${DEPLOY_ROOT}/config.env"
  exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "Deploying ${APP_NAME} to Kubernetes Cluster"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Namespace: ${CLUSTER_NAMESPACE}"
echo "  Registry: ${REGISTRY_URL}"
echo "  Image: ${IMAGE_FULL}"
echo "  Domain: ${APP_DOMAIN}"
echo ""

# Check kubectl connectivity
if ! kubectl cluster-info &> /dev/null; then
  echo "❌ ERROR: Cannot connect to Kubernetes cluster"
  exit 1
fi

# Apply manifests
echo "Applying manifests to namespace ${CLUSTER_NAMESPACE}..."
kubectl apply -f "${SCRIPT_DIR}/montage-ai.yaml"

# Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/${APP_NAME} -n ${CLUSTER_NAMESPACE}

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Deployment Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Status:"
kubectl get pods -n ${CLUSTER_NAMESPACE} -l ${APP_LABEL}
echo ""
echo "Access Information:"
echo "  • Web UI: https://${APP_DOMAIN}"
echo "  • Port forward: kubectl port-forward -n ${CLUSTER_NAMESPACE} svc/${SERVICE_NAME} ${SERVICE_PORT}:${SERVICE_TARGET_PORT}"
echo "  • Logs: kubectl logs -n ${CLUSTER_NAMESPACE} -l ${APP_LABEL} -f"
echo ""
echo ""
echo "View logs:"
echo "  kubectl logs -n ${NAMESPACE} -l app=montage-ai -f"
