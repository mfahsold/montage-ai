#!/bin/bash
# Remove montage-ai from Kubernetes cluster
# Sources centralized configuration from deploy/config.env
# Supports cleanup of any overlay deployment

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
echo "Removing ${APP_NAME} from Kubernetes Cluster"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Target Namespace: ${CLUSTER_NAMESPACE}"
echo ""

# Check kubectl connectivity
if ! kubectl cluster-info &> /dev/null; then
  echo "❌ ERROR: Cannot connect to Kubernetes cluster"
  exit 1
fi

# Delete deployment and associated resources
echo "⏳ Deleting deployment ${APP_NAME}..."
kubectl delete deployment ${APP_NAME} -n ${CLUSTER_NAMESPACE} --ignore-not-found=true

echo "⏳ Deleting services..."
kubectl delete svc ${APP_NAME} -n ${CLUSTER_NAMESPACE} --ignore-not-found=true

echo "⏳ Deleting configmaps..."
kubectl delete configmap montage-ai-config -n ${CLUSTER_NAMESPACE} --ignore-not-found=true

echo "⏳ Waiting for pods to terminate..."
kubectl wait --for=delete pod -l ${APP_LABEL} -n ${CLUSTER_NAMESPACE} --timeout=60s || true

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Cleanup Complete"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Remaining resources in namespace ${CLUSTER_NAMESPACE}:"
kubectl get all -n ${CLUSTER_NAMESPACE} 2>/dev/null || echo "  (none)"
echo ""
echo "To delete namespace completely:"
echo "  kubectl delete namespace ${CLUSTER_NAMESPACE}"
echo ""
