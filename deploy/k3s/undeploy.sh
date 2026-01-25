#!/bin/bash
# Remove montage-ai from Kubernetes cluster
# Sources centralized configuration from deploy/k3s/config-global.yaml
# Supports cleanup of any overlay deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${DEPLOY_ROOT}/.." && pwd)"

# Source centralized configuration
CONFIG_GLOBAL="${CONFIG_GLOBAL:-${SCRIPT_DIR}/config-global.yaml}"
CONFIG_ENV_SCRIPT="${REPO_ROOT}/scripts/ops/render_cluster_config_env.sh"
CONFIG_ENV_OUT="${SCRIPT_DIR}/base/cluster-config.env"

if [ -x "${CONFIG_ENV_SCRIPT}" ]; then
  CONFIG_GLOBAL="${CONFIG_GLOBAL}" ENV_OUT="${CONFIG_ENV_OUT}" bash "${CONFIG_ENV_SCRIPT}"
fi

if [ -f "${CONFIG_ENV_OUT}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_ENV_OUT}"
else
  echo "❌ ERROR: Configuration file not found at ${CONFIG_ENV_OUT}"
  exit 1
fi

CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
APP_NAME="${APP_NAME:-montage-ai-web}"
APP_LABEL="${APP_LABEL:-app.kubernetes.io/name=montage-ai}"

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
