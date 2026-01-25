#!/bin/bash
# Deploy Montage AI to Kubernetes using Kustomize
# Supports multiple environments: dev, staging, production
# Sources centralized configuration from deploy/k3s/config-global.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${DEPLOY_ROOT}/.." && pwd)"
OVERLAY="${1:-dev}"  # Default to dev overlay

# Validate overlay exists
if [ ! -d "${SCRIPT_DIR}/overlays/${OVERLAY}" ]; then
  echo "❌ ERROR: Overlay '${OVERLAY}' not found"
  echo "Available overlays:"
  ls -1 "${SCRIPT_DIR}/overlays/" | sed 's/^/  - /'
  exit 1
fi

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
echo "Deploying ${APP_NAME} to Kubernetes (Overlay: ${OVERLAY})"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Overlay: ${OVERLAY}"
echo "  Namespace: ${CLUSTER_NAMESPACE:-montage-ai}"
echo "  Registry: ${REGISTRY_URL:-}"
echo "  Image: ${IMAGE_FULL:-}"
echo ""

# Check kubectl connectivity
if ! kubectl cluster-info &> /dev/null; then
  echo "❌ ERROR: Cannot connect to Kubernetes cluster"
  exit 1
fi

# Validate kustomize is available
if ! command -v kustomize &> /dev/null; then
  echo "❌ ERROR: kustomize not found. Install with: go install sigs.k8s.io/kustomize/kustomize/cmd/kustomize@latest"
  exit 1
fi

# Build manifests with kustomize (includes registry substitution)
echo "Building manifests from overlay '${OVERLAY}'..."
MANIFEST_FILE=$(mktemp)
trap "rm -f ${MANIFEST_FILE}" EXIT

# Jobs are immutable; delete render job before re-apply when needed
if [[ "${OVERLAY}" == "distributed" ]]; then
  kubectl delete job montage-ai-render -n "${CLUSTER_NAMESPACE}" --ignore-not-found > /dev/null 2>&1 || true
fi

kustomize build --load-restrictor LoadRestrictionsNone "${SCRIPT_DIR}/overlays/${OVERLAY}" > "${MANIFEST_FILE}"

# Apply manifests
echo "Applying manifests to namespace ${CLUSTER_NAMESPACE}..."
kubectl apply -f "${MANIFEST_FILE}"

# Wait for deployment
echo "Waiting for deployment to be ready (up to 300s)..."
if kubectl wait --for=condition=available --timeout=300s deployment/${APP_NAME} -n ${CLUSTER_NAMESPACE} 2>/dev/null; then
  echo "✅ Deployment ready!"
else
  echo "⚠️  Deployment not ready after 300s. Check pod status:"
  kubectl get pods -n ${CLUSTER_NAMESPACE} -l ${APP_LABEL}
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Manifests Applied!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Status:"
kubectl get pods -n ${CLUSTER_NAMESPACE} -l ${APP_LABEL}
echo ""
echo "Useful commands:"
echo "  • View logs: kubectl logs -n ${CLUSTER_NAMESPACE} -l ${APP_LABEL} -f"
echo "  • Port forward: kubectl port-forward -n ${CLUSTER_NAMESPACE} svc/${APP_NAME} 8080:80"
echo "  • Delete deployment: ./undeploy.sh"
echo "  • Describe pod: kubectl describe pod <pod-name> -n ${CLUSTER_NAMESPACE}"
echo ""
