#!/bin/bash
# Deploy Montage AI to Kubernetes using Kustomize
# Supports multiple environments: dev, staging, production
# Sources centralized configuration from deploy/config.env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(dirname "$SCRIPT_DIR")"
OVERLAY="${1:-dev}"  # Default to dev overlay

# Validate overlay exists
if [ ! -d "${SCRIPT_DIR}/overlays/${OVERLAY}" ]; then
  echo "❌ ERROR: Overlay '${OVERLAY}' not found"
  echo "Available overlays:"
  ls -1 "${SCRIPT_DIR}/overlays/" | sed 's/^/  - /'
  exit 1
fi

# Source centralized configuration
if [ -f "${DEPLOY_ROOT}/config.env" ]; then
  source "${DEPLOY_ROOT}/config.env"
else
  echo "❌ ERROR: Configuration file not found at ${DEPLOY_ROOT}/config.env"
  exit 1
fi

# Auto-detect cluster registry when defaults are used
if [ -f "${SCRIPT_DIR}/registry-detect.sh" ]; then
  source "${SCRIPT_DIR}/registry-detect.sh"
fi

echo "════════════════════════════════════════════════════════════"
echo "Deploying ${APP_NAME} to Kubernetes (Overlay: ${OVERLAY})"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Overlay: ${OVERLAY}"
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

# Ensure image reference matches the configured registry/tag
pushd "${SCRIPT_DIR}/overlays/${OVERLAY}" > /dev/null
kustomize edit set image "ghcr.io/mfahsold/montage-ai=${IMAGE_FULL}"
popd > /dev/null

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
