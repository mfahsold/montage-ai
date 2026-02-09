#!/bin/bash
# Deploy Montage AI to Kubernetes using Kustomize
# Supports multiple environments: dev, staging, production
# Sources centralized configuration from deploy/k3s/config-global.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${DEPLOY_ROOT}/.." && pwd)"
OVERLAY="${1:-cluster}"  # Default to cluster overlay (canonical)

# Guard: config-global.yaml must exist
CONFIG_CHECK="${SCRIPT_DIR}/config-global.yaml"
if [ ! -f "${CONFIG_CHECK}" ]; then
  echo "❌ ERROR: config-global.yaml not found."
  echo "Run: cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml"
  echo "Then replace all <...> placeholders and run: make -C deploy/k3s config"
  exit 1
fi

# Guard: run pre-flight checks
if [ -x "${SCRIPT_DIR}/pre-flight-check.sh" ]; then
  echo "Running pre-flight checks..."
  bash "${SCRIPT_DIR}/pre-flight-check.sh" || exit 1
  echo ""
fi

# Validate overlay exists
if [ "${OVERLAY}" != "cluster" ]; then
  echo "❌ ERROR: Only the canonical 'cluster' overlay is supported."
  echo "Use local mode for non-Kubernetes runs, or deploy with:"
  echo "  ./deploy.sh cluster"
  exit 1
fi

if [ ! -d "${SCRIPT_DIR}/overlays/${OVERLAY}" ]; then
  echo "❌ ERROR: Overlay '${OVERLAY}' not found"
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

# Validate config-global.yaml has no unreplaced placeholders
if [ -f "${CONFIG_GLOBAL}" ] && grep -qE '<[A-Z_]+>' "${CONFIG_GLOBAL}"; then
  echo "⚠️  WARNING: Unreplaced placeholders found in ${CONFIG_GLOBAL}:"
  grep -n '<[A-Z_]*>' "${CONFIG_GLOBAL}" | while IFS= read -r line; do
    echo "   ${line}"
  done
  echo ""
  echo "Replace all <...> placeholders before deploying."
  echo "See: deploy/k3s/config-global.yaml.example for reference."
  exit 1
fi

# Inject configured namespace into overlay before kustomize build
(cd "${SCRIPT_DIR}/overlays/${OVERLAY}" && kustomize edit set namespace "${CLUSTER_NAMESPACE}")

# Ensure namespace exists
kubectl get namespace "${CLUSTER_NAMESPACE}" >/dev/null 2>&1 || \
  kubectl create namespace "${CLUSTER_NAMESPACE}"
kubectl label namespace "${CLUSTER_NAMESPACE}" \
  app.kubernetes.io/name=montage-ai \
  app.kubernetes.io/component=web-ui \
  fluxibri.ai/tier=app \
  "fluxibri.ai/adaptive-quota=true" \
  --overwrite 2>/dev/null

# Build manifests with kustomize (includes registry substitution)
echo "Building manifests from overlay '${OVERLAY}'..."
MANIFEST_FILE=$(mktemp)
NON_PVC_FILE=$(mktemp)
PVC_DIR=$(mktemp -d)
trap "rm -f ${MANIFEST_FILE} ${NON_PVC_FILE}; rm -rf ${PVC_DIR}" EXIT

kustomize build --load-restrictor LoadRestrictionsNone "${SCRIPT_DIR}/overlays/${OVERLAY}" > "${MANIFEST_FILE}"

# Apply manifests with PVC guard (immutable fields cannot be updated)
echo "Applying manifests to namespace ${CLUSTER_NAMESPACE}..."
csplit -s -f "${PVC_DIR}/doc-" -b "%03d.yaml" "${MANIFEST_FILE}" '/^---$/' '{*}' >/dev/null 2>&1 || true

for doc in "${PVC_DIR}"/doc-*.yaml; do
  if ! grep -q "[^[:space:]]" "${doc}"; then
    continue
  fi

  if grep -q "^kind: PersistentVolumeClaim" "${doc}"; then
    pvc_name=$(awk '/^  name: /{print $2; exit}' "${doc}")
    if [ -n "${pvc_name}" ] && kubectl get pvc "${pvc_name}" -n "${CLUSTER_NAMESPACE}" >/dev/null 2>&1; then
      echo "Skipping existing PVC ${pvc_name}"
      continue
    fi
    echo "Applying PVC ${pvc_name}"
    kubectl apply -n "${CLUSTER_NAMESPACE}" -f "${doc}"
    continue
  fi

  cat "${doc}" >> "${NON_PVC_FILE}"
  echo "---" >> "${NON_PVC_FILE}"
done

if [ -s "${NON_PVC_FILE}" ]; then
  kubectl apply -n "${CLUSTER_NAMESPACE}" -f "${NON_PVC_FILE}"
fi

# Wait for deployment
echo "Waiting for deployment to be ready (up to 300s)..."
if kubectl wait --for=condition=available --timeout=300s deployment/${APP_NAME} -n ${CLUSTER_NAMESPACE} 2>/dev/null; then
  echo "✅ Deployment ready!"
else
  echo "⚠️  Deployment not ready after 300s. Check pod status:"
  kubectl get pods -n ${CLUSTER_NAMESPACE} -l ${APP_LABEL}
  echo ""
  echo "Rollback commands:"
  echo "  kubectl rollout undo deployment/${APP_NAME} -n ${CLUSTER_NAMESPACE}"
  echo "  See: docs/operations/rollback.md"
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
