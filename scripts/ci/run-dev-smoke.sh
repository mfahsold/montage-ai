#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run-dev-smoke.sh [options] [TAG]

Options:
  --image IMAGE          Full image reference (overrides --registry-url/--image-name/--tag)
  --tag TAG              Image tag (default: dev-local or positional TAG)
  --registry-url URL     Registry host:port (default from config-global.yaml)
  --image-name NAME      Image name (default from config-global.yaml)
  --namespace NS         Kubernetes namespace (default from config-global.yaml)
  --overlay PATH         Optional kustomize overlay to apply before smoke
  --health-url URL       In-pod health URL (default: http://127.0.0.1:80/api/status)
  -h, --help             Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_LIB="${REPO_ROOT}/scripts/ops/lib/config_global.sh"

TAG=""
IMAGE=""
OVERLAY=""
REGISTRY_URL_ARG=""
IMAGE_NAME_ARG=""
NAMESPACE_ARG=""
HEALTH_URL="${HEALTH_URL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"
      shift
      ;;
    --tag)
      TAG="${2:-}"
      shift
      ;;
    --registry-url)
      REGISTRY_URL_ARG="${2:-}"
      shift
      ;;
    --image-name)
      IMAGE_NAME_ARG="${2:-}"
      shift
      ;;
    --namespace)
      NAMESPACE_ARG="${2:-}"
      shift
      ;;
    --overlay)
      OVERLAY="${2:-}"
      shift
      ;;
    --health-url)
      HEALTH_URL="${2:-}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "$TAG" ]]; then
        TAG="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
  shift
done

if [[ -f "$CONFIG_LIB" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_LIB"
fi
if command -v config_global_export >/dev/null 2>&1; then
  eval "$(config_global_export)"
fi

: "${CLUSTER_NAMESPACE:=montage-ai}"
: "${REGISTRY_URL:=127.0.0.1:30500}"
: "${IMAGE_NAME:=montage-ai}"

if [[ -n "$NAMESPACE_ARG" ]]; then
  CLUSTER_NAMESPACE="$NAMESPACE_ARG"
fi
if [[ -n "$REGISTRY_URL_ARG" ]]; then
  REGISTRY_URL="$REGISTRY_URL_ARG"
fi
if [[ -n "$IMAGE_NAME_ARG" ]]; then
  IMAGE_NAME="$IMAGE_NAME_ARG"
fi

if [[ -z "$IMAGE" ]]; then
  TAG="${TAG:-dev-local}"
  IMAGE="${REGISTRY_URL}/${IMAGE_NAME}:${TAG}"
fi

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:80/api/status}"

echo "Running dev autoscale smoke (image=${IMAGE}, namespace=${CLUSTER_NAMESPACE})"

if [[ -n "$OVERLAY" ]]; then
  echo "Applying overlay: ${OVERLAY}"
  kubectl -n "${CLUSTER_NAMESPACE}" apply -k "${OVERLAY}"
fi

# ensure image is set
kubectl -n "${CLUSTER_NAMESPACE}" set image deploy/montage-ai-web montage-ai="${IMAGE}" --record
kubectl -n "${CLUSTER_NAMESPACE}" set image deploy/montage-ai-worker worker="${IMAGE}" --record

# rollout
kubectl -n "${CLUSTER_NAMESPACE}" rollout restart deployment montage-ai-web montage-ai-worker
kubectl -n "${CLUSTER_NAMESPACE}" rollout status deploy/montage-ai-web --timeout=3m || true
kubectl -n "${CLUSTER_NAMESPACE}" rollout status deploy/montage-ai-worker --timeout=3m || true


# quick health
kubectl -n "${CLUSTER_NAMESPACE}" get pods -l app=montage-ai -o wide
# Exec into a running web pod (prefer a Ready pod) to check health
WEB_POD=$(kubectl -n "${CLUSTER_NAMESPACE}" get pods -l app.kubernetes.io/component=web-ui -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | awk '{print $1}')
if [[ -n "${WEB_POD}" ]]; then
  echo "Using web pod: ${WEB_POD}"
  kubectl -n "${CLUSTER_NAMESPACE}" exec -it "${WEB_POD}" -- curl -fsS -m 5 "${HEALTH_URL}" | jq -C '.' || true
else
  echo "No running web pod found in namespace ${CLUSTER_NAMESPACE}, skipping health check"
fi

# run the repo's opt-in smoke (runner must have cluster networking or port-forward)
export RUN_SCALE_TESTS=1
pytest -q tests/integration/test_queue_scaling.py -q

echo "Dev autoscale smoke completed"
