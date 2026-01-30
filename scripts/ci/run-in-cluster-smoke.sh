#!/usr/bin/env bash
set -euo pipefail

# Render the smoke-runner job from template and run it in-cluster.
# Usage: ./scripts/ci/run-in-cluster-smoke.sh [--image IMAGE] [--namespace NS]

IMAGE_ARG=""
NAMESPACE_ARG="montage-ai"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE_ARG="$2"; shift 2;;
    --namespace)
      NAMESPACE_ARG="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--image IMAGE] [--namespace NS]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OVERLAY_DIR="$REPO_ROOT/deploy/k3s/overlays/cluster"
TEMPLATE="$OVERLAY_DIR/smoke-runner-job.yaml.in"
RENDERED="/tmp/montage-ai-smoke-runner.yaml"

# Ensure cluster config has been rendered
make -C deploy/k3s config >/dev/null
source deploy/k3s/base/cluster-config.env

IMAGE_TO_USE="${IMAGE_ARG:-${IMAGE_FULL:-}}"
if [[ -z "$IMAGE_TO_USE" ]]; then
  echo "ERROR: No image specified. Provide via --image or configure IMAGE_FULL in deploy/k3s/base/cluster-config.env" >&2
  exit 1
fi

# Render template
export IMAGE_FULL="$IMAGE_TO_USE"
envsubst < "$TEMPLATE" > "$RENDERED"

# Apply job
kubectl -n "$NAMESPACE_ARG" apply -f "$RENDERED"
JOB_NAME="montage-ai-smoke-runner"

# Wait for job to complete and stream logs from the pod
echo "Waiting for Job pod to start..."
POD=""
for i in $(seq 1 60); do
  POD=$(kubectl -n "$NAMESPACE_ARG" get pods -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [[ -n "$POD" ]]; then break; fi
  sleep 1
done

if [[ -z "$POD" ]]; then
  echo "ERROR: Job pod didn't start in time" >&2
  kubectl -n "$NAMESPACE_ARG" get jobs,po -l job-name=${JOB_NAME}
  exit 2
fi

echo "Streaming logs for pod $POD (ctrl-c to stop streaming, job will continue)..."
kubectl -n "$NAMESPACE_ARG" logs -f "$POD"

# Wait for job completion status
echo "Waiting for job completion..."
set +e
kubectl -n "$NAMESPACE_ARG" wait --for=condition=complete job/${JOB_NAME} --timeout=300s
rc_wait=$?
kubectl -n "$NAMESPACE_ARG" get job ${JOB_NAME} -o yaml
set -e

# Collect final pod logs
echo "Final logs:"
kubectl -n "$NAMESPACE_ARG" logs "$POD" || true

# Cleanup job (keep for debugging if it failed)
if [[ $rc_wait -eq 0 ]]; then
  echo "Job completed successfully; cleaning up job resources..."
  kubectl -n "$NAMESPACE_ARG" delete job ${JOB_NAME} --ignore-not-found
else
  echo "Job failed or timed out; keeping job for investigation" >&2
  kubectl -n "$NAMESPACE_ARG" describe job ${JOB_NAME}
  exit $rc_wait
fi
