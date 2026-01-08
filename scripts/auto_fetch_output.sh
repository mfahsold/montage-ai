#!/usr/bin/env bash
set -euo pipefail

# Automatically fetch the latest rendered montage (or a specific job's file)
# from the running montage-ai worker pod to your local machine.
#
# Usage:
#   scripts/auto_fetch_output.sh                # fetches the newest *.mp4 from /tmp/montage_output
#   JOB_ID=20260108_165802 scripts/auto_fetch_output.sh   # fetches the newest file matching the job id
#
# Optional env vars:
#   NAMESPACE       Kubernetes namespace (default: montage-ai)
#   WORKER_LABEL    Pod selector (default: app=montage-ai-worker)
#   REMOTE_DIR      Remote output dir (default: /tmp/montage_output)
#   LOCAL_DIR       Local download dir (default: ./downloads/outputs)

NAMESPACE="${NAMESPACE:-montage-ai}"
WORKER_LABEL="${WORKER_LABEL:-app=montage-ai-worker}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/montage_output}"
LOCAL_DIR="${LOCAL_DIR:-./downloads/outputs}"
JOB_ID="${JOB_ID:-}"  # optional filter

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required on the client side" >&2
  exit 1
fi

POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l "$WORKER_LABEL" -o jsonpath='{.items[0].metadata.name}')
if [[ -z "$POD_NAME" ]]; then
  echo "No worker pod found with label $WORKER_LABEL in namespace $NAMESPACE" >&2
  exit 1
fi

# Pick the newest mp4, optionally filtered by JOB_ID
if [[ -n "$JOB_ID" ]]; then
  REMOTE_FILE=$(kubectl exec -n "$NAMESPACE" "$POD_NAME" -- /bin/sh -c "ls -1t $REMOTE_DIR/*$JOB_ID*.mp4 2>/dev/null | head -1")
else
  REMOTE_FILE=$(kubectl exec -n "$NAMESPACE" "$POD_NAME" -- /bin/sh -c "ls -1t $REMOTE_DIR/*.mp4 2>/dev/null | head -1")
fi

if [[ -z "$REMOTE_FILE" ]]; then
  echo "No matching .mp4 files found in $REMOTE_DIR (JOB_ID=$JOB_ID)" >&2
  exit 1
fi

mkdir -p "$LOCAL_DIR"
BASENAME=$(basename "$REMOTE_FILE")
LOCAL_PATH="$LOCAL_DIR/$BASENAME"

echo "Fetching $REMOTE_FILE from pod $POD_NAME to $LOCAL_PATH ..."
kubectl cp "$NAMESPACE/$POD_NAME:$REMOTE_FILE" "$LOCAL_PATH"
echo "Done. Saved to $LOCAL_PATH"
