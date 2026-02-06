#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
CGPU_CONFIG_PATH="${CGPU_CONFIG_PATH:-$HOME/.config/cgpu/config.json}"
CGPU_SESSION_PATH="${CGPU_SESSION_PATH:-$HOME/.config/cgpu/state/session.json}"

if ! command -v cgpu >/dev/null 2>&1; then
  echo "cgpu not found in PATH. Install cgpu first." >&2
  exit 1
fi

if [ ! -f "$CGPU_CONFIG_PATH" ]; then
  echo "Missing cgpu config: $CGPU_CONFIG_PATH" >&2
  exit 1
fi

echo "Starting cgpu OAuth flow..."

echo "- Keep this terminal open until login completes."
if ! cgpu connect; then
  echo "cgpu connect failed. Re-run after completing OAuth." >&2
  exit 1
fi

if [ ! -f "$CGPU_SESSION_PATH" ]; then
  echo "Session file not found: $CGPU_SESSION_PATH" >&2
  exit 1
fi

echo "Updating cluster secret cgpu-credentials in namespace $CLUSTER_NAMESPACE..."

kubectl -n "$CLUSTER_NAMESPACE" create secret generic cgpu-credentials \
  --from-file=config.json="$CGPU_CONFIG_PATH" \
  --from-file=session.json="$CGPU_SESSION_PATH" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Restarting worker + cgpu-server deployments to pick up new session..."

kubectl -n "$CLUSTER_NAMESPACE" rollout restart \
  deploy/montage-ai-worker \
  deploy/cgpu-server

echo "Done. Validate with: kubectl -n $CLUSTER_NAMESPACE exec deploy/montage-ai-worker -- cgpu status"
