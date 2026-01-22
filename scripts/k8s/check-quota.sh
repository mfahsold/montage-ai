#!/usr/bin/env bash
set -euo pipefail

# Preflight check for montage-ai distributed deploy
# Usage: ./scripts/k8s/check-quota.sh -n montage-ai

NAMESPACE=${1:-montage-ai}

echo "Checking ResourceQuota for namespace: $NAMESPACE"
kubectl -n "$NAMESPACE" get resourcequota -o wide || { echo "No ResourceQuota found in $NAMESPACE"; exit 1; }

echo "Checking current PVCs..."
kubectl -n "$NAMESPACE" get pvc -o wide

echo "Checking available storage vs requested (summary)"
kubectl -n "$NAMESPACE" get resourcequota storage-quota -o yaml | yq -r '.status'

cat <<'NOTE'
Quick validations (fail early):
- persistentvolumeclaims should be < quota
- requests.storage should be < quota
- Internal registry reachable from cluster nodes (test with curl to registry)

If any check fails, request SRE to either:
- increase ResourceQuota for the namespace, or
- pre-provision the PVs used by the distributed overlay.
NOTE
