#!/bin/bash
set -e

# Delete and recreate job
kubectl delete job montage-ai-cinema-trailer -n montage-ai --ignore-not-found --wait=true
kubectl wait --for=delete pod -l job-name=montage-ai-cinema-trailer -n montage-ai --timeout=60s || true
sleep 2
kubectl apply -f deploy/k3s/job-cinema-trailer.yaml

# Wait for pod to be running
echo "Waiting for pod to be running..."

# Get the newest running pod name (retry in case pod list is still empty)
POD=""
for i in {1..20}; do
  POD=$(kubectl get pods -l job-name=montage-ai-cinema-trailer -n montage-ai --field-selector=status.phase=Running --sort-by=.metadata.creationTimestamp -o name | tail -n 1 | sed 's#pod/##')
  if [ -n "$POD" ]; then
    break
  fi
  sleep 2
done

if [ -z "$POD" ]; then
  echo "Failed to find trailer pod"
  exit 1
fi

kubectl wait --for=condition=ready pod "$POD" -n montage-ai --timeout=60s
echo "Pod: $POD"

echo "Checking moviepy version..."
kubectl exec $POD -n montage-ai -- pip show moviepy || echo "Failed to check moviepy version"

# Copy source code
echo "Preparing source code..."
TEMP_SRC=$(mktemp -d)
echo "Staging code in $TEMP_SRC..."

# Use rsync to copy source code excluding __pycache__ and other artifacts
# We ignore errors because we expect some permission denied on root-owned files we don't want anyway
rsync -av --exclude='__pycache__' --exclude='*.pyc' src/ "$TEMP_SRC/src/" || true

echo "Copying code to pod..."
kubectl cp "$TEMP_SRC/src" "$POD":/app/ -n montage-ai

# Clean up
rm -rf "$TEMP_SRC"

echo "Code synced. Tailing logs..."
kubectl logs $POD -n montage-ai -f
