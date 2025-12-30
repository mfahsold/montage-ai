#!/bin/bash
# Script to retrieve results from the latest Kubernetes job
set -e

NAMESPACE="montage-ai"
DEBUG_POD="debug-pvc"
LOCAL_OUTPUT_DIR="data/output"

echo "🔍 Checking for completed jobs in namespace '$NAMESPACE'..."

# Get the latest job name
LATEST_JOB=$(kubectl get jobs -n $NAMESPACE --sort-by=.metadata.creationTimestamp -o name | tail -n 1)

if [ -z "$LATEST_JOB" ]; then
    echo "❌ No jobs found."
    exit 1
fi

JOB_NAME=${LATEST_JOB#job.batch/}
echo "📋 Found latest job: $JOB_NAME"

# Check if debug pod exists
if ! kubectl get pod $DEBUG_POD -n $NAMESPACE > /dev/null 2>&1; then
    echo "❌ Debug pod '$DEBUG_POD' not found. Please deploy it first."
    exit 1
fi

echo "📂 Listing files in cluster output directory..."
kubectl exec -n $NAMESPACE $DEBUG_POD -- ls -lh /data/output

echo "⬇️  Downloading results to $LOCAL_OUTPUT_DIR..."
mkdir -p $LOCAL_OUTPUT_DIR

# Download all files from /data/output
# Note: kubectl cp doesn't support wildcards directly in the source path for some versions,
# so we list and copy individually or copy the directory.
# Copying the directory content:
kubectl cp -n $NAMESPACE $DEBUG_POD:/data/output/ $LOCAL_OUTPUT_DIR/

echo "✅ Download complete!"
ls -lh $LOCAL_OUTPUT_DIR
