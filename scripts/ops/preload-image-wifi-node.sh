#!/usr/bin/env bash
# Pre-load an image onto a node that has registry connectivity issues (e.g. WiFi nodes)
# Usage: ./preload-image-wifi-node.sh <NODE_IP> <IMAGE_NAME> [SSH_OPTS]

set -euo pipefail

NODE_IP=${1:-}
IMAGE_NAME=${2:-ghcr.io/mfahsold/montage-ai:latest}
SSH_OPTS=${3:-}
WORKER_NODE="codeai-worker-amd64" # Default worker that has registry access

if [ -z "$NODE_IP" ]; then
  echo "Usage: $0 <NODE_IP> [IMAGE_NAME] [TARGET_NAME] [SSH_OPTS]"
  exit 1
fi

TARGET_NAME=${3:-$IMAGE_NAME}
SSH_OPTS=${4:-}

echo "Pulling $IMAGE_NAME on $WORKER_NODE..."
ssh $SSH_OPTS "$WORKER_NODE" "docker pull $IMAGE_NAME"

echo "Streaming $IMAGE_NAME from $WORKER_NODE to $NODE_IP..."
# Save from worker and import directly into k3s containerd on target node
ssh $SSH_OPTS "$WORKER_NODE" "docker save $IMAGE_NAME" | \
  ssh $SSH_OPTS "$NODE_IP" "sudo ctr --address /run/k3s/containerd/containerd.sock -n k8s.io images import -"

if [ "$IMAGE_NAME" != "$TARGET_NAME" ]; then
  echo "Tagging image as $TARGET_NAME..."
  ssh $SSH_OPTS "$NODE_IP" "sudo ctr --address /run/k3s/containerd/containerd.sock -n k8s.io images tag $IMAGE_NAME $TARGET_NAME"
fi

echo "Image successfully pre-loaded on $NODE_IP."
echo "Ensure your deployment uses 'imagePullPolicy: IfNotPresent'."
