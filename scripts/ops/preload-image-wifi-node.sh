#!/usr/bin/env bash
# Pre-load an image onto a node that has registry connectivity issues (e.g. WiFi nodes)
# Usage: ./preload-image-wifi-node.sh <NODE_IP> <IMAGE_NAME> [SSH_OPTS]

set -euo pipefail

NODE_IP=${1:-}
IMAGE_NAME=${2:-}
SSH_OPTS=${3:-}
WORKER_NODE="${WORKER_NODE:-${REGISTRY_ACCESS_NODE:-}}"

if [ -z "$NODE_IP" ]; then
  echo "Usage: $0 <NODE_IP> [IMAGE_NAME] [TARGET_NAME] [SSH_OPTS]"
  echo "Set WORKER_NODE or REGISTRY_ACCESS_NODE to a node with registry access."
  exit 1
fi

if [ -z "$IMAGE_NAME" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  if [ -x "${REPO_ROOT}/scripts/resolve-image.sh" ]; then
    IMAGE_NAME="$("${REPO_ROOT}/scripts/resolve-image.sh")"
  fi
fi

if [ -z "$WORKER_NODE" ]; then
  echo "ERROR: WORKER_NODE or REGISTRY_ACCESS_NODE must be set."
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
