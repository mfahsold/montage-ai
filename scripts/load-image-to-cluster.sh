#!/usr/bin/env bash
set -euo pipefail

TAR_PATH=${1:-/tmp/montage-ai-canary.tar}
KUBECTL=${KUBECTL:-kubectl}
SSH_OPTS=${SSH_OPTS:-}

if [ ! -f "$TAR_PATH" ]; then
  echo "Image tar not found: $TAR_PATH" >&2
  exit 1
fi

# Get node IPs (internal)
mapfile -t NODES < <($KUBECTL get nodes -o jsonpath='{range .items[*]}{.status.addresses[?(@.type=="InternalIP")].address} {end}')

for ip in "${NODES[@]}"; do
  echo "Transferring to $ip..."
  scp $SSH_OPTS "$TAR_PATH" "$ip":/tmp/ || { echo "scp failed for $ip"; continue; }
  echo "Importing image on $ip..."
  ssh $SSH_OPTS "$ip" "sudo ctr images import /tmp/$(basename $TAR_PATH) && sudo ctr images ls | grep montage-ai || true"
  echo "Done for $ip"
done

echo "All nodes processed."
