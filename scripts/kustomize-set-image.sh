#!/usr/bin/env bash
set -euo pipefail

# Set kustomize images across overlays to the resolved image
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="$1"
if [ -z "$IMAGE" ]; then
  IMAGE=$("$ROOT_DIR/scripts/resolve-image.sh")
fi

echo "Setting kustomize image to: $IMAGE"

# List of kustomize dirs to update
KUST_DIRS=(
  "$ROOT_DIR/deploy/k3s/base"
  "$ROOT_DIR/deploy/k3s/overlays/cluster"
)

for d in "${KUST_DIRS[@]}"; do
  if [ -d "$d" ]; then
    echo "-> Updating images in $d"
    pushd "$d" >/dev/null
    # attempting to replace the default image name with computed image
    kustomize edit set image montage-ai=${IMAGE} || true
    popd >/dev/null
  fi
done

echo "Kustomize images updated (modified kustomization files). Remember to commit these changes if desired."
