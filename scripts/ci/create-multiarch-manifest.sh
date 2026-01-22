#!/usr/bin/env bash
set -euo pipefail

# Create a multi-arch manifest using buildx imagetools
# Requires: docker buildx imagetools (Docker CLI with imagetools)
# Usage: ./scripts/ci/create-multiarch-manifest.sh <manifest-tag> <amd64-image> <arm64-image>

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <manifest-tag> <amd64-image> <arm64-image>"
  exit 2
fi

MANIFEST_TAG=$1
AMD64_IMAGE=$2
ARM64_IMAGE=$3

echo "Creating manifest $MANIFEST_TAG from:\n  $AMD64_IMAGE\n  $ARM64_IMAGE"
docker buildx imagetools create --tag "$MANIFEST_TAG" "$AMD64_IMAGE" "$ARM64_IMAGE"

echo "Manifest created: $MANIFEST_TAG"
