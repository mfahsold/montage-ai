#!/usr/bin/env bash
set -euo pipefail

# Seed a small preview clip into the cluster montage-input PVC by copying
# a local test media file into a running pod that mounts the PVC.
# Usage: scripts/ops/seed-test-media.sh [--namespace NAMESPACE] [local_file]

NAMESPACE=${1:-montage-ai}
LOCAL_FILE=${2:-test_data/preview.mp4}

# Ensure local file exists — try to generate a tiny preview clip if ffmpeg available
if [[ ! -f "${LOCAL_FILE}" ]]; then
  echo "No local test clip found at ${LOCAL_FILE}. Attempting to generate a 1s MP4 with ffmpeg..."
  if command -v ffmpeg >/dev/null 2>&1; then
    mkdir -p $(dirname "${LOCAL_FILE}")
    ffmpeg -y -f lavfi -i color=c=black:s=320x240:d=1 -f lavfi -i anullsrc -shortest -c:v libx264 -t 1 -pix_fmt yuv420p "${LOCAL_FILE}"
    echo "Generated ${LOCAL_FILE}"
  else
    echo "ffmpeg not available and no local clip found; please provide a small MP4 at ${LOCAL_FILE}" >&2
    exit 1
  fi
fi

# Find a pod that has the montage-input PVC mounted (prefer worker pods)
POD=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/component=worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -z "${POD}" ]]; then
  POD=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/component=web-ui -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
fi
if [[ -z "${POD}" ]]; then
  echo "No web or worker pod found in namespace ${NAMESPACE}; cannot seed PVC" >&2
  exit 2
fi

TARGET_DIR="/data/input"
echo "Copying ${LOCAL_FILE} -> ${NAMESPACE}/${POD}:${TARGET_DIR}/"
kubectl -n "${NAMESPACE}" exec "${POD}" -- mkdir -p "${TARGET_DIR}"
kubectl -n "${NAMESPACE}" cp "${LOCAL_FILE}" "${NAMESPACE}/${POD}:${TARGET_DIR}/"

echo "Seeded test media into PVC via pod ${POD}. Verify files:"
kubectl -n "${NAMESPACE}" exec "${POD}" -- ls -la "${TARGET_DIR}/"

echo "Done. Run ./scripts/ci/run-dev-smoke.sh to exercise autoscaling and end-to-end flow."