#!/usr/bin/env bash
# Seed local data directories (input, music) into cluster PVCs.
# Requires a running deployment (uses kubectl cp via a web-ui pod).
#
# Usage:
#   scripts/ops/k8s-seed-data.sh [--namespace montage-ai] [--input data/input] [--music data/music]
set -euo pipefail

NAMESPACE="montage-ai"
INPUT_DIR="data/input"
MUSIC_DIR="data/music"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace|-n) NAMESPACE="$2"; shift 2 ;;
    --input)        INPUT_DIR="$2"; shift 2 ;;
    --music)        MUSIC_DIR="$2"; shift 2 ;;
    *)              shift ;;
  esac
done

# Find a pod with PVC mounts
find_pod() {
  local pod
  pod=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/component=web-ui \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [[ -z "${pod}" ]]; then
    pod=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/component=worker \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  fi
  echo "${pod}"
}

POD=$(find_pod)
if [[ -z "${POD}" ]]; then
  echo "No web-ui or worker pod found in namespace ${NAMESPACE}." >&2
  echo "Deploy first: make -C deploy/k3s deploy-cluster" >&2
  exit 1
fi

echo "Using pod: ${NAMESPACE}/${POD}"

copy_dir() {
  local src="$1" dest="$2" label="$3"
  if [[ ! -d "${src}" ]]; then
    echo "Skipping ${label}: ${src} not found"
    return
  fi
  local count
  count=$(find "${src}" -type f | wc -l)
  if [[ "${count}" -eq 0 ]]; then
    echo "Skipping ${label}: ${src} is empty"
    return
  fi
  echo "Copying ${label} (${count} files) -> ${POD}:${dest}/"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- mkdir -p "${dest}"
  # Copy each file (kubectl cp doesn't reliably copy directory contents)
  find "${src}" -type f | while read -r f; do
    kubectl -n "${NAMESPACE}" cp "${f}" "${NAMESPACE}/${POD}:${dest}/$(basename "${f}")"
  done
  echo "  Done: ${label}"
}

copy_dir "${INPUT_DIR}" "/data/input" "input videos"
copy_dir "${MUSIC_DIR}" "/data/music" "music tracks"

echo ""
echo "Cluster PVC contents:"
kubectl -n "${NAMESPACE}" exec "${POD}" -- sh -c 'echo "=== /data/input ===" && ls /data/input/ 2>/dev/null || echo "(empty)"; echo "=== /data/music ===" && ls /data/music/ 2>/dev/null || echo "(empty)"'
echo ""
echo "Ready to submit a test job."
