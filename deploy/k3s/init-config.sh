#!/usr/bin/env bash
# Generate config-global.yaml from example with sensible defaults (issue #128).
#
# All values can be overridden via environment variables:
#   REGISTRY_HOST=ghcr.io CLUSTER_NAMESPACE=prod make -C deploy/k3s init-config
#
# Without env vars, uses safe local-dev defaults (k3d/minikube/kind compatible).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE="${SCRIPT_DIR}/config-global.yaml.example"
OUTPUT="${SCRIPT_DIR}/config-global.yaml"

if [ ! -f "${EXAMPLE}" ]; then
  echo "❌ ERROR: ${EXAMPLE} not found."
  exit 1
fi

if [ -f "${OUTPUT}" ]; then
  echo "⚠️  ${OUTPUT} already exists."
  read -r -p "   Overwrite? [y/N] " answer
  if [[ ! "${answer}" =~ ^[Yy]$ ]]; then
    echo "   Aborted."
    exit 0
  fi
fi

# Defaults: safe for local k3d/minikube/kind clusters.
# Set REGISTRY_HOST="" REGISTRY_PORT="" REGISTRY_URL="" for no-registry mode
# (k3d image import / kind load docker-image workflow).
_REGISTRY_HOST="${REGISTRY_HOST-localhost}"
_REGISTRY_PORT="${REGISTRY_PORT-5000}"
if [ -n "${REGISTRY_URL+set}" ]; then
  _REGISTRY_URL="${REGISTRY_URL}"
elif [ -n "${_REGISTRY_HOST}" ]; then
  _REGISTRY_URL="${_REGISTRY_HOST}${_REGISTRY_PORT:+:${_REGISTRY_PORT}}"
else
  _REGISTRY_URL=""
fi
_CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
_CLUSTER_DOMAIN="${CLUSTER_DOMAIN:-cluster.local}"
_CONTROL_PLANE_IP="${CONTROL_PLANE_IP:-127.0.0.1}"
_GPU_NODE_IP="${GPU_NODE_IP:-}"
_NFS_SERVER_IP="${NFS_SERVER_IP:-}"

if [ -z "${_REGISTRY_URL}" ]; then
  echo "Generating ${OUTPUT} (no-registry mode — use k3d image import / kind load):"
else
  echo "Generating ${OUTPUT} with:"
  echo "  REGISTRY_URL:      ${_REGISTRY_URL}"
fi
echo "  CLUSTER_NAMESPACE: ${_CLUSTER_NAMESPACE}"
echo "  CLUSTER_DOMAIN:    ${_CLUSTER_DOMAIN}"
echo "  CONTROL_PLANE_IP:  ${_CONTROL_PLANE_IP}"
echo ""

cp "${EXAMPLE}" "${OUTPUT}"

# Replace all placeholders
sed -i \
  -e "s|<REGISTRY_HOST>|${_REGISTRY_HOST}|g" \
  -e "s|<REGISTRY_PORT>|${_REGISTRY_PORT}|g" \
  -e "s|<REGISTRY_URL>|${_REGISTRY_URL}|g" \
  -e "s|<CLUSTER_NAMESPACE>|${_CLUSTER_NAMESPACE}|g" \
  -e "s|<CLUSTER_DOMAIN>|${_CLUSTER_DOMAIN}|g" \
  -e "s|<CONTROL_PLANE_IP>|${_CONTROL_PLANE_IP}|g" \
  -e "s|<GPU_NODE_IP>|${_GPU_NODE_IP}|g" \
  -e "s|<NFS_SERVER_IP>|${_NFS_SERVER_IP}|g" \
  "${OUTPUT}"

# Remove any remaining angle-bracket placeholders (set to empty)
sed -i 's|<[A-Z_]*>||g' "${OUTPUT}"

# Verify no placeholders remain
if grep -qE '<[A-Z_]+>' "${OUTPUT}"; then
  echo "⚠️  WARNING: Some placeholders remain:"
  grep -n '<[A-Z_]*>' "${OUTPUT}"
else
  echo "✅ ${OUTPUT} generated — no unresolved placeholders."
fi

echo ""
echo "Next steps:"
echo "  1. Review: \$EDITOR ${OUTPUT}"
echo "  2. Render:  make -C deploy/k3s config"
echo "  3. Deploy:  make -C deploy/k3s deploy-cluster"
