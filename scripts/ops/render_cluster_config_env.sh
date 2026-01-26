#!/usr/bin/env bash
# Render deploy/k3s/base/cluster-config.env from deploy/k3s/config-global.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIB_CONFIG_GLOBAL="$REPO_ROOT/scripts/ops/lib/config_global.sh"
CONFIG_GLOBAL_PATH="${CONFIG_GLOBAL:-$REPO_ROOT/deploy/k3s/config-global.yaml}"
ENV_OUT="${ENV_OUT:-$REPO_ROOT/deploy/k3s/base/cluster-config.env}"

if [ -f "$LIB_CONFIG_GLOBAL" ]; then
  # shellcheck disable=SC1090
  source "$LIB_CONFIG_GLOBAL"
fi

if command -v config_global_export >/dev/null 2>&1; then
  eval "$(config_global_export "$CONFIG_GLOBAL_PATH")"
fi

: "${REGISTRY_URL:=registry.registry.svc.cluster.local:5000}"
: "${IMAGE_NAME:=montage-ai}"
: "${IMAGE_TAG:=latest}"
: "${IMAGE_FULL:=${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}}"
: "${CLUSTER_NAMESPACE:=montage-ai}"
: "${MONTAGE_HOSTNAME:=montage-ai.local}"
: "${STORAGE_CLASS_DEFAULT:=local-path}"
: "${STORAGE_CLASS_NFS:=nfs-client}"
: "${NFS_SERVER:=}"
: "${NFS_PATH:=}"

cat > "$ENV_OUT" <<EOF_ENV
REGISTRY_URL=${REGISTRY_URL}
IMAGE_NAME=${IMAGE_NAME}
IMAGE_TAG=${IMAGE_TAG}
IMAGE_FULL=${IMAGE_FULL}
CLUSTER_NAMESPACE=${CLUSTER_NAMESPACE}
MONTAGE_HOSTNAME=${MONTAGE_HOSTNAME}
STORAGE_CLASS_DEFAULT=${STORAGE_CLASS_DEFAULT}
STORAGE_CLASS_NFS=${STORAGE_CLASS_NFS}
NFS_SERVER=${NFS_SERVER}
NFS_PATH=${NFS_PATH}
EOF_ENV

echo "Wrote ${ENV_OUT}"
