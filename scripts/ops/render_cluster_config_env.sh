#!/usr/bin/env bash
# Render deploy/k3s/base/cluster-config.env from deploy/k3s/config-global.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIB_CONFIG_GLOBAL="$REPO_ROOT/scripts/ops/lib/config_global.sh"
CONFIG_GLOBAL_PATH="${CONFIG_GLOBAL:-$REPO_ROOT/deploy/k3s/config-global.yaml}"
ENV_OUT="${ENV_OUT:-$REPO_ROOT/deploy/k3s/base/cluster-config.env}"
DEPLOY_CONFIG_ENV="${DEPLOY_CONFIG_ENV:-$REPO_ROOT/deploy/config.env}"

if [ -f "$LIB_CONFIG_GLOBAL" ]; then
  # shellcheck disable=SC1090
  source "$LIB_CONFIG_GLOBAL"
fi

if [ -f "$DEPLOY_CONFIG_ENV" ]; then
  # shellcheck disable=SC1090
  source "$DEPLOY_CONFIG_ENV"
fi

if command -v config_global_export >/dev/null 2>&1; then
  eval "$(config_global_export "$CONFIG_GLOBAL_PATH")"
fi

: "${REGISTRY_URL:=}"
: "${IMAGE_NAME:=montage-ai}"
: "${IMAGE_TAG:=latest}"
: "${IMAGE_FULL:=${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}}"
: "${CLUSTER_NAMESPACE:=montage-ai}"
: "${MONTAGE_HOSTNAME:=${APP_DOMAIN:-montage-ai.local}}"
: "${STORAGE_CLASS_DEFAULT:=local-path}"
: "${STORAGE_CLASS_NFS:=nfs-client}"
: "${NFS_SERVER:=}"
: "${NFS_PATH:=}"
: "${REDIS_HOST:=redis.${CLUSTER_NAMESPACE}.svc.cluster.local}"
: "${REDIS_PORT:=6379}"
: "${OPENAI_API_BASE:=}"
: "${OPENAI_API_KEY:=}"
: "${OPENAI_MODEL:=auto}"
: "${OPENAI_VISION_MODEL:=}"
: "${OLLAMA_HOST:=}"
: "${FFMPEG_MCP_ENDPOINT:=}"
: "${FFMPEG_MCP_HOST:=}"
: "${FFMPEG_MCP_PORT:=8080}"
: "${WORKER_MIN_REPLICAS:=1}"
: "${WORKER_MAX_REPLICAS:=8}"

# KEDA thresholds (list lengths used by ScaledObjects)
: "${WORKER_QUEUE_SCALE_THRESHOLD:=10}"
: "${WORKER_HEAVY_QUEUE_SCALE_THRESHOLD:=20}"
: "${REGISTRY_NAMESPACE:=}"
: "${CLUSTER_DOMAIN:=}"

if [ -z "${REGISTRY_URL:-}" ]; then
  if [ -n "${REGISTRY_HOST:-}" ]; then
    REGISTRY_URL="${REGISTRY_HOST}${REGISTRY_PORT:+:${REGISTRY_PORT}}"
  elif [ -n "${REGISTRY_NAMESPACE}" ] || [ -n "${K3S_CLUSTER_DOMAIN}" ] || [ -n "${CLUSTER_DOMAIN}" ]; then
    registry_ns="${REGISTRY_NAMESPACE:-registry}"
    cluster_domain="${K3S_CLUSTER_DOMAIN:-${CLUSTER_DOMAIN:-cluster.local}}"
    REGISTRY_URL="registry.${registry_ns}.svc.${cluster_domain}:5000"
  fi
fi

if [ -z "${OLLAMA_HOST:-}" ] && [ -n "${CLUSTER_NAMESPACE:-}" ]; then
  cluster_domain="${K3S_CLUSTER_DOMAIN:-${CLUSTER_DOMAIN:-cluster.local}}"
  OLLAMA_HOST="http://ollama.${CLUSTER_NAMESPACE}.svc.${cluster_domain}:11434"
fi

if [ -z "${FFMPEG_MCP_ENDPOINT:-}" ] && [ -n "${CLUSTER_NAMESPACE:-}" ]; then
  cluster_domain="${K3S_CLUSTER_DOMAIN:-${CLUSTER_DOMAIN:-cluster.local}}"
  FFMPEG_MCP_ENDPOINT="http://ffmpeg-mcp.${CLUSTER_NAMESPACE}.svc.${cluster_domain}:${FFMPEG_MCP_PORT}"
fi

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
WORKER_QUEUE_SCALE_THRESHOLD=${WORKER_QUEUE_SCALE_THRESHOLD}
WORKER_HEAVY_QUEUE_SCALE_THRESHOLD=${WORKER_HEAVY_QUEUE_SCALE_THRESHOLD}
WORKER_MIN_REPLICAS=${WORKER_MIN_REPLICAS}
WORKER_MAX_REPLICAS=${WORKER_MAX_REPLICAS}
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT}
OPENAI_API_BASE=${OPENAI_API_BASE}
OPENAI_API_KEY=${OPENAI_API_KEY}
OPENAI_MODEL=${OPENAI_MODEL}
OPENAI_VISION_MODEL=${OPENAI_VISION_MODEL}
OLLAMA_HOST=${OLLAMA_HOST}
FFMPEG_MCP_ENDPOINT=${FFMPEG_MCP_ENDPOINT}
FFMPEG_MCP_HOST=${FFMPEG_MCP_HOST}
FFMPEG_MCP_PORT=${FFMPEG_MCP_PORT}
EOF_ENV

echo "Wrote ${ENV_OUT}"
