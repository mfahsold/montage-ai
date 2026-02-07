#!/usr/bin/env bash
set -euo pipefail

# Render keda-scaledobjects.yaml from keda-scaledobjects.yaml.in using base/cluster-config.env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/base/cluster-config.env"
IN_FILE="${SCRIPT_DIR}/overlays/cluster/keda-scaledobjects.yaml.in"
OUT_FILE="${SCRIPT_DIR}/overlays/cluster/keda-scaledobjects.yaml"

if [ ! -f "${IN_FILE}" ]; then
  echo "No input template ${IN_FILE}; nothing to render." >&2
  exit 0
fi

if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "ERROR: Missing env file ${ENV_FILE}; run 'make config' first." >&2
  exit 1
fi

# Export variables of interest for envsubst
export WORKER_QUEUE_SCALE_THRESHOLD
export WORKER_QUEUE_LIST_NAME
export WORKER_MIN_REPLICAS
export WORKER_MAX_REPLICAS
export REDIS_HOST
export REDIS_PORT
export CLUSTER_NAMESPACE

# Use envsubst to render
envsubst < "${IN_FILE}" > "${OUT_FILE}"

echo "Rendered ${OUT_FILE} from template."
