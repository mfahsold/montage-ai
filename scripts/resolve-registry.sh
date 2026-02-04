#!/usr/bin/env bash
# Print resolved REGISTRY_URL (uses deploy/k3s/config-global.yaml when present)
set -euo pipefail
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
CONFIG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090
source "${CONFIG_ROOT}/scripts/common.sh"

if [ -n "${REGISTRY_URL:-}" ]; then
  echo "${REGISTRY_URL}"
  exit 0
fi

echo "REGISTRY_URL not set. Configure deploy/config.env or deploy/k3s/config-global.yaml." >&2
exit 1
