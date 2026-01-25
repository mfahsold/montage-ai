#!/usr/bin/env bash
set -euo pipefail

# Simple registry health checks (defaults from deploy/k3s/config-global.yaml if present)
# Usage: ./scripts/registry_check.sh [registry-host]
# Default: use REGISTRY_HOST env var or 'localhost'
REGISTRY_HOST=${1:-${REGISTRY_HOST:-localhost}}
PORTS=(${REGISTRY_PORT:-5000} 30500)

echo "Checking host: $REGISTRY_HOST"

# Prefer the Python-based check which gives more structured output
if command -v python3 >/dev/null 2>&1 && [ -f "$(dirname "$0")/check-registry.py" ]; then
  python3 "$(dirname "$0")/check-registry.py" "$REGISTRY_HOST"
  exit 0
fi

ping -c 2 "$REGISTRY_HOST" || true
for p in "${PORTS[@]}"; do
  echo "Testing TCP $p"
  nc -zv "$REGISTRY_HOST" "$p" || true
  echo "Testing http://$REGISTRY_HOST:$p/v2/"
  curl -v --max-time 10 "http://$REGISTRY_HOST:$p/v2/" || true
  echo "Testing https://$REGISTRY_HOST:$p/v2/"
  curl -v --max-time 10 "https://$REGISTRY_HOST:$p/v2/" || true
done

echo "Done. If the registry is behind TLS, ensure the certs are installed or provide the correct port/hostname."
