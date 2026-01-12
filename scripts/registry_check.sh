#!/usr/bin/env bash
set -euo pipefail

# Simple registry health checks for 192.168.1.12
REGISTRY_HOST=${1:-192.168.1.12}
PORTS=(5000 30500)

echo "Checking host: $REGISTRY_HOST"
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