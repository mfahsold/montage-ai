#!/bin/bash
# Auto-detect cluster registry endpoint when defaults are used.
# Exports REGISTRY_HOST/REGISTRY_PORT/REGISTRY_URL/IMAGE_FULL.

set -e

if [[ -z "${REGISTRY_HOST}" || "${REGISTRY_HOST}" == "registry.example.com" ]]; then
  if command -v kubectl >/dev/null 2>&1; then
    nodeport=$(kubectl get svc -n registry registry -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || true)
    if [[ -n "$nodeport" ]]; then
      nodeip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null | awk '{print $1}')
      if [[ -n "$nodeip" ]]; then
        REGISTRY_HOST="$nodeip"
        REGISTRY_PORT="$nodeport"
      fi
    fi
  fi
fi

REGISTRY_URL="${REGISTRY_HOST}${REGISTRY_PORT:+:${REGISTRY_PORT}}"
IMAGE_FULL="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"

export REGISTRY_HOST REGISTRY_PORT REGISTRY_URL IMAGE_FULL
