#!/bin/bash
# Validate config-global.yaml: ensure all <...> placeholders are replaced
set -euo pipefail

CONFIG="${1:-deploy/k3s/config-global.yaml}"

if [ ! -f "$CONFIG" ]; then
  echo "❌ Config file not found: $CONFIG"
  echo "   Copy the example first: cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml"
  exit 1
fi

PLACEHOLDERS=$(grep -c '<[A-Z_]*>' "$CONFIG" 2>/dev/null || true)

if [ "$PLACEHOLDERS" -gt 0 ]; then
  echo "❌ Found $PLACEHOLDERS unresolved placeholder(s) in $CONFIG:"
  grep -n '<[A-Z_]*>' "$CONFIG"
  echo ""
  echo "Replace all <...> values before deploying."
  echo "See: deploy/k3s/config-global.yaml.example for reference."
  exit 1
fi

echo "✅ No placeholders found in $CONFIG"
