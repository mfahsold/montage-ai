#!/bin/bash
# Pre-flight checks for Montage AI Kubernetes deployment
# Run before deploy.sh to catch common issues early.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_GLOBAL="${CONFIG_GLOBAL:-${SCRIPT_DIR}/config-global.yaml}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

echo "Pre-flight checks for Montage AI K8s deployment"
echo "================================================"
echo ""

# 1. Check required tools
for cmd in kubectl kustomize make; do
  if command -v "$cmd" &>/dev/null; then
    echo -e "${GREEN}[OK]${NC} $cmd found: $(command -v "$cmd")"
  else
    echo -e "${RED}[FAIL]${NC} $cmd not found. Install it before deploying."
    ERRORS=$((ERRORS + 1))
  fi
done

# 2. Check config-global.yaml exists
if [ -f "$CONFIG_GLOBAL" ]; then
  echo -e "${GREEN}[OK]${NC} config-global.yaml exists"
else
  echo -e "${RED}[FAIL]${NC} config-global.yaml not found at ${CONFIG_GLOBAL}"
  echo "       Run: cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml"
  ERRORS=$((ERRORS + 1))
fi

# 3. Check for unreplaced placeholders
if [ -f "$CONFIG_GLOBAL" ]; then
  PLACEHOLDERS=$(grep -c '<[A-Z_]*>' "$CONFIG_GLOBAL" 2>/dev/null || true)
  if [ "${PLACEHOLDERS:-0}" -gt 0 ]; then
    echo -e "${RED}[FAIL]${NC} ${PLACEHOLDERS} unreplaced placeholder(s) in config-global.yaml:"
    grep -n '<[A-Z_]*>' "$CONFIG_GLOBAL" | while IFS= read -r line; do
      echo "       ${line}"
    done
    ERRORS=$((ERRORS + 1))
  else
    echo -e "${GREEN}[OK]${NC} No unreplaced placeholders"
  fi
fi

# 4. Check kubectl connectivity
if command -v kubectl &>/dev/null; then
  if kubectl cluster-info &>/dev/null; then
    echo -e "${GREEN}[OK]${NC} kubectl connected to cluster"
  else
    echo -e "${RED}[FAIL]${NC} kubectl cannot connect to cluster"
    ERRORS=$((ERRORS + 1))
  fi
fi

# 5. Check StorageClass availability
if command -v kubectl &>/dev/null && kubectl cluster-info &>/dev/null; then
  SC_COUNT=$(kubectl get storageclass --no-headers 2>/dev/null | wc -l || echo "0")
  if [ "${SC_COUNT:-0}" -gt 0 ]; then
    echo -e "${GREEN}[OK]${NC} ${SC_COUNT} StorageClass(es) available"
  else
    echo -e "${YELLOW}[WARN]${NC} No StorageClass found. Pods requiring PVCs will stay Pending."
    echo "       For single-node: K3s includes local-path by default."
    echo "       For multi-node: Install an RWX-capable provisioner (NFS, Longhorn)."
    echo "       See: docs/cluster-deploy.md#storage-setup"
  fi
fi

# 6. Check node architectures
if command -v kubectl &>/dev/null && kubectl cluster-info &>/dev/null; then
  ACTUAL_ARCHS=$(kubectl get nodes -o jsonpath='{.items[*].status.nodeInfo.architecture}' 2>/dev/null | tr ' ' '\n' | sort -u)
  if [ -n "$ACTUAL_ARCHS" ]; then
    echo -e "${GREEN}[OK]${NC} Cluster node architecture(s): $(echo $ACTUAL_ARCHS | tr '\n' ', ' | sed 's/,$//')"
    echo "       Verify these match your config-global.yaml node definitions."
    echo "       Run: kubectl get nodes -o wide"
  else
    echo -e "${YELLOW}[WARN]${NC} Could not detect node architectures"
  fi
fi

# 7. Check kustomize builds
if command -v kustomize &>/dev/null && [ -d "${SCRIPT_DIR}/base" ]; then
  if kustomize build --load-restrictor LoadRestrictionsNone "${SCRIPT_DIR}/base" >/dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} kustomize build (base) succeeds"
  else
    echo -e "${YELLOW}[WARN]${NC} kustomize build (base) failed — run 'make config' first"
  fi
fi

echo ""
echo "================================================"
if [ "$ERRORS" -gt 0 ]; then
  echo -e "${RED}${ERRORS} check(s) failed.${NC} Fix the issues above before deploying."
  exit 1
else
  echo -e "${GREEN}All checks passed.${NC} Ready to deploy."
fi
