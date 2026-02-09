#!/bin/bash
# Test cluster deployment idempotency: deploy twice and verify no destructive changes.
# Usage: ./scripts/test-cluster-idempotency.sh [NAMESPACE]
set -euo pipefail

NAMESPACE="${1:-${CLUSTER_NAMESPACE:-montage-ai}}"
DEPLOY_DIR="deploy/k3s"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Testing cluster deployment idempotency (namespace: ${NAMESPACE})"
echo "================================================================"
echo ""

# Snapshot current state
echo "1. Capturing pre-deploy state..."
PVC_BEFORE=$(kubectl get pvc -n "$NAMESPACE" -o json 2>/dev/null || echo '{"items":[]}')
CM_BEFORE=$(kubectl get configmap -n "$NAMESPACE" -o json 2>/dev/null || echo '{"items":[]}')

# First deploy
echo "2. Running first deploy..."
make -C "$DEPLOY_DIR" deploy-cluster
echo ""

# Snapshot after first deploy
PVC_AFTER1=$(kubectl get pvc -n "$NAMESPACE" -o json 2>/dev/null)
PODS_AFTER1=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)

# Second deploy (idempotency test)
echo "3. Running second deploy (idempotency test)..."
make -C "$DEPLOY_DIR" deploy-cluster
echo ""

# Snapshot after second deploy
PVC_AFTER2=$(kubectl get pvc -n "$NAMESPACE" -o json 2>/dev/null)
PODS_AFTER2=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)

# Verify PVCs unchanged
echo "4. Verifying PVCs unchanged..."
PVC_NAMES1=$(echo "$PVC_AFTER1" | python3 -c "import sys,json; print('\n'.join(sorted(i['metadata']['name'] for i in json.load(sys.stdin).get('items',[]))))" 2>/dev/null || true)
PVC_NAMES2=$(echo "$PVC_AFTER2" | python3 -c "import sys,json; print('\n'.join(sorted(i['metadata']['name'] for i in json.load(sys.stdin).get('items',[]))))" 2>/dev/null || true)

ERRORS=0

if [ "$PVC_NAMES1" = "$PVC_NAMES2" ]; then
  echo -e "   ${GREEN}[OK]${NC} PVCs identical after re-deploy"
else
  echo -e "   ${RED}[FAIL]${NC} PVC set changed between deploys"
  ERRORS=$((ERRORS + 1))
fi

# Check all PVCs are Bound
PENDING=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null | grep -v Bound | wc -l || echo "0")
if [ "${PENDING:-0}" -eq 0 ]; then
  echo -e "   ${GREEN}[OK]${NC} All PVCs are Bound"
else
  echo -e "   ${YELLOW}[WARN]${NC} ${PENDING} PVC(s) not Bound"
fi

# Check pods are running
NOT_RUNNING=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | grep -v Running | grep -v Completed | wc -l || echo "0")
if [ "${NOT_RUNNING:-0}" -eq 0 ]; then
  echo -e "   ${GREEN}[OK]${NC} All pods Running/Completed"
else
  echo -e "   ${YELLOW}[WARN]${NC} ${NOT_RUNNING} pod(s) not in Running state"
fi

echo ""
echo "================================================================"
if [ "$ERRORS" -gt 0 ]; then
  echo -e "${RED}Idempotency test FAILED (${ERRORS} error(s))${NC}"
  exit 1
else
  echo -e "${GREEN}Idempotency test PASSED${NC}"
fi
