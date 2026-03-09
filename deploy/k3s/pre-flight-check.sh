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
ALLOW_LEGACY_CLUSTER_STATE="${ALLOW_LEGACY_CLUSTER_STATE:-false}"

_namespace_from_config() {
  awk '
    /^cluster:/ {in_cluster=1; next}
    in_cluster == 1 && /^[^[:space:]]/ {in_cluster=0}
    in_cluster == 1 && $1 == "namespace:" {
      gsub(/"/, "", $2)
      print $2
      exit
    }
  ' "$CONFIG_GLOBAL"
}

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

# 8. Check live cluster drift against rollout contract
if command -v kubectl &>/dev/null && kubectl cluster-info &>/dev/null; then
  LIVE_NAMESPACE=""
  if [ -f "${SCRIPT_DIR}/base/cluster-config.env" ]; then
    # shellcheck disable=SC1090
    source "${SCRIPT_DIR}/base/cluster-config.env"
    LIVE_NAMESPACE="${CLUSTER_NAMESPACE:-}"
  fi
  if [ -z "${LIVE_NAMESPACE}" ]; then
    LIVE_NAMESPACE="$(_namespace_from_config)"
  fi
  if [ -z "${LIVE_NAMESPACE}" ]; then
    LIVE_NAMESPACE="montage-ai"
  fi

  if kubectl get namespace "${LIVE_NAMESPACE}" >/dev/null 2>&1; then
    if kubectl get networkpolicy allow-all -n "${LIVE_NAMESPACE}" >/dev/null 2>&1; then
      if [ "${ALLOW_LEGACY_CLUSTER_STATE}" = "true" ]; then
        echo -e "${YELLOW}[WARN]${NC} Legacy allow-all NetworkPolicy still exists in namespace ${LIVE_NAMESPACE}"
      else
        echo -e "${RED}[FAIL]${NC} Legacy allow-all NetworkPolicy exists in namespace ${LIVE_NAMESPACE}"
        echo "       Delete it to enforce default-deny policy baseline."
        ERRORS=$((ERRORS + 1))
      fi
    else
      echo -e "${GREEN}[OK]${NC} No legacy allow-all NetworkPolicy found in live namespace ${LIVE_NAMESPACE}"
    fi

    LIVE_RWO_PVCS=""
    for pvc_name in \
      "${PVC_INPUT_NAME:-montage-ai-input-nfs}" \
      "${PVC_OUTPUT_NAME:-montage-ai-output-nfs}" \
      "${PVC_MUSIC_NAME:-montage-ai-music-nfs}" \
      "${PVC_ASSETS_NAME:-montage-ai-assets-nfs}"; do
      if kubectl get pvc "${pvc_name}" -n "${LIVE_NAMESPACE}" >/dev/null 2>&1; then
        access_mode="$(kubectl get pvc "${pvc_name}" -n "${LIVE_NAMESPACE}" -o jsonpath='{.spec.accessModes[*]}')"
        if echo "${access_mode}" | grep -q "ReadWriteOnce"; then
          LIVE_RWO_PVCS+="${pvc_name} "
        fi
      fi
    done

    if [ -n "${LIVE_RWO_PVCS}" ]; then
      if [ "${ALLOW_LEGACY_CLUSTER_STATE}" = "true" ]; then
        echo -e "${YELLOW}[WARN]${NC} Live PVCs still use RWO: ${LIVE_RWO_PVCS}"
      else
        echo -e "${RED}[FAIL]${NC} Live PVCs still use ReadWriteOnce: ${LIVE_RWO_PVCS}"
        echo "       Existing PVC access mode is immutable; run RWX migration before go-live sign-off."
        echo "       See: docs/operations/RWX_PVC_MIGRATION_RUNBOOK.md"
        ERRORS=$((ERRORS + 1))
      fi
    else
      echo -e "${GREEN}[OK]${NC} Live PVC access mode contract satisfies RWX baseline"
    fi
  else
    echo -e "${YELLOW}[WARN]${NC} Namespace ${LIVE_NAMESPACE} not found yet (skipping live drift checks)"
  fi
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
    echo -e "${GREEN}[OK]${NC} ${SC_COUNT} StorageClass(es) available:"
    # Show details: name, provisioner, and default marker
    kubectl get storageclass --no-headers 2>/dev/null | while IFS= read -r line; do
      SC_NAME=$(echo "$line" | awk '{print $1}')
      SC_PROV=$(echo "$line" | awk '{print $2}')
      SC_DEFAULT=""
      if echo "$line" | grep -q "(default)"; then
        SC_DEFAULT=" (default)"
      fi
      echo "       - ${SC_NAME} [${SC_PROV}]${SC_DEFAULT}"
    done
    echo "       Shared media volumes require RWX (ReadWriteMany) for multi-node clusters."
  else
    echo -e "${YELLOW}[WARN]${NC} No StorageClass found. Pods requiring PVCs will stay Pending."
    echo "       For single-node: K3s includes local-path by default."
    echo "       For multi-node: Install an RWX-capable provisioner (NFS, Longhorn)."
    echo "       See: docs/cluster-deploy.md#storage-setup"
  fi
fi

# 6. Check node architectures (compare config vs actual cluster)
if command -v kubectl &>/dev/null && kubectl cluster-info &>/dev/null; then
  ACTUAL_ARCHS=$(kubectl get nodes -o jsonpath='{.items[*].status.nodeInfo.architecture}' 2>/dev/null | tr ' ' '\n' | sort -u)
  if [ -n "$ACTUAL_ARCHS" ]; then
    echo -e "${GREEN}[OK]${NC} Cluster node architecture(s): $(echo $ACTUAL_ARCHS | tr '\n' ', ' | sed 's/,$//')"
    # Compare against configured architectures in config-global.yaml
    if [ -f "$CONFIG_GLOBAL" ]; then
      CONFIGURED_ARCHS=$(grep 'arch:' "$CONFIG_GLOBAL" 2>/dev/null | sed 's/.*arch:[[:space:]]*["'\'']\?\([a-z0-9]*\)["'\'']\?.*/\1/' | sort -u)
      if [ -n "$CONFIGURED_ARCHS" ]; then
        MISMATCH=""
        for cfg_arch in $CONFIGURED_ARCHS; do
          if ! echo "$ACTUAL_ARCHS" | grep -q "^${cfg_arch}$"; then
            MISMATCH="${MISMATCH} ${cfg_arch}"
          fi
        done
        if [ -n "$MISMATCH" ]; then
          echo -e "${RED}[FAIL]${NC} Architecture mismatch: config-global.yaml defines${MISMATCH} but cluster only has: $(echo $ACTUAL_ARCHS | tr '\n' ' ')"
          echo "       Update config-global.yaml node arch values or add matching nodes."
          ERRORS=$((ERRORS + 1))
        else
          echo -e "${GREEN}[OK]${NC} Configured architectures match cluster nodes"
        fi
      fi
    fi
  else
    echo -e "${YELLOW}[WARN]${NC} Could not detect node architectures"
  fi
fi

# 7. Check kustomize builds
if command -v kustomize &>/dev/null && [ -d "${SCRIPT_DIR}/base" ]; then
  RENDERED_MANIFEST="$(mktemp)"
  trap 'rm -f "${RENDERED_MANIFEST}"' EXIT

  if kustomize build --load-restrictor LoadRestrictionsNone "${SCRIPT_DIR}/base" >"${RENDERED_MANIFEST}" 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} kustomize build (base) succeeds"

    # Hard gate: permissive allow-all network policy must not exist.
    if grep -q "name: allow-all" "${RENDERED_MANIFEST}"; then
      echo -e "${RED}[FAIL]${NC} Found deprecated permissive NetworkPolicy (allow-all) in rendered base manifests"
      echo "       Replace with default-deny + explicit allows before go-live."
      ERRORS=$((ERRORS + 1))
    else
      echo -e "${GREEN}[OK]${NC} No deprecated allow-all NetworkPolicy found"
    fi

    # Hard gate: shared Montage PVCs must use ReadWriteMany in canonical manifests.
    RWO_PVCS=$(awk '
      BEGIN {doc=""; kind=""; name=""; in_modes=0; has_rwo=0}
      /^---/ {
        if (kind == "PersistentVolumeClaim" && has_rwo == 1 && name ~ /^montage-/) {
          print name
        }
        doc=""; kind=""; name=""; in_modes=0; has_rwo=0
        next
      }
      /^kind:/ {kind=$2}
      /^metadata:/ {in_metadata=1; next}
      in_metadata == 1 && /^  name:/ {name=$2; in_metadata=0}
      /^  accessModes:/ {in_modes=1; next}
      in_modes == 1 && /^    - ReadWriteOnce/ {has_rwo=1}
      in_modes == 1 && /^  [a-zA-Z]/ {in_modes=0}
      END {
        if (kind == "PersistentVolumeClaim" && has_rwo == 1 && name ~ /^montage-/) {
          print name
        }
      }
    ' "${RENDERED_MANIFEST}" | sort -u)

    if [ -n "${RWO_PVCS}" ]; then
      echo -e "${RED}[FAIL]${NC} Found montage PVCs with ReadWriteOnce in rendered base manifests:"
      while IFS= read -r pvc_name; do
        [ -n "${pvc_name}" ] && echo "       - ${pvc_name}"
      done <<< "${RWO_PVCS}"
      echo "       Shared media contract requires RWX (ReadWriteMany) for cluster rollout."
      ERRORS=$((ERRORS + 1))
    else
      echo -e "${GREEN}[OK]${NC} Montage PVC access mode contract is RWX in rendered base manifests"
    fi

    # Hard gate: canonical deployment must ship montage-specific observability artifacts.
    if ! grep -q "kind: ServiceMonitor" "${RENDERED_MANIFEST}"; then
      echo -e "${RED}[FAIL]${NC} ServiceMonitor missing from rendered base manifests"
      ERRORS=$((ERRORS + 1))
    else
      echo -e "${GREEN}[OK]${NC} ServiceMonitor present in rendered base manifests"
    fi

    if ! grep -q "kind: PrometheusRule" "${RENDERED_MANIFEST}"; then
      echo -e "${RED}[FAIL]${NC} PrometheusRule missing from rendered base manifests"
      ERRORS=$((ERRORS + 1))
    else
      echo -e "${GREEN}[OK]${NC} PrometheusRule present in rendered base manifests"
    fi
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
