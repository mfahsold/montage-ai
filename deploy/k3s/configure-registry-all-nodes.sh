#!/bin/bash
# =============================================================================
# Configure K3s nodes to allow HTTP registry
# =============================================================================
# Deploys registries.yaml to all cluster nodes and restarts K3s services.
#
# Usage:
#   # Set your nodes via environment variables:
#   CONTROL_PLANE=user@10.0.0.1 WORKERS="user@10.0.0.2 user@10.0.0.3" ./configure-registry-all-nodes.sh 10.0.0.1 5000
#
#   # Or edit the CONTROL_PLANE and WORKERS variables below
# =============================================================================

set -e

REGISTRY_HOST="${1:-your-registry}"
REGISTRY_PORT="${2:-5000}"

# =============================================================================
# NODE CONFIGURATION - Edit for your cluster
# =============================================================================
# Control plane: user@ip
CONTROL_PLANE="${CONTROL_PLANE:-user@control-plane}"

# Workers: space-separated list of user@ip
WORKERS_STR="${WORKERS:-}"
if [ -n "$WORKERS_STR" ]; then
  read -ra WORKERS <<< "$WORKERS_STR"
else
  WORKERS=(
    # Add your worker nodes here:
    # "user@10.0.0.2"
    # "user@10.0.0.3"
  )
fi

# Validate configuration
if [ "$CONTROL_PLANE" = "user@control-plane" ]; then
  echo "ERROR: CONTROL_PLANE not configured!"
  echo ""
  echo "Usage: CONTROL_PLANE=user@10.0.0.1 WORKERS=\"user@10.0.0.2 user@10.0.0.3\" $0 <registry-ip> <registry-port>"
  exit 1
fi

if [ "$REGISTRY_HOST" = "your-registry" ]; then
  echo "ERROR: Registry host not specified!"
  echo ""
  echo "Usage: $0 <registry-ip> <registry-port>"
  exit 1
fi

# registries.yaml template for HTTP-only registry with insecure skip
REGISTRIES_YAML=$(cat <<EOF
mirrors:
  "${REGISTRY_HOST}:${REGISTRY_PORT}":
    endpoint:
      - "http://${REGISTRY_HOST}:${REGISTRY_PORT}"
configs:
  "${REGISTRY_HOST}:${REGISTRY_PORT}":
    tls:
      insecure_skip_verify: true
EOF
)

echo "════════════════════════════════════════════════════════════"
echo "Configuring K3s Registry for HTTP: ${REGISTRY_HOST}:${REGISTRY_PORT}"
echo "════════════════════════════════════════════════════════════"
echo ""

configure_node() {
  local NODE=$1
  local SERVICE=$2
  echo "📝 Configuring ${NODE}..."

  # Create/update registries.yaml
  ssh "${NODE}" "cat > /tmp/registries.yaml" <<< "${REGISTRIES_YAML}"

  # Move to K3s config directory
  ssh "${NODE}" "sudo mv /tmp/registries.yaml /etc/rancher/k3s/registries.yaml"

  # Restart K3s service
  echo "  🔄 Restarting ${SERVICE}..."
  ssh "${NODE}" "sudo systemctl restart ${SERVICE}"

  # Wait for service to be ready
  sleep 3

  echo "  ✅ ${NODE} configured"
}

# Configure control plane first
echo "🔧 Control Plane:"
configure_node "${CONTROL_PLANE}" "k3s"
echo ""

# Configure all worker nodes
if [ ${#WORKERS[@]} -gt 0 ]; then
  echo "🔧 Worker Nodes:"
  for WORKER in "${WORKERS[@]}"; do
    configure_node "${WORKER}" "k3s-agent"
  done
  echo ""
fi

# Wait for cluster to stabilize
echo "⏳ Waiting for cluster to stabilize..."
sleep 5

# Verify all nodes are Ready
echo ""
echo "🔍 Verifying cluster status..."
kubectl get nodes

# Check registry is reachable from a pod
echo ""
echo "🔗 Testing registry connectivity from pod..."
kubectl run --rm -it registry-test --image=alpine:latest --restart=Never 2>/dev/null -- \
  sh -c "wget -q -O- http://${REGISTRY_HOST}:${REGISTRY_PORT}/v2/ && echo '✅ Registry reachable'" || {
  echo "⚠️  Registry test pod failed (may still be OK if nodes are restarting)"
}

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Registry Configuration Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Deploy montage-ai: ./deploy/k3s/deploy.sh cluster"
echo "  2. Watch pods: kubectl get pods -n montage-ai -w"
echo "  3. Check logs: kubectl logs -n montage-ai -l app.kubernetes.io/name=montage-ai -f"
echo ""
