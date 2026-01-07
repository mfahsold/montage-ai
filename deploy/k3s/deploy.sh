#!/bin/bash
# Deploy montage-ai to K3s cluster

set -e

NAMESPACE="montage-ai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Deploying montage-ai to K3s..."

# Apply manifests
kubectl apply -f "${SCRIPT_DIR}/montage-ai.yaml"

# Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/montage-ai -n ${NAMESPACE}

# Show status
echo ""
echo "=== Deployment Status ==="
kubectl get pods -n ${NAMESPACE}
kubectl get svc -n ${NAMESPACE}
kubectl get ingress -n ${NAMESPACE}

echo ""
echo "✓ Montage AI deployed successfully!"
echo ""
echo "Access points:"
echo "  • Web UI: http://montage.local (add to /etc/hosts)"
echo "  • Port forward: kubectl port-forward -n ${NAMESPACE} svc/montage-ai 5000:5000"
echo ""
echo "View logs:"
echo "  kubectl logs -n ${NAMESPACE} -l app=montage-ai -f"
