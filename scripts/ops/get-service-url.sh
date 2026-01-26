#!/bin/bash
# Discover the canonical external access URL for a service in the cluster.
# Usage: ./scripts/ops/get-service-url.sh <service-name> [namespace]

set -euo pipefail

SET_NS=${2:-default}
SVC_NAME=${1:-}

if [ -z "$SVC_NAME" ]; then
    echo "Usage: $0 <service-name> [namespace]"
    exit 1
fi

CONTROL_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

INGRESS_NAME=$(kubectl get ingress -n "$SET_NS" -o jsonpath="{.items[?(@.spec.rules[*].http.paths[*].backend.service.name=='$SVC_NAME')].metadata.name}" 2>/dev/null | head -n 1)
if [ -n "$INGRESS_NAME" ]; then
    INGRESS_HOST=$(kubectl get ingress -n "$SET_NS" "$INGRESS_NAME" -o jsonpath="{.spec.rules[0].host}")
    echo "--- Ingress detected ---"
    echo "URL:  http://$INGRESS_HOST"
    echo "Note: Ensure your /etc/hosts contains '$INGRESS_HOST' pointing to $CONTROL_IP"
    echo "      or use the wildcard record: *.fluxibri.lan -> $CONTROL_IP"
    exit 0
fi

LB_IP=$(kubectl get svc -n "$SET_NS" "$SVC_NAME" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
if [ -n "$LB_IP" ]; then
    PORT=$(kubectl get svc -n "$SET_NS" "$SVC_NAME" -o jsonpath='{.spec.ports[0].port}')
    echo "--- LoadBalancer detected ---"
    echo "URL:  http://$LB_IP:$PORT"
    exit 0
fi

NODE_PORT=$(kubectl get svc -n "$SET_NS" "$SVC_NAME" -o jsonpath='{.spec.ports[?(@.nodePort)].nodePort}' 2>/dev/null)
if [ -n "$NODE_PORT" ]; then
    echo "--- NodePort detected ---"
    echo "URL:  http://${CONTROL_IP}:${NODE_PORT}"
    exit 0
fi

echo "No external access (Ingress/NodePort) found for service '$SVC_NAME' in namespace '$SET_NS'."
echo "Internal ClusterIP: $(kubectl get svc -n "$SET_NS" "$SVC_NAME" -o jsonpath='{.spec.clusterIP}')"
exit 1
