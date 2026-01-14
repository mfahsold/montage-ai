#!/bin/bash
set -e
NS="montage-ai"
SVC="montage-ai-web"
echo "Verifying Montage AI Deployment..."
kubectl get pods -n $NS
kubectl run verify-temp --image=curlimages/curl:8.1.2 -n $NS --rm -it --restart=Never -- curl -s http://$SVC/health
kubectl run verify-temp-web --image=curlimages/curl:8.1.2 -n $NS --rm -it --restart=Never -- curl -s http://$SVC/ | grep "<title>"
