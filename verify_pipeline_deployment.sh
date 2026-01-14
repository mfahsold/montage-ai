#!/bin/bash
# Montage AI Verification Script

NAMESPACE="montage-ai"
SVC_NAME="montage-ai-web"
URL="http://$SVC_NAME.$NAMESPACE.svc.cluster.local"

echo "Checking Montage AI Health..."
HEALTH=$(kubectl exec -n ci -it pods/$(kubectl get pods -n ci -l tekton.dev/pipelineRun=montage-ai-manual-build-4pfwq,tekton.dev/task=deploy-kubectl -o jsonpath='{.items[0].metadata.name}') -- curl -s $URL/health)

if [[ $HEALTH == *"healthy"* ]]; then
  echo "✅ Backend Health: OK"
else
  echo "❌ Backend Health: FAILED"
  echo "Response: $HEALTH"
fi

echo "Checking Frontend..."
FRONTEND=$(kubectl exec -n ci -it pods/$(kubectl get pods -n ci -l tekton.dev/pipelineRun=montage-ai-manual-build-4pfwq,tekton.dev/task=deploy-kubectl -o jsonpath='{.items[0].metadata.name}') -- curl -s $URL/ | grep "<title>")

if [[ $FRONTEND == *"Montage AI"* ]]; then
  echo "✅ Frontend: OK"
else
  echo "❌ Frontend: FAILED"
  echo "Response: $FRONTEND"
fi
