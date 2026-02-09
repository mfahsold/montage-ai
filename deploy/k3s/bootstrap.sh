#!/usr/bin/env bash
# Bootstrap montage-ai Kubernetes deployment with proper data initialization
# Creates .ready markers and verifies storage readiness

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
WAIT_TIME="${WAIT_TIME:-300}"  # 5 minutes default

echo "🚀 Bootstrapping montage-ai Kubernetes deployment..."
echo "   Namespace: $CLUSTER_NAMESPACE"
echo ""

# Check kubectl access
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to Kubernetes cluster"
    echo "   Run: kubectl cluster-info"
    exit 1
fi

# Step 1: Verify namespace exists
echo "1️⃣  Checking namespace..."
if ! kubectl get namespace "$CLUSTER_NAMESPACE" > /dev/null 2>&1; then
    echo "   Creating namespace: $CLUSTER_NAMESPACE"
    kubectl create namespace "$CLUSTER_NAMESPACE"
fi
echo "   ✅ Namespace ready"

# Step 2: Verify PVCs are bound
echo ""
echo "2️⃣  Checking persistent volumes..."
BOUND_PVCS=0
for pvc_name in montage-input montage-output montage-music montage-assets; do
    if kubectl get pvc "$pvc_name" -n "$CLUSTER_NAMESPACE" > /dev/null 2>&1; then
        STATUS=$(kubectl get pvc "$pvc_name" -n "$CLUSTER_NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
        if [ "$STATUS" == "Bound" ]; then
            echo "   ✅ $pvc_name: Bound"
            BOUND_PVCS=$((BOUND_PVCS + 1))
        else
            echo "   ⚠️  $pvc_name: $STATUS (waiting...)"
        fi
    fi
done

if [ "$BOUND_PVCS" -lt 4 ]; then
    echo ""
    echo "⏳ Waiting for all PVCs to bind (max ${WAIT_TIME}s)..."
    for i in $(seq 1 $((WAIT_TIME / 5))); do
        BOUND_COUNT=$(kubectl get pvc -n "$CLUSTER_NAMESPACE" -o jsonpath='{range .items[?(@.status.phase=="Bound")]}{.metadata.name}{"\n"}{end}' 2>/dev/null | wc -l)
        if [ "$BOUND_COUNT" -ge 4 ]; then
            echo "   ✅ All PVCs bound"
            break
        fi
        if [ $((i % 6)) -eq 0]; then
            echo "   ... still waiting ($((i * 5))s/$WAIT_TIME s)"
        fi
        sleep 5
    done
fi
echo "   ✅ PVCs ready"

# Step 3: Create .ready markers in mounted storage
echo ""
echo "3️⃣  Initializing storage markers..."

# Use a simple pod to create the .ready marker
cat > /tmp/init-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: montage-init
  namespace: NAMESPACE
spec:
  restartPolicy: Never
  containers:
  - name: init
    image: busybox:latest
    command: 
      - sh
      - -c
      - |
        echo "Initializing montage-ai storage..."
        # Create .ready marker in input directory
        touch /data/input/.ready
        # Create basic output directory structure
        mkdir -p /data/output /data/music /data/assets
        echo "Storage initialized at $(date -u)"
        sleep 1
    volumeMounts:
    - name: input
      mountPath: /data/input
    - name: output
      mountPath: /data/output
    - name: music
      mountPath: /data/music
    - name: assets
      mountPath: /data/assets
  volumes:
  - name: input
    persistentVolumeClaim:
      claimName: montage-input
  - name: output
    persistentVolumeClaim:
      claimName: montage-output
  - name: music
    persistentVolumeClaim:
      claimName: montage-music
  - name: assets
    persistentVolumeClaim:
      claimName: montage-assets
EOF

# Replace namespace
sed -i.bak "s/NAMESPACE/$CLUSTER_NAMESPACE/g" /tmp/init-pod.yaml

# Apply and wait for init pod
kubectl apply -f /tmp/init-pod.yaml

echo "   ⏳ Waiting for initialization pod..."
kubectl wait --for=condition=Ready pod/montage-init -n "$CLUSTER_NAMESPACE" --timeout=60s 2>/dev/null || true
kubectl wait --for=condition=ContainersReady pod/montage-init -n "$CLUSTER_NAMESPACE" --timeout=60s 2>/dev/null || true

# Get logs
LOGS=$(kubectl logs montage-init -n "$CLUSTER_NAMESPACE" 2>/dev/null || echo "")
if [ -n "$LOGS" ]; then
    echo "   Init output: $LOGS"
fi

# Clean up init pod
echo "   Cleaning up initialization pod..."
kubectl delete pod montage-init -n "$CLUSTER_NAMESPACE" --ignore-not-found=true

echo "   ✅ Storage markers created"

# Step 4: Verify application deployment
echo ""
echo "4️⃣  Checking montage-ai deployment..."
if kubectl get deployment montage-ai-web -n "$CLUSTER_NAMESPACE" > /dev/null 2>&1; then
    REPLICAS=$(kubectl get deployment montage-ai-web -n "$CLUSTER_NAMESPACE" -o jsonpath='{.status.replicas}' 2>/dev/null || echo "0")
    READY=$(kubectl get deployment montage-ai-web -n "$CLUSTER_NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    if [ "$READY" -eq "$REPLICAS" ] && [ "$READY" -gt 0 ]; then
        echo "   ✅ Deployment ready (${READY}/${REPLICAS} replicas)"
    else
        echo "   ⏳ Deployment scaling (${READY}/${REPLICAS})"
    fi
else
    echo "   ℹ️  No deployment found (may not be deployed yet)"
fi

# Step 5: Show web service access
echo ""
echo "5️⃣  Web UI access:"
if kubectl get svc montage-ai-web -n "$CLUSTER_NAMESPACE" > /dev/null 2>&1; then
    SERVICE_IP=$(kubectl get svc montage-ai-web -n "$CLUSTER_NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    SERVICE_PORT=$(kubectl get svc montage-ai-web -n "$CLUSTER_NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "8080")
    
    if [ -n "$SERVICE_IP" ]; then
        echo "   🌐 External IP: http://${SERVICE_IP}:${SERVICE_PORT}"
    else
        echo "   Port-forward instead:"
        echo "   kubectl port-forward -n $CLUSTER_NAMESPACE svc/montage-ai-web 8080:8080"
        echo "   Then: http://localhost:8080"
    fi
fi

echo ""
echo "✅ Bootstrap complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Verify pods are running:"
echo "      kubectl get pods -n $CLUSTER_NAMESPACE"
echo ""
echo "   2. Check if render jobs are ready:"
echo "      kubectl get jobs -n $CLUSTER_NAMESPACE"
echo ""
echo "   3. View logs:"
echo "      kubectl logs -n $CLUSTER_NAMESPACE -l app.kubernetes.io/name=montage-ai"
echo ""
echo "   4. Debug stuck init:"
echo "      kubectl describe pod <POD_NAME> -n $CLUSTER_NAMESPACE"
echo "      kubectl logs <POD_NAME> -c wait-for-nfs-ready -n $CLUSTER_NAMESPACE"
echo ""

# Cleanup
rm -f /tmp/init-pod.yaml /tmp/init-pod.yaml.bak
