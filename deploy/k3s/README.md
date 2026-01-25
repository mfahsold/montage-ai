# Kubernetes Deployment Guide

> **Single source of truth for K8s/K3s deployment.**

## Quick Start

```bash
# 1. Render cluster config
make -C deploy/k3s config

# 2. Build & push multi-arch image
cd deploy/k3s
./build-and-push.sh

# 3. Access Web UI (choose one)
# If you have ingress configured:
open "https://<MONTAGE_HOSTNAME>"

# Or port-forward locally:
CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
LOCAL_PORT="${LOCAL_PORT:-5000}"
kubectl port-forward -n "$CLUSTER_NAMESPACE" svc/montage-ai-web "${LOCAL_PORT}:8080"
open "http://localhost:${LOCAL_PORT}"
```

Set defaults used throughout this guide:

```bash
export CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
export MONTAGE_HOSTNAME="${MONTAGE_HOSTNAME:-montage-ai.local}"
```

## Clean Deploy (Fluxibri-core aligned)

Use the canonical, simplified deploy flow (build + push + apply overlay):

```bash
# Copy and edit canonical config
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
$EDITOR deploy/k3s/config-global.yaml

# Render cluster-config.env
make -C deploy/k3s config

# Build & push
cd deploy/k3s
./build-and-push.sh

# Deploy dev overlay
./deploy.sh dev
```

Notes:

- Update `deploy/k3s/config-global.yaml` to change registry, tag, or storage defaults.
- `make -C deploy/k3s config` regenerates `deploy/k3s/base/cluster-config.env`.

That's it. For manual control, see sections below.

---

## Prerequisites

- Kubernetes cluster (K3s, K8s, EKS, GKE, etc.)
- `kubectl` configured with cluster access
- Container registry accessible from cluster

## Directory Structure

```
deploy/k3s/
├── base/                    # Base manifests (all overlays use these)
│   ├── namespace.yaml       # montage-ai namespace
│   ├── configmap.yaml       # Environment configuration
│   ├── deployment.yaml      # Web UI deployment
│   ├── pvc.yaml             # Storage claims (local-path)
│   ├── nfs-pv.yaml          # NFS PersistentVolumes (distributed)
│   ├── job.yaml             # One-off render job
│   ├── cronjob.yaml         # Scheduled renders
│   └── kustomization.yaml   # Base kustomization
├── overlays/
│   ├── dev/                 # Fast preview (360p, minimal resources)
│   ├── staging/             # Balanced quality
│   ├── production/          # High quality + AMD GPU
│   ├── amd/                 # AMD GPU (VAAPI) acceleration
│   ├── jetson/              # NVIDIA Jetson (NVENC)
│   ├── gpu/                 # Generic NVIDIA GPU
│   ├── distributed/         # Multi-node GPU with NFS
│   └── distributed-parallel/# Indexed Job shards
└── app/                     # Full app deployment (ingress, monitoring)
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    └── kustomization.yaml
```

---

## Deployment Commands

### Option 1: Makefile (Recommended)

```bash
# Render cluster config
make -C deploy/k3s config

# Deploy specific overlay
make -C deploy/k3s deploy-production

# Other overlays
make -C deploy/k3s deploy-dev
make -C deploy/k3s deploy-staging
```

### Option 2: Shell Scripts

```bash
cd deploy/k3s

# Build & push to cluster registry
./build-and-push.sh

# Deploy (default: base)
./deploy.sh

# Deploy specific environment
./deploy.sh dev
./deploy.sh staging
./deploy.sh production

# Undeploy
./undeploy.sh
```

### Option 3: kubectl/kustomize Direct

```bash
# Render cluster-config.env first
make -C deploy/k3s config

# Base deployment
kubectl apply -k deploy/k3s/base/

# With overlay
kubectl apply -k deploy/k3s/overlays/production/

# Full app deployment (includes ingress, monitoring)
kubectl apply -k deploy/k3s/app/
```

---

## Environment Overlays

| Overlay | Quality | Resources | Use Case |
|---------|---------|-----------|----------|
| `base` | Standard | 4Gi/2CPU | Generic deployment |
| `dev` | Preview (360p) | 2Gi/500m | Fast iteration |
| `staging` | Standard | 4Gi/2CPU | QA testing |
| `production` | High | 8Gi/4CPU | Production renders |
| `amd` | High + VAAPI | 8Gi/4CPU | AMD GPU encoding |
| `jetson` | Standard + NVENC | 4Gi | Edge/Jetson devices |
| `distributed` | High | NFS storage | Multi-node GPU |

---

## GPU Acceleration

### Available GPU Resources

| Node (example) | GPU Type | Resource Key | Encoder |
|------|----------|--------------|---------|
| gpu-node-nvidia | NVIDIA Jetson | `nvidia.com/gpu: 1` | NVENC |
| gpu-node-amd | AMD Radeon | `amd.com/gpu: 1` | VAAPI |

### Deploy with GPU

```bash
# Check available GPUs
kubectl get nodes -o custom-columns='NAME:.metadata.name,NVIDIA:.status.allocatable.nvidia\.com/gpu,AMD:.status.allocatable.amd\.com/gpu'

# AMD GPU (VAAPI)
kubectl apply -k deploy/k3s/overlays/amd/

# NVIDIA Jetson (NVENC)
kubectl apply -k deploy/k3s/overlays/jetson/

# Generic NVIDIA
kubectl apply -k deploy/k3s/overlays/gpu/
```

---

## Storage Configuration

### Local Storage (Single Node)

Default: `local-path` StorageClass (K3s default)

```yaml
# In pvc.yaml
storageClassName: local-path  # K3s
storageClassName: standard    # GKE
storageClassName: gp2         # AWS EBS
```

### NFS Storage (Multi-Node / Distributed)

For multi-node GPU rendering, use NFS:

```bash
# 1. Setup NFS server exports
sudo mkdir -p /mnt/nfs-montage/{input,music,output,assets}
echo "/mnt/nfs-montage *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a

# 2. Update nfs-pv.yaml with your NFS server IP
# 3. Deploy distributed overlay
kubectl apply -k deploy/k3s/overlays/distributed/
```

---

## Node Labels (Required)

Apply these labels so overlays can schedule without hardcoded hostnames:

```bash
# Storage (hostPath PVs)
kubectl label node <node> fluxibri.ai/storage-primary=true

# Registry hosting + build access
kubectl label node <node> fluxibri.ai/registry-host=true
kubectl label node <node> fluxibri.ai/registry-access=true

# GPU classes (overlays/amd, overlays/jetson)
kubectl label node <node> accelerator=amd-gpu
kubectl label node <node> accelerator=nvidia-jetson
```

---

## Access Methods

### Port Forward (Development)

```bash
CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
LOCAL_PORT="${LOCAL_PORT:-5000}"
kubectl port-forward -n "$CLUSTER_NAMESPACE" svc/montage-ai-web "${LOCAL_PORT}:8080"
open "http://localhost:${LOCAL_PORT}"
```

### Ingress (Production)

```bash
# Ensure DNS/hosts entry exists
echo "YOUR_CLUSTER_IP  <MONTAGE_HOSTNAME>" | sudo tee -a /etc/hosts

open "http://<MONTAGE_HOSTNAME>"
```

---

## Running Render Jobs

### One-off Render

```bash
# Create job from template
kubectl create job montage-render-$(date +%s) \
  --from=job/montage-ai-render \
  -n "$CLUSTER_NAMESPACE"

# Watch progress
kubectl logs -n "$CLUSTER_NAMESPACE" -f job/montage-render-*
```

### With Custom Settings

```bash
# Override via ConfigMap patch
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: montage-ai-config
  namespace: "<CLUSTER_NAMESPACE>"
data:
  CUT_STYLE: "hitchcock"
  STABILIZE: "true"
  UPSCALE: "true"
  TARGET_DURATION: "60"
EOF

kubectl create job montage-custom-$(date +%s) \
  --from=job/montage-ai-render \
  -n "$CLUSTER_NAMESPACE"
```

### Scheduled (CronJob)

```bash
kubectl apply -f deploy/k3s/base/cronjob.yaml
kubectl get cronjobs -n "$CLUSTER_NAMESPACE"
```

---

## Editing Styles

| Style | Description |
|-------|-------------|
| `dynamic` | Fast cuts, high energy (default) |
| `hitchcock` | Suspenseful, methodical pacing |
| `mtv` | Music video, beat-synchronized |
| `action` | Quick cuts, action sequences |
| `documentary` | Slower, informative |
| `minimalist` | Clean, simple |
| `wes_anderson` | Symmetric, stylized |

---

## Operations

### View Logs

```bash
kubectl logs -n "$CLUSTER_NAMESPACE" -l app.kubernetes.io/name=montage-ai -f
```

### Restart Deployment

```bash
kubectl rollout restart deployment/montage-ai-web -n "$CLUSTER_NAMESPACE"
```

### Scale

```bash
# Stop
kubectl scale deployment montage-ai-web -n "$CLUSTER_NAMESPACE" --replicas=0

# Start
kubectl scale deployment montage-ai-web -n "$CLUSTER_NAMESPACE" --replicas=1
```

### Shell Access

```bash
kubectl exec -it -n "$CLUSTER_NAMESPACE" deployment/montage-ai-web -- /bin/bash
```

### Check Resource Usage

```bash
kubectl top pods -n "$CLUSTER_NAMESPACE"
kubectl top nodes
```

---

## Node Benchmarking

Use a lightweight benchmark job to label nodes with a performance multiplier
for task routing (label: `montage-ai/bench-score`).

```bash
python3 scripts/benchmarks/run_cluster_benchmark.py --namespace "$CLUSTER_NAMESPACE"
```

---

## Troubleshooting

### Pod Not Starting

```bash
# Check events
kubectl describe pod -n "$CLUSTER_NAMESPACE" -l app.kubernetes.io/name=montage-ai

# Check logs
kubectl logs -n "$CLUSTER_NAMESPACE" -l app.kubernetes.io/name=montage-ai --previous
```

### Image Pull Errors

```bash
# Verify image exists in registry
curl "http://<REGISTRY_URL>/v2/<IMAGE_NAME>/tags/list"

# Check pod image
kubectl get pod -n "$CLUSTER_NAMESPACE" -o jsonpath='{.items[0].spec.containers[0].image}'
```

### Storage Issues

```bash
# Check PVC status (all should be "Bound")
kubectl get pvc -n "$CLUSTER_NAMESPACE"

# Check disk usage in pod
kubectl exec -n "$CLUSTER_NAMESPACE" deployment/montage-ai-web -- df -h /data/
```

### Network Issues

```bash
# Test DNS from pod
kubectl exec -n "$CLUSTER_NAMESPACE" deployment/montage-ai-web -- nslookup google.com

# Check network policies
kubectl get networkpolicy -n "$CLUSTER_NAMESPACE"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────────────────────────┐   │
│  │  ConfigMap  │────▶│        montage-ai Pod           │   │
│  │  (settings) │     │  ┌─────────────────────────┐    │   │
│  └─────────────┘     │  │   montage-ai container  │    │   │
│                      │  │   - Flask Web UI        │    │   │
│  ┌─────────────┐     │  │   - FFmpeg              │    │   │
│  │ PVC: input  │────▶│  │   - Python ML           │    │   │
│  │ (footage)   │     │  └─────────────────────────┘    │   │
│  └─────────────┘     └──────────────┬──────────────────┘   │
│  ┌─────────────┐                    │                      │
│  │ PVC: music  │────────────────────┤                      │
│  └─────────────┘                    │                      │
│  ┌─────────────┐                    │                      │
│  │ PVC: output │◀───────────────────┘                      │
│  └─────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Related Documentation

- **[Configuration Reference](../CONFIGURATION.md)** — Environment variables
- **[Kubernetes Runbook](../../docs/KUBERNETES_RUNBOOK.md)** — Operations (public stub; internal runbook on request)
- **[Getting Started](../../docs/getting-started.md)** — Local development setup
