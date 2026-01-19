# Kubernetes Deployment Guide

> **Single source of truth for K8s/K3s deployment.**

## Quick Start

```bash
# 1. Build & push multi-arch image
make cluster

# 2. Access Web UI (choose one)
# If you have ingress configured:
open http://YOUR_MONTAGE_HOST

# Or port-forward locally:
kubectl port-forward -n montage-ai svc/montage-ai-web 5000:8080
open http://localhost:5000
```

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
# Build multi-arch + push + deploy (all-in-one)
make cluster

# Just build locally
make dev

# Deploy specific overlay
kubectl apply -k deploy/k3s/overlays/production/
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

## Access Methods

### Port Forward (Development)

```bash
kubectl port-forward -n montage-ai svc/montage-ai-web 5000:8080
open http://localhost:5000
```

### Ingress (Production)

```bash
# Ensure DNS/hosts entry exists
echo "YOUR_CLUSTER_IP  montage-ai.local" | sudo tee -a /etc/hosts

open http://montage-ai.local
```

---

## Running Render Jobs

### One-off Render

```bash
# Create job from template
kubectl create job montage-render-$(date +%s) \
  --from=job/montage-ai-render \
  -n montage-ai

# Watch progress
kubectl logs -n montage-ai -f job/montage-render-*
```

### With Custom Settings

```bash
# Override via ConfigMap patch
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: montage-ai-config
  namespace: montage-ai
data:
  CUT_STYLE: "hitchcock"
  STABILIZE: "true"
  UPSCALE: "true"
  TARGET_DURATION: "60"
EOF

kubectl create job montage-custom-$(date +%s) \
  --from=job/montage-ai-render \
  -n montage-ai
```

### Scheduled (CronJob)

```bash
kubectl apply -f deploy/k3s/base/cronjob.yaml
kubectl get cronjobs -n montage-ai
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
kubectl logs -n montage-ai -l app.kubernetes.io/name=montage-ai -f
```

### Restart Deployment

```bash
kubectl rollout restart deployment/montage-ai-web -n montage-ai
```

### Scale

```bash
# Stop
kubectl scale deployment montage-ai-web -n montage-ai --replicas=0

# Start
kubectl scale deployment montage-ai-web -n montage-ai --replicas=1
```

### Shell Access

```bash
kubectl exec -it -n montage-ai deployment/montage-ai-web -- /bin/bash
```

### Check Resource Usage

```bash
kubectl top pods -n montage-ai
kubectl top nodes
```

---

## Node Benchmarking

Use a lightweight benchmark job to label nodes with a performance multiplier
for task routing (label: `montage-ai/bench-score`).

```bash
python3 scripts/benchmarks/run_cluster_benchmark.py --namespace montage-ai
```

---

## Troubleshooting

### Pod Not Starting

```bash
# Check events
kubectl describe pod -n montage-ai -l app.kubernetes.io/name=montage-ai

# Check logs
kubectl logs -n montage-ai -l app.kubernetes.io/name=montage-ai --previous
```

### Image Pull Errors

```bash
# Verify image exists in registry
curl http://YOUR_REGISTRY:5000/v2/montage-ai/tags/list

# Check pod image
kubectl get pod -n montage-ai -o jsonpath='{.items[0].spec.containers[0].image}'
```

### Storage Issues

```bash
# Check PVC status (all should be "Bound")
kubectl get pvc -n montage-ai

# Check disk usage in pod
kubectl exec -n montage-ai deployment/montage-ai-web -- df -h /data/
```

### Network Issues

```bash
# Test DNS from pod
kubectl exec -n montage-ai deployment/montage-ai-web -- nslookup google.com

# Check network policies
kubectl get networkpolicy -n montage-ai
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
