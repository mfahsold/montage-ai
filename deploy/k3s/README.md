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

If you operate the Fluxibri cluster stack, deploy core infrastructure via
`mfahsold/fluxibri_core` first (canonical `make deploy-core`). This repo assumes
those services are already present and focuses on the Montage-AI app layer.

```bash
# Copy and edit canonical config
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
$EDITOR deploy/k3s/config-global.yaml

# Render cluster-config.env
make -C deploy/k3s config

# Build & push
cd deploy/k3s
./build-and-push.sh

# Deploy cluster overlay (canonical)
./deploy.sh cluster
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
├── base/                    # Canonical cluster manifests (cluster-agnostic)
│   ├── namespace.yaml       # montage-ai namespace
│   ├── cluster-config.env   # Rendered from config-global.yaml
│   ├── deployment.yaml      # Web UI deployment
│   ├── worker.yaml          # Queue workers (default + heavy)
│   ├── pvc.yaml             # Storage claims (RWX-friendly)
│   └── kustomization.yaml   # Base kustomization
└── overlays/
    ├── cluster/             # Canonical cluster overlay
    └── legacy/              # Archived overlays (internal/legacy)
```

---

## Deployment Commands

### Option 1: Makefile (Recommended)

```bash
# Render cluster config
make -C deploy/k3s config

# Deploy canonical cluster overlay
make -C deploy/k3s deploy-cluster

# If you are on a small or heterogeneous developer cluster, use the dev profile
# which applies reduced requests and a toleration to help scheduling:
make -C deploy/k3s deploy-dev
```

### Option 2: Shell Scripts

```bash
cd deploy/k3s

# Build & push to cluster registry
./build-and-push.sh

# Deploy (canonical cluster overlay)
./deploy.sh cluster

# Undeploy
./undeploy.sh
```

### Option 3: kubectl/kustomize Direct

```bash
# Render cluster-config.env first
make -C deploy/k3s config

# Canonical cluster deployment
kubectl apply -k deploy/k3s/overlays/cluster/
```

---

## Canonical Deployment

Montage‑AI supports exactly two deployment modes:

- **Local**: run via `./montage-ai.sh web` or Docker Compose.
- **Cluster**: deploy `deploy/k3s/overlays/cluster/` (Kustomize).

Legacy overlays are preserved in `deploy/k3s/overlays/legacy/` for reference only.

## GPU Acceleration

### Available GPU Resources

| Node (example) | GPU Type | Resource Key | Encoder |
|------|----------|--------------|---------|
| gpu-node-nvidia | NVIDIA Jetson | `nvidia.com/gpu: 1` | NVENC |
| gpu-node-amd | AMD Radeon | `amd.com/gpu: 1` | VAAPI |

### GPU Enablement (Cluster-Agnostic)

```bash
# Check available GPUs
kubectl get nodes -o custom-columns='NAME:.metadata.name,NVIDIA:.status.allocatable.nvidia\.com/gpu,AMD:.status.allocatable.amd\.com/gpu'

# Use your cluster's device plugins (NVIDIA/AMD/Jetson) and taints/tolerations.
# The canonical cluster overlay avoids hardcoded node names.
#
# Legacy GPU overlays are archived in:
#   deploy/k3s/overlays/legacy/
```

---

## Storage Configuration

### Local Storage (Single Node)

Default is controlled by `deploy/k3s/config-global.yaml` via `STORAGE_CLASS_DEFAULT`.
For multi-node clusters, prefer an RWX-capable storage class (e.g., NFS/CSI).

```yaml
# In config-global.yaml
storage:
  classes:
    default: "<RWX_STORAGE_CLASS>"
```

### NFS Storage (Multi-Node)

Provision an RWX storage class via your cluster (NFS/CSI), then set it in
`deploy/k3s/config-global.yaml`.

---

## Node Labels (Optional)

Apply these labels so overlays can schedule without hardcoded hostnames:

```bash
# Storage (hostPath PVs)
kubectl label node <node> fluxibri.ai/storage-primary=true

# Registry hosting + build access
kubectl label node <node> fluxibri.ai/registry-host=true
kubectl label node <node> fluxibri.ai/registry-access=true

# Optional GPU hints (if your cluster uses these labels)
kubectl label node <node> fluxibri.ai/gpu-type=amd-rocm
kubectl label node <node> fluxibri.ai/gpu-type=nvidia-tegra
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

## Running Render Jobs (Canonical)

Jobs are submitted via the API (same path used by the Web UI).

```bash
# Submit a job via API
python3 -m montage_ai.cli jobs --api-base http://<montage-service> submit \
  --style cinema_trailer \
  --option options.target_duration=30 \
  --option options.llm_clip_selection=true

# Inspect status
python3 -m montage_ai.cli jobs --api-base http://<montage-service> status <JOB_ID>
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
