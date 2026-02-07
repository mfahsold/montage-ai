# Kubernetes Deployment Guide

> **Single source of truth for K8s/K3s deployment.**

## Quick Start

```bash
# 1. Render cluster config
make -C deploy/k3s config

# 2. Build & push multi-arch image
cd deploy/k3s
BUILD_MULTIARCH=true ./build-and-push.sh

# 3. Access Web UI (choose one)
# If you have ingress configured:
open "https://<MONTAGE_HOSTNAME>"

# Or port-forward locally:
CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
LOCAL_PORT="${LOCAL_PORT:-5000}"
kubectl port-forward -n "$CLUSTER_NAMESPACE" svc/montage-ai-web "${LOCAL_PORT}:8080"
open "http://localhost:${LOCAL_PORT}"
```

Note: If you build a single-arch image (amd64 only), pin workloads to that
architecture via an overlay patch or add a node selector before deploying.
For heterogeneous clusters, prefer a multi-arch build and set
`cluster.allowMixedArch: true` in `deploy/k3s/config-global.yaml`.
`build-and-push.sh` now auto-selects an existing native multi-arch builder
(`montage-multiarch`, `simple-builder`, etc.) before falling back to local QEMU,
and enables registry layer cache by default (`CACHE_REF`).

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
│   ├── worker.yaml          # Queue workers (single adaptive tier)
│   ├── pvc.yaml             # Storage claims (RWX-friendly)
│   └── kustomization.yaml   # Base kustomization
└── overlays/
    ├── cluster/             # Canonical cluster overlay
    └── legacy/              # Archived overlays (reference only)
```

---

## Rendering Image Placeholders

Some manifests use `<IMAGE_FULL>` placeholders (jobs and patches). Render them
with the shared image resolver:

```bash
scripts/render-image.sh deploy/k3s/job-documentary-trailer.yaml > /tmp/job.yaml
kubectl apply -f /tmp/job.yaml
```

---

## Deployment Commands

### Option 1: Makefile (Recommended)

```bash
# Render cluster config
make -C deploy/k3s config

# Deploy canonical cluster overlay
make -C deploy/k3s deploy-cluster
```

### Option 2: Shell Scripts

```bash
cd deploy/k3s

# Build & push to cluster registry
./build-and-push.sh

# If your registry is HTTP (insecure), ensure buildx uses the local config:
# - Verify buildkitd.toml has [registry."HOST:PORT"] http=true/insecure=true
# - Recreate the builder once if needed:
#   FORCE_BUILDER_RECREATE=true ./build-and-push.sh

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

- **Local**: run via Docker Compose (`docker compose up`).
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

## CGPU (Cloud GPU Offload)

Montage‑AI can offload heavy GPU tasks (final encode, upscale, voice isolation) to CGPU/Colab
when enabled. In cluster mode, the worker pods must have CGPU credentials mounted.

### 1) Create the credentials secret

```bash
kubectl -n "${CLUSTER_NAMESPACE:-montage-ai}" create secret generic cgpu-credentials \
  --from-file=config.json=/path/to/cgpu/config.json \
  --from-file=session.json=/path/to/cgpu/session.json
```

You can automate this with:

```bash
scripts/ops/cgpu-refresh-session.sh
```

### 2) Enable CGPU in cluster config

Set these in `deploy/config.env` (or export in your shell before `make -C deploy/k3s config`):

```bash
CGPU_ENABLED=true
CGPU_GPU_ENABLED=true
CGPU_HOST=cgpu-server.montage-ai.svc.cluster.local
FINAL_ENCODE_BACKEND=router
```

Then re-render + deploy:

```bash
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster
```

If the secret is missing, CGPU checks will fail fast and the system will fall back to local CPU/GPU.

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
  pvc:
    input: "montage-ai-input-nfs"
    output: "montage-ai-output-nfs"
    music: "montage-ai-music-nfs"
    assets: "montage-ai-assets-nfs"
```

### NFS Storage (Multi-Node)

Provision an RWX storage class via your cluster (NFS/CSI), then set it in
`deploy/k3s/config-global.yaml`.

If you already have dedicated PVCs (RWX split), set the PVC names under
`storage.pvc` and they will be propagated to `cluster-config.env` for
distributed jobs.

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

## Distributed Jobs (Cluster Mode)

Cluster-mode jobs (scene detection + distributed render) use the in-cluster
Kubernetes API and require:

- `deploy/k3s/base/cluster-rbac.yaml` applied (includes `jobs`, `jobs/status`,
  and `configmaps` permissions).
- ServiceAccount `montage-ai-cluster` (already in `cluster-rbac.yaml`).
- If JobSet CRD is **not** installed, the system falls back to standard Jobs.

### Required Cluster Env (for in-cluster jobs)

These values are read by the job submitter. Prefer configuring them in
`deploy/k3s/config-global.yaml` (rendered into `cluster-config.env`).

- `REGISTRY_HOST` / `REGISTRY_PORT` (or `REGISTRY_URL`)
- `IMAGE_NAME` / `IMAGE_TAG`
- `PVC_INPUT_NAME`, `PVC_OUTPUT_NAME`, `PVC_MUSIC_NAME`, `PVC_ASSETS_NAME`
- `WORKER_QUEUE_LIST_NAME`, `WORKER_QUEUE_SCALE_THRESHOLD`
- `WORKER_MIN_REPLICAS`, `WORKER_MAX_REPLICAS`
- `CLUSTER_PARALLELISM`, `MAX_SCENE_WORKERS`, `MAX_PARALLEL_JOBS`, `FFMPEG_THREADS`

Example `cluster-config.env` excerpt:

```env
REGISTRY_URL=192.168.1.12:30500
IMAGE_NAME=montage-ai
IMAGE_TAG=latest
PVC_INPUT_NAME=montage-ai-input-nfs
PVC_OUTPUT_NAME=montage-ai-output-nfs
PVC_MUSIC_NAME=montage-ai-music-nfs
PVC_ASSETS_NAME=montage-ai-assets-nfs
WORKER_QUEUE_LIST_NAME=rq:queue:default:intermediate
WORKER_QUEUE_SCALE_THRESHOLD=4
WORKER_MIN_REPLICAS=3
WORKER_MAX_REPLICAS=24
CLUSTER_PARALLELISM=24
MAX_SCENE_WORKERS=24
MAX_PARALLEL_JOBS=24
FFMPEG_THREADS=16
```

### Performance Tuning (Recommended)

For large AV1/4K inputs, scene detection is CPU‑bound. Use a larger tier and
shared proxy cache to avoid repeated decodes:

```env
SCENE_DETECT_TIER=xlarge
SCENE_CACHE_DIR=/data/output/scene_cache
PROXY_CACHE_DIR=/data/output/proxy_cache
CLUSTER_PARALLELISM=24
MAX_SCENE_WORKERS=24
MAX_PARALLEL_JOBS=24
FFMPEG_THREADS=16
```

Notes:
- `CLUSTER_PARALLELISM` controls shard count for distributed jobs.
- KEDA scales workers based on `WORKER_QUEUE_LIST_NAME`; keep this aligned with the queue name used by the web/API process.
- If KEDA does not scale, verify `ScaledObject` health with `kubectl -n montage-ai describe scaledobject montage-ai-worker-scaler`.
- With a single video, time‑based sharding is automatic; for mixed inputs, consider
  pre‑splitting very large files to balance load.

### Example (ad‑hoc job)

```yaml
spec:
  template:
    spec:
      serviceAccountName: montage-ai-cluster
      containers:
        - name: montage-ai
          env:
            - name: REGISTRY_HOST
              value: 192.168.1.12
            - name: REGISTRY_PORT
              value: "30500"
            - name: PVC_INPUT_NAME
              value: montage-ai-input-nfs
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
