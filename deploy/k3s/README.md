# Kubernetes Deployment for Montage-AI

Deploy montage-ai as a Kubernetes Job for batch video processing.

## Directory Structure

```text
deploy/k3s/
├── base/                    # Base manifests
│   ├── namespace.yaml       # montage-ai namespace
│   ├── configmap.yaml       # Environment configuration
│   ├── pvc.yaml             # Storage claims (local-path)
│   ├── nfs-pv.yaml          # NFS PersistentVolumes (for distributed)
│   ├── job.yaml             # One-off render job
│   ├── cronjob.yaml         # Scheduled renders
│   └── kustomization.yaml   # Base kustomization
├── overlays/
│   ├── dev/                 # Fast preview settings
│   │   └── kustomization.yaml
│   ├── production/          # HQ render + AMD GPU targeting
│   │   └── kustomization.yaml
│   ├── amd/                 # AMD GPU (VAAPI) acceleration
│   │   ├── kustomization.yaml
│   │   └── patch-job-amd.yaml
│   ├── jetson/              # NVIDIA Jetson (NVENC) acceleration
│   │   └── kustomization.yaml
│   ├── distributed/         # Multi-node GPU with NFS storage
│   │   ├── kustomization.yaml
│   │   ├── nfs-pvc.yaml     # NFS PersistentVolumeClaims
│   │   └── patch-job-distributed.yaml
│   └── gpu/                 # Generic NVIDIA GPU acceleration
│       ├── kustomization.yaml
│       ├── patch-job-gpu.yaml
│       └── patch-web-gpu.yaml
└── README.md
```

## Prerequisites

- Kubernetes cluster (K3s, K8s, EKS, GKE, etc.)
- `kubectl` configured with cluster access
- Container image available (see Build section)

## Build the Image

Build and push the container image before deploying:

### Option A: Local Build + Push to Registry

```bash
# Build for amd64 architecture
docker buildx build --platform linux/amd64 \
  -t ghcr.io/mfahsold/montage-ai:latest .

# Push to GitHub Container Registry
docker push ghcr.io/mfahsold/montage-ai:latest

# Or push to local/private registry
docker tag ghcr.io/mfahsold/montage-ai:latest your-registry/montage-ai:latest
docker push your-registry/montage-ai:latest
```

### Option B: In-Cluster Build (Kaniko)

For clusters with Kaniko or similar build systems, create a build job that:

1. Clones this repository
2. Builds the image using `Dockerfile`
3. Pushes to your registry

See your cluster's build documentation (e.g., fluxibri_core for Fluxibri clusters).

### Option C: Manual Import

```bash
# Build locally
docker build -t ghcr.io/mfahsold/montage-ai:latest .

# Export and import to cluster node
docker save ghcr.io/mfahsold/montage-ai:latest | \
  ssh user@cluster-node "sudo ctr -n k8s.io images import -"
```

## Deploy to Cluster

### Step 1: Create Namespace and Resources

```bash
# Deploy base resources (namespace, configmap, PVCs)
kubectl apply -k deploy/k3s/base/
```

### Step 2: Load Media Data

```bash
# Option A: Use a data loader pod
kubectl run -it --rm data-loader \
  --image=busybox \
  -n montage-ai \
  -- sh
# Then mount PVCs and copy files inside the pod

# Option B: Copy via kubectl (requires running pod)
kubectl cp ./my-videos/ montage-ai/<pod-name>:/data/input/
kubectl cp ./my-music.mp3 montage-ai/<pod-name>:/data/music/

# Option C: Use existing shared storage
# Configure PVCs to use your NFS/SMB storage class
```

### Step 3: Run Render Job

```bash
# Start render job
kubectl apply -f deploy/k3s/base/job.yaml

# Watch progress
kubectl logs -n montage-ai -f job/montage-ai-render

# Check status
kubectl get jobs -n montage-ai
```

### Step 4: Retrieve Output

```bash
# Copy rendered video from output PVC
kubectl cp montage-ai/<pod-name>:/data/output/ ./rendered/
```

## Deployment Variants

```bash
# Base (any amd64 node)
kubectl apply -k deploy/k3s/base/

# Development (fast preview, low resources)
kubectl apply -k deploy/k3s/overlays/dev/

# Production (AMD GPU node, high quality)
kubectl apply -k deploy/k3s/overlays/production/

# NVIDIA GPU node (generic - device plugin + runtimeclass "nvidia" required)
kubectl apply -k deploy/k3s/overlays/gpu/

# AMD GPU with VAAPI (codeai-fluxibriserver)
kubectl apply -k deploy/k3s/overlays/amd/

# NVIDIA Jetson (codeaijetson-desktop)
kubectl apply -k deploy/k3s/overlays/jetson/
```

### GPU Overlay Comparison

| Overlay       | Target Node                | GPU Type     | Encoder | Use Case                    |
| ------------- | -------------------------- | ------------ | ------- | --------------------------- |
| `gpu`         | Any NVIDIA GPU node        | NVIDIA CUDA  | NVENC   | Generic NVIDIA acceleration |
| `amd`         | codeai-fluxibriserver      | AMD Radeon   | VAAPI   | AMD GPU encoding            |
| `jetson`      | codeaijetson-desktop       | NVIDIA Tegra | NVMPI   | Edge device rendering       |
| `distributed` | Any GPU node (NFS storage) | Auto-detect  | Auto    | Multi-node GPU scheduling   |

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────────────────────────┐   │
│  │  ConfigMap  │────▶│        montage-ai Job           │   │
│  │  (settings) │     │  ┌─────────────────────────┐    │   │
│  └─────────────┘     │  │   montage-ai container  │    │   │
│                      │  │   - FFmpeg              │    │   │
│  ┌─────────────┐     │  │   - librosa             │    │   │
│  │ PVC: input  │────▶│  │   - Real-ESRGAN         │    │   │
│  │ (footage)   │     │  └─────────────────────────┘    │   │
│  └─────────────┘     └──────────────┬──────────────────┘   │
│  ┌─────────────┐                    │                      │
│  │ PVC: music  │────────────────────┤                      │
│  └─────────────┘                    │                      │
│  ┌─────────────┐                    │                      │
│  │ PVC: assets │────────────────────┤                      │
│  └─────────────┘                    ▼                      │
│  ┌─────────────┐     ┌─────────────────────────────────┐   │
│  │ PVC: output │◀────│      Generated Video            │   │
│  └─────────────┘     └─────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Optional: Ollama/KubeAI for Creative Director LLM  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Edit ConfigMap

Modify `configmap.yaml` or use Kustomize overlays:

```yaml
# deploy/k3s/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../
patches:
  - patch: |-
      apiVersion: v1
      kind: ConfigMap
      metadata:
        name: montage-ai-config
      data:
        CUT_STYLE: "hitchcock"
        STABILIZE: "true"
        UPSCALE: "true"
```

### Available Styles

| Style          | Description                          |
| -------------- | ------------------------------------ |
| `dynamic`      | Fast cuts, high energy (default)     |
| `hitchcock`    | Suspenseful, methodical pacing       |
| `mtv`          | Music video style, beat-synchronized |
| `action`       | Quick cuts, action sequences         |
| `documentary`  | Slower, informative style            |
| `minimalist`   | Clean, simple edits                  |
| `wes_anderson` | Symmetric, stylized compositions     |

### Storage Classes

Uncomment and modify `storageClassName` in `pvc.yaml` for your cluster:

```yaml
# Examples:
storageClassName: local-path      # K3s default
storageClassName: standard        # GKE default
storageClassName: gp2             # AWS EBS
storageClassName: longhorn        # Longhorn
storageClassName: nfs-client      # NFS provisioner
```

## GPU Support

### Available GPU Resources

The cluster has the following GPU resources available:

| Node                     | GPU Type            | Resource Key        | Encoder |
| ------------------------ | ------------------- | ------------------- | ------- |
| codeaijetson-desktop     | NVIDIA Jetson Tegra | `nvidia.com/gpu: 1` | NVENC   |
| codeai-fluxibriserver    | AMD Radeon          | `amd.com/gpu: 1`    | VAAPI   |

### Using GPU Overlays

```bash
# Check available GPU resources
kubectl get nodes -o custom-columns='NAME:.metadata.name,NVIDIA:.status.allocatable.nvidia\.com/gpu,AMD:.status.allocatable.amd\.com/gpu'

# Deploy to AMD server with VAAPI
kubectl apply -k deploy/k3s/overlays/amd/

# Deploy to Jetson with NVENC
kubectl apply -k deploy/k3s/overlays/jetson/

# Distributed rendering (multi-node GPU)
kubectl apply -k deploy/k3s/overlays/distributed/

# Distributed parallel rendering (indexed Job shards)
kubectl apply -k deploy/k3s/overlays/distributed-parallel/
```

## Distributed Rendering (Multi-Node GPU)

The `distributed` overlay enables jobs to run on **any GPU node** in the cluster using NFS shared storage.

### Parallel Sharding (Indexed Jobs)

Use the `distributed-parallel` overlay to run multiple shards in parallel across the cluster.
Each shard processes a subset of variants using `CLUSTER_SHARD_INDEX` and `CLUSTER_SHARD_COUNT`.

```bash
# Example: 2-way sharding
kubectl apply -k deploy/k3s/overlays/distributed-parallel/
```

Set `NUM_VARIANTS` in the ConfigMap (or job env) to the total number of variants you want
and adjust `completions`, `parallelism`, and `CLUSTER_SHARD_COUNT` together.

### Setup NFS Storage

1. **Create NFS exports on your NFS server** (e.g., fluxibriserver):

```bash
# On NFS server
sudo mkdir -p /mnt/nfs-montage/{input,music,output,assets}
sudo chown -R nobody:nogroup /mnt/nfs-montage

# Add to /etc/exports
echo "/mnt/nfs-montage *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a
```

2. **Update NFS PV configuration** in `deploy/k3s/base/nfs-pv.yaml`:

```yaml
nfs:
  server: 192.168.1.16  # Your NFS server IP
  path: /mnt/nfs-montage/input
```

3. **Deploy the distributed overlay**:

```bash
kubectl apply -k deploy/k3s/overlays/distributed/
```

### How It Works

- **NFS PVCs** replace local-path PVCs for multi-node access (ReadWriteMany)
- **Node affinity** prefers GPU nodes (nvidia.com/gpu or amd.com/gpu)
- **Auto GPU detection**: `FFMPEG_HWACCEL=auto` selects NVENC or VAAPI based on node
- Jobs can land on Jetson (NVENC), AMD server (VAAPI), or any GPU node

### Cluster Topology

```text
┌─────────────────────────────────────────────────────────────┐
│                 NFS Server (fluxibriserver)                 │
│                   /mnt/nfs-montage/                         │
│     input/     music/     output/     assets/               │
└──────────────────────┬──────────────────────────────────────┘
                       │ NFS Mount
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Jetson    │ │  AMD GPU    │ │  CPU Node   │
│   (NVENC)   │ │  (VAAPI)    │ │  (fallback) │
└─────────────┘ └─────────────┘ └─────────────┘
```

### Prerequisites

- **NVIDIA GPU**: Requires nvidia-device-plugin daemonset and `nvidia` RuntimeClass
- **AMD GPU**: Requires amd-gpu-device-plugin daemonset and `/dev/dri` access

## Triggering Jobs

### One-off Render

```bash
# Create job with unique name
kubectl create job montage-ai-$(date +%s) \
  --from=job/montage-ai-render \
  -n montage-ai
```

### With Custom Settings

```bash
# Override environment variables
kubectl create job montage-ai-custom \
  --from=job/montage-ai-render \
  -n montage-ai \
  -- env CUT_STYLE=hitchcock STABILIZE=true
```

### CronJob (Scheduled)

See `cronjob.yaml` for scheduled batch processing.

## Integration with Fluxibri Cluster

If deploying to a Fluxibri cluster, use the existing Ollama service:

```yaml
# In configmap.yaml
OLLAMA_HOST: "http://ollama.kubeai-system.svc.cluster.local:11434"
```

## Troubleshooting

### Check Job Status

```bash
kubectl get jobs -n montage-ai
kubectl describe job montage-ai-render -n montage-ai
```

### View Logs

```bash
kubectl logs -n montage-ai -l app.kubernetes.io/name=montage-ai --tail=100
```

### Debug Pod

```bash
kubectl run -it --rm debug \
  --image=ghcr.io/mfahsold/montage-ai:latest \
  --namespace=montage-ai \
  -- /bin/bash
```

### Storage Issues

```bash
# Check PVC status
kubectl get pvc -n montage-ai

# Check if PVs are bound
kubectl get pv | grep montage-ai
```
