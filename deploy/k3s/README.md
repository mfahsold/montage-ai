# Kubernetes Deployment for Montage-AI

Deploy montage-ai as a Kubernetes Job for batch video processing.

## Directory Structure

```text
deploy/k3s/
├── base/                    # Base manifests
│   ├── namespace.yaml       # montage-ai namespace
│   ├── configmap.yaml       # Environment configuration
│   ├── pvc.yaml             # Storage claims
│   ├── job.yaml             # One-off render job
│   ├── cronjob.yaml         # Scheduled renders
│   └── kustomization.yaml   # Base kustomization
├── overlays/
│   ├── dev/                 # Fast preview settings
│   │   └── kustomization.yaml
│   └── production/          # HQ render + AMD GPU targeting
│       └── kustomization.yaml
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
```

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

For hardware-accelerated upscaling, uncomment GPU resources in `job.yaml`:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

Requires NVIDIA device plugin installed in cluster.

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
