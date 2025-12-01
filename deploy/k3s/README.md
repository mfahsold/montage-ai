# Kubernetes Deployment for Montage-AI

Deploy montage-ai as a Kubernetes Job for batch video processing.

## Quick Start

```bash
# 1. Create namespace and resources
kubectl apply -k deploy/k3s/

# 2. Copy your footage to the input PVC
kubectl cp ./my-footage/ montage-ai/montage-ai-input:/data/input/

# 3. Copy music files
kubectl cp ./my-music/ montage-ai/montage-ai-music:/data/music/

# 4. Trigger a render job
kubectl create -f deploy/k3s/job.yaml

# 5. Watch progress
kubectl logs -n montage-ai -f job/montage-ai-render

# 6. Copy output
kubectl cp montage-ai/montage-ai-output:/data/output/ ./rendered/
```

## Architecture

```
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
