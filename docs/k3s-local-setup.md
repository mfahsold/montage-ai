# K3s Local Development Setup

Run a local Kubernetes cluster for developing and testing Montage AI's cluster deployment without a remote server.

---

## Choose a Tool

| Tool | Best for | Install |
|------|----------|---------|
| **k3d** | K3s in Docker (recommended) | `curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh \| bash` |
| **minikube** | General-purpose local K8s | `curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && sudo install minikube-linux-amd64 /usr/local/bin/minikube` |
| **kind** | CI/testing (K8s in Docker) | `go install sigs.k8s.io/kind@latest` |

> **Recommendation:** Use **k3d** if you already have Docker. It creates real K3s clusters inside containers with minimal overhead.

---

## k3d Quick Start

### 1. Create Cluster

```bash
# Single-node cluster with port mapping for Web UI
k3d cluster create montage-dev \
  --port "8080:80@loadbalancer" \
  --agents 0

# Verify
kubectl cluster-info
kubectl get nodes
```

### 2. Load Image (No Registry Needed)

```bash
# Build locally and import into k3d
docker build -t montage-ai:latest .
k3d image import montage-ai:latest -c montage-dev
```

> **Tip:** This avoids needing a container registry entirely. Re-run after each image rebuild.

### 3. Configure and Deploy

```bash
# Copy config
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml

# Set minimal values for local dev
sed -i 's/<CLUSTER_NAMESPACE>/montage-ai/g' deploy/k3s/config-global.yaml
sed -i 's/<CLUSTER_DOMAIN>/cluster.local/g' deploy/k3s/config-global.yaml
sed -i 's/<CONTROL_PLANE_IP>/127.0.0.1/g' deploy/k3s/config-global.yaml
sed -i 's/<NFS_SERVER_IP>/127.0.0.1/g' deploy/k3s/config-global.yaml

# Generate config and deploy
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster
```

### 4. Access Web UI

```bash
kubectl port-forward -n montage-ai svc/montage-ai-web 8080:80
# Open http://localhost:8080
```

### 5. Cleanup

```bash
k3d cluster delete montage-dev
```

---

## minikube Quick Start

```bash
# Create cluster
minikube start --memory 8192 --cpus 4

# Load image
minikube image load montage-ai:latest

# Deploy (same config steps as k3d)
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
# ... replace placeholders ...
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster

# Access
kubectl port-forward -n montage-ai svc/montage-ai-web 8080:80
```

---

## kind Quick Start

```bash
# Create cluster
kind create cluster --name montage-dev

# Load image
kind load docker-image montage-ai:latest --name montage-dev

# Deploy (same config steps)
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster

# Access
kubectl port-forward -n montage-ai svc/montage-ai-web 8080:80
```

---

## Storage Setup (Single-Node)

All three tools include a default `local-path` StorageClass that works for single-node development:

```bash
kubectl get storageclass
# NAME                   PROVISIONER             RECLAIMPOLICY
# local-path (default)   rancher.io/local-path   Delete
```

Set in `config-global.yaml`:

```yaml
storage:
  classes:
    default: "local-path"
```

> **Note:** `local-path` only supports ReadWriteOnce (RWO). This is fine for single-node dev clusters. For multi-node testing, see [Cluster Deployment: Storage Setup](cluster-deploy.md#storage-setup).

---

## Registry Alternatives

### No Registry (Simplest)

Load images directly into the cluster (shown above). Best for quick iteration.

### Local Registry with k3d

```bash
# Create cluster with a local registry
k3d cluster create montage-dev \
  --registry-create montage-registry:5111 \
  --port "8080:80@loadbalancer"

# Tag and push
docker tag montage-ai:latest localhost:5111/montage-ai:latest
docker push localhost:5111/montage-ai:latest

# Use in manifests: image: montage-registry:5111/montage-ai:latest
```

### External Registry

If you have a registry (Docker Hub, GitHub Container Registry, etc.), push there and reference it in `config-global.yaml`:

```yaml
registry:
  url: "ghcr.io/your-org"
  image: "montage-ai"
  tag: "latest"
```

---

## Dev Workflow Tips

**Fast iteration cycle:**

```bash
# 1. Edit code
# 2. Rebuild image
docker build -t montage-ai:latest .

# 3. Import into k3d
k3d image import montage-ai:latest -c montage-dev

# 4. Restart deployment to pick up new image
kubectl rollout restart deployment/montage-ai-web -n montage-ai

# 5. Watch rollout
kubectl rollout status deployment/montage-ai-web -n montage-ai
```

**View logs:**

```bash
kubectl logs -f -n montage-ai -l app.kubernetes.io/component=web-ui
```

**Debug a pod:**

```bash
kubectl exec -it -n montage-ai deploy/montage-ai-web -- /bin/bash
```

---

## Troubleshooting

**Pods stuck in `Pending`:**
Check StorageClass exists: `kubectl get storageclass`

**Image not found:**
Re-import after rebuild: `k3d image import montage-ai:latest -c montage-dev`

**Port already in use:**
Change the port mapping: `k3d cluster create montage-dev --port "9090:80@loadbalancer"`

**Insufficient resources:**
Increase Docker Desktop memory allocation, or use `minikube start --memory 12288`.

---

## Related Documentation

- [Cluster Deployment Guide](cluster-deploy.md) -- full K8s deployment steps
- [Configuration Reference](configuration.md) -- all environment variables
- [Troubleshooting](troubleshooting.md) -- common issues
- [Performance Tuning](performance-tuning.md) -- optimization settings
