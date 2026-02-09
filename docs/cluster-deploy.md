# Cluster Deployment

Step-by-step guide for deploying Montage AI to a Kubernetes (K3s/K8s) cluster.

---

## Prerequisites

- [ ] Kubernetes cluster running (`kubectl cluster-info` succeeds)
- [ ] `kustomize` installed (`kustomize version`)
- [ ] `make` available
- [ ] Container registry accessible from all nodes
- [ ] Storage class with RWX support (for shared media volumes)

---

## Quick Start

```bash
# 1. Configure
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
# Edit: replace ALL <...> placeholders with your values

# 2. Render manifests
make -C deploy/k3s config

# 3. Validate
make -C deploy/k3s validate

# 4. Deploy
make -C deploy/k3s deploy-cluster

# 5. Access Web UI
kubectl port-forward -n montage-ai svc/montage-ai-web 8080:80
# Open http://localhost:8080
```

---

## Detailed Steps

### 1. Create Configuration

Copy the example config and fill in your cluster details:

```bash
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
```

Edit `deploy/k3s/config-global.yaml` and replace all angle-bracket placeholders:

| Placeholder | Example | Description |
|-------------|---------|-------------|
| `<CLUSTER_NAMESPACE>` | `montage-ai` | Kubernetes namespace |
| `<CLUSTER_DOMAIN>` | `cluster.local` | Cluster DNS domain |
| `<CONTROL_PLANE_IP>` | `192.168.1.10` | Control plane node IP |
| `<NFS_SERVER_IP>` | `192.168.1.10` | NFS server for shared storage |

### 2. Render and Validate

```bash
# Generate cluster-config.env from config-global.yaml
make -C deploy/k3s config

# Validate kustomize manifests build correctly
make -C deploy/k3s validate
```

If validation fails, check that all placeholders are replaced:

```bash
grep '<' deploy/k3s/config-global.yaml
# Should return nothing
```

### 3. Build and Push Image

```bash
cd deploy/k3s

# Multi-arch build (recommended for mixed ARM/AMD clusters)
BUILD_MULTIARCH=true ./build-and-push.sh

# Single-arch build (faster, for homogeneous clusters)
./build-and-push.sh
```

### 4. Deploy

```bash
make -C deploy/k3s deploy-cluster
```

This will:
- Create the namespace (if it doesn't exist)
- Apply all Kubernetes manifests (Deployments, Services, ConfigMaps, PVCs)
- Wait for rollout to complete

### 5. Verify

```bash
# Check pod status
make -C deploy/k3s status

# Or manually
kubectl get all -n montage-ai

# Check logs
kubectl logs -n montage-ai -l app.kubernetes.io/component=web-ui
```

### 6. Access

```bash
# Port-forward to access locally
kubectl port-forward -n montage-ai svc/montage-ai-web 8080:80

# Or use ingress if configured
open "https://<your-hostname>"
```

---

## Re-deploying (Safety)

The deployment is **idempotent** and safe to re-run:

| Resource | On Re-deploy |
|----------|--------------|
| PVCs (data volumes) | Preserved (immutable fields protected) |
| ConfigMaps | Updated with latest config |
| Deployments | Rolling update (zero downtime) |
| Services | Preserved |

```bash
# Safe to run multiple times:
make -C deploy/k3s deploy-cluster
# Existing data in PVCs will NOT be deleted
```

---

## Troubleshooting

**Pods stuck in `CreateContainerConfigError`:**
ConfigMap name mismatch. Run `make -C deploy/k3s config` to regenerate.

**Namespace not found:**
The deploy script creates the namespace automatically. If it fails, create manually:
```bash
kubectl create namespace montage-ai
```

**Image pull errors:**
Verify the registry is accessible and the image exists:
```bash
docker pull <your-registry>/montage-ai:<tag>
```

**Unresolved placeholders:**
Check config-global.yaml for remaining `<...>` values:
```bash
grep '<' deploy/k3s/config-global.yaml
```

See [Troubleshooting: Kubernetes](troubleshooting.md#kubernetes-deploy-errors) for more.

---

## Advanced

For advanced topics (multi-arch builds, custom overlays, KEDA autoscaling, distributed rendering), see [deploy/k3s/README.md](../deploy/k3s/README.md).

For operational runbooks (scaling, SLOs, incident response), see [operations/](operations/README.md).
