# Cluster Deployment

Step-by-step guide for deploying Montage AI to a Kubernetes (K3s/K8s) cluster.

---

## Prerequisites

- [ ] Kubernetes cluster running (`kubectl cluster-info` succeeds)
- [ ] `kustomize` installed (`kustomize version`)
- [ ] `make` available
- [ ] Container registry accessible from all nodes
- [ ] Storage class with RWX support (for shared media volumes) — see [Storage Setup](#storage-setup) below

---

## Quick Start

> **Storage required:** Pods will stay in `Pending` or `Init:0/2` if no storage class is available. Verify with `kubectl get storageclass` before deploying. See [Storage Setup](#storage-setup) below.

```bash
# 1. Configure
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
# Edit: replace ALL <...> placeholders with your values

# 2. Render manifests
make -C deploy/k3s config

# 3. Pre-flight check (tools, placeholders, cluster connectivity)
make -C deploy/k3s pre-flight

# 4. Validate
make -C deploy/k3s validate

# 5. Deploy
make -C deploy/k3s deploy-cluster

# 6. Access Web UI
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

| Resource | On Re-deploy | Immutable Fields | Notes |
|----------|-------------|------------------|-------|
| **PVCs** | Preserved | `storageClassName`, `accessModes`, `storage` size | Data never deleted; mismatches cause errors |
| **ConfigMaps** | Updated | — | New values applied, but **pods do NOT auto-restart** |
| **Deployments** | Rolling update | — | Zero-downtime; pods restart if image or env changes |
| **Services** | Preserved | `clusterIP` | Stable endpoints |
| **Namespace** | Preserved | `name` | Labels updated |

**Important details:**
- **ConfigMap changes** require a pod restart to take effect: `kubectl rollout restart deployment/montage-ai-web -n montage-ai`
- **Image updates** only trigger a rollout if the image tag changes (use unique tags, not just `latest`)
- **PVC spec is immutable** after creation — if storage class or size changes in config, delete old PVCs first (data loss!)

**Preview changes before applying:**

```bash
# Show what would change without applying
make -C deploy/k3s diff
```

**Safe to run multiple times:**

```bash
make -C deploy/k3s deploy-cluster
# Existing data in PVCs will NOT be deleted
```

See [Rollback Guide](operations/rollback.md) for recovery procedures.

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

**Need to rollback?** See [Rollback Guide](operations/rollback.md).

See [Troubleshooting: Kubernetes](troubleshooting.md#kubernetes-deploy-errors) for more.

---

## Storage Setup

Montage AI requires shared storage (ReadWriteMany PVCs) for media files across pods.

### Storage Requirements by Cluster Type

| Cluster Type | Access Mode | Recommended Provider | Notes |
|-------------|-------------|---------------------|-------|
| **Single-node** (testing) | RWO | `local-path` (K3s default) | Works out of the box |
| **Multi-node** (production) | **RWX** | NFS, Longhorn, Rook/Ceph | Required for shared media across pods |
| **Cloud** (EKS/GKE) | RWX | EFS (AWS), Filestore (GCP) | Managed RWX providers |

**Why RWX?** Render workers and the web UI pod both need access to the same `/data/input`, `/data/output`, `/data/music`, and `/data/assets` volumes. With RWO (ReadWriteOnce), only one pod on one node can mount the volume — multi-node scheduling will fail with `Multi-Attach error`.

**Check your cluster's StorageClass capabilities:**

```bash
# List available storage classes
kubectl get storageclass

# Check if a class supports RWX (look for ReadWriteMany in allowedTopologies or docs)
kubectl describe storageclass <name>
```

**PVC stuck in Pending?** Common causes:
1. No StorageClass exists → install a provisioner (see options below)
2. StorageClass doesn't support RWX → switch to NFS/Longhorn for multi-node
3. Insufficient disk → check node disk space with `df -h`

### Single-Node / Testing

K3s includes `local-path` provisioner by default (no setup needed):

```bash
kubectl get storageclass
# NAME         PROVISIONER             RECLAIMPOLICY
# local-path   rancher.io/local-path   Delete
```

Set in `config-global.yaml`:
```yaml
storage:
  classes:
    default: "local-path"
```

> **Note:** `local-path` does **not** support RWX (ReadWriteMany). For single-node testing this works, but multi-node clusters need NFS or another RWX provider.

### Multi-Node / Production

For multi-node clusters, you need an RWX-capable storage backend:

| Option | Best for | Setup |
|--------|----------|-------|
| **NFS provisioner** | On-prem, self-managed | Install `nfs-kernel-server` on a node, then deploy NFS CSI provisioner |
| **Longhorn** | K3s clusters | `kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/master/deploy/longhorn.yaml` |
| **Rook/Ceph** | Large clusters | See [Rook documentation](https://rook.io/docs/rook/latest/) |
| **AWS EFS** | EKS | Use EFS CSI driver |
| **GCP Filestore** | GKE | Use Filestore CSI driver |

**Quick NFS setup (Ubuntu/Debian):**

```bash
# On the NFS server node:
sudo apt install nfs-kernel-server
sudo mkdir -p /mnt/nfs-montage
echo "/mnt/nfs-montage *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -ra

# Then set in config-global.yaml:
# storage.nfs.server: "<NFS_SERVER_IP>"
# storage.nfs.path: "/mnt/nfs-montage"
```

**Using Fluxibri Core:** If you run the [Fluxibri Core](https://github.com/mfahsold/fluxibri_core) infrastructure stack, NFS provisioning is handled automatically via `make deploy-core`.

---

## Advanced

For advanced topics (multi-arch builds, custom overlays, KEDA autoscaling, distributed rendering), see [deploy/k3s/README.md](../deploy/k3s/README.md).

For operational runbooks (scaling, SLOs, incident response), see [operations/](operations/README.md).
