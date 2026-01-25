# Deployment

> **Single entry point for all deployment documentation.**

## Quick Start

```bash
# Local development
make dev && make dev-test

# Kubernetes cluster
make -C deploy/k3s config
make -C deploy/k3s deploy-production
```

## Documentation Map

| Document | Purpose |
|----------|---------|
| **[k3s/README.md](k3s/README.md)** | Kubernetes deployment guide (K3s, K8s, EKS, GKE) |
| **[CONFIGURATION.md](CONFIGURATION.md)** | Environment variables & config reference |
| **[../docs/KUBERNETES_RUNBOOK.md](../docs/KUBERNETES_RUNBOOK.md)** | Operations runbook (public stub; internal on request) |

## Directory Structure

```
deploy/
├── README.md              ← You are here
├── CONFIGURATION.md       ← Config reference
└── k3s/
    ├── config-global.yaml ← Global config (K8s ConfigMap source)
    ├── base/cluster-config.env ← Generated from config-global.yaml
    ├── README.md          ← Main K8s deployment guide
    ├── base/              ← Base manifests
    ├── overlays/          ← Environment-specific configs
    ├── app/               ← Full app deployment
    ├── deploy.sh          ← Deploy script
    ├── undeploy.sh        ← Cleanup script
    └── build-and-push.sh  ← Build & push image
```

## Deployment Options

### 1. Local Development (No K8s)

```bash
# One-time setup
make dev

# Run with volume mounts (instant code changes)
make dev-test

# Or use Docker Compose
docker-compose up
```

### 2. Single-Node K3s

```bash
# Render config + deploy
make -C deploy/k3s config
make -C deploy/k3s deploy-production

# Or step by step:
./deploy/k3s/build-and-push.sh
./deploy/k3s/deploy.sh
```

### 3. Multi-Node Cluster (Distributed)

```bash
# Setup NFS storage first (see k3s/README.md)
kubectl apply -k deploy/k3s/overlays/distributed/
```

### 4. GPU-Accelerated

```bash
# AMD GPU (VAAPI)
kubectl apply -k deploy/k3s/overlays/amd/

# NVIDIA Jetson
kubectl apply -k deploy/k3s/overlays/jetson/
```

## K3s Make Targets

```bash
make -C deploy/k3s help
make -C deploy/k3s deploy-dev
make -C deploy/k3s deploy-staging
make -C deploy/k3s deploy-production
```

## Environment Configuration

Copy and customize:

```bash
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
# Edit with your values, then render:
make -C deploy/k3s config
```

Key variables:

```yaml
registry:
  url: "registry.registry.svc.cluster.local:5000"
cluster:
  namespace: "montage-ai"
```

See **[CONFIGURATION.md](CONFIGURATION.md)** for full reference.
