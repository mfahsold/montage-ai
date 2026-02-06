# Deployment

> **Single entry point for all deployment documentation.**

## Quick Start

```bash
# Local development (Docker)
docker compose up

# Kubernetes cluster (kustomize)
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster
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
    ├── overlays/          ← Canonical cluster overlay + archived legacy (reference only)
    ├── deploy.sh          ← Deploy script
    ├── undeploy.sh        ← Cleanup script
    └── build-and-push.sh  ← Build & push image
```

## Deployment Options

### 1. Local Development (Docker)

```bash
docker compose up
```

### 2. Cluster Mode (Kubernetes)

```bash
# Render config + deploy
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster

# Or step by step:
./deploy/k3s/build-and-push.sh
./deploy/k3s/deploy.sh
```

## K3s Make Targets

```bash
make -C deploy/k3s help
make -C deploy/k3s deploy-cluster
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
  url: "<REGISTRY_URL>"
cluster:
  namespace: "<CLUSTER_NAMESPACE>"
```

See **[CONFIGURATION.md](CONFIGURATION.md)** for full reference.
