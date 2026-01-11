# Deployment

> **Single entry point for all deployment documentation.**

## Quick Start

```bash
# Local development
make dev && make dev-test

# Kubernetes cluster
make cluster
```

## Documentation Map

| Document | Purpose |
|----------|---------|
| **[k3s/README.md](k3s/README.md)** | Kubernetes deployment guide (K3s, K8s, EKS, GKE) |
| **[CONFIGURATION.md](CONFIGURATION.md)** | Environment variables & config reference |
| **[../docs/KUBERNETES_RUNBOOK.md](../docs/KUBERNETES_RUNBOOK.md)** | Operations runbook & failure recovery |

## Directory Structure

```
deploy/
├── README.md              ← You are here
├── CONFIGURATION.md       ← Config reference
├── config.env             ← Environment template
├── config-global.yaml     ← Global config (K8s ConfigMap source)
└── k3s/
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
# Build + push + deploy
make cluster

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

## Make Targets

```bash
make help          # Show all available commands

# Local development
make dev           # Build base image with cache
make dev-test      # Run with volume mounts
make dev-shell     # Interactive shell

# Cluster deployment
make cluster       # Build + push + deploy (all-in-one)

# Web UI
make web           # Start web UI locally
make web-deploy    # Deploy to K8s

# Maintenance
make test          # Run tests
make clean         # Clean caches
```

## Environment Configuration

Copy and customize:

```bash
cp deploy/config.env.example deploy/config.env
# Edit with your values
```

Key variables:

```bash
REGISTRY_HOST="YOUR_REGISTRY"     # Container registry
REGISTRY_PORT="5000"              # Registry port
CLUSTER_NAMESPACE="montage-ai"    # K8s namespace
```

See **[CONFIGURATION.md](CONFIGURATION.md)** for full reference.
