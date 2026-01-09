# Montage AI - K3s Deployment Guide

Deploy montage-ai video editor to K3s cluster with Exo LLM integration.

## Prerequisites

- K3s cluster running (control plane + workers)
- Local Docker registry at `YOUR_REGISTRY:5000`
- Exo deployed and running in `default` namespace
- kubectl configured for cluster access

## Quick Deploy

```bash
# 1. Build and push image
cd /home/codeai/montage-ai
./deploy/k3s/build-and-push.sh

# 2. Deploy to cluster
./deploy/k3s/deploy.sh
```

## Architecture

```
┌─────────────────┐
│   User Browser  │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  Traefik Ingress│ (montage.local)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Montage AI Pod │
│  - Web UI       │ :5000
│  - FFmpeg       │
│  - Python 3.10  │
└────────┬────────┘
         │ LLM API
         ▼
┌─────────────────┐
│   Exo ClusterIP │ http://exo-api.default.svc.cluster.local:8000
│   - Llama 3.2   │
│   - Distributed │
└─────────────────┘
```

## Configuration

### LLM Backend

Montage AI uses Exo distributed inference by default:

```yaml
OPENAI_API_BASE: "http://exo-api.default.svc.cluster.local:8000/v1"
OPENAI_MODEL: "llama-3.2-1b"
```

### Storage

- **Shared storage is required** between Web UI and worker pods:
  - Web uploads land in `/data/input`
  - Workers read from `/data/input` and write to `/data/output`
- **Single-node (local-path)**: pin web + worker to the same node or they will not see the same files.
- **Multi-node**: use RWX storage (NFS) so uploads are visible to all nodes.
  - See `deploy/k3s/base/nfs-pv.yaml` for reference PVs.

### Resources

- **Requests**: 4Gi RAM, 2 CPU
- **Limits**: 12Gi RAM, 4 CPU

## Access

Add to `/etc/hosts`:
```
YOUR_REGISTRY_IP  montage.local
```

Access: https://montage.local

Or port forward:
```bash
kubectl port-forward -n montage-ai svc/montage-ai 5000:5000
```

## Operations

```bash
# View logs
kubectl logs -n montage-ai -l app=montage-ai -f

# Update image
./deploy/k3s/build-and-push.sh
kubectl rollout restart deployment/montage-ai -n montage-ai

# Debug
kubectl exec -it -n montage-ai deployment/montage-ai -- /bin/bash
```
