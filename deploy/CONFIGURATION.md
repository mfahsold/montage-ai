# Deployment Configuration

This guide covers configuration for **production deployments** (Docker Compose, Kubernetes).

For runtime configuration (styles, LLMs, quality profiles), see [`docs/configuration.md`](../docs/configuration.md).

---

## Docker Compose

Override settings at runtime via environment variables:

```bash
# Style and quality
CUT_STYLE=hitchcock QUALITY_PROFILE=high docker compose up

# Resource limits
DOCKER_MEMORY_LIMIT=24g DOCKER_CPU_LIMIT=8 docker compose up

# Port override
WEB_PORT=8081 docker compose up
```

Key variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUT_STYLE` | `dynamic` | Editing style |
| `QUALITY_PROFILE` | `standard` | Quality preset (preview/standard/high/master) |
| `DOCKER_MEMORY_LIMIT` | `12g` | Container memory limit |
| `DOCKER_CPU_LIMIT` | `4` | Container CPU limit |
| `WEB_PORT` | `8080` | Host port for Web UI |
| `FFMPEG_HWACCEL` | `auto` | GPU encoder (auto/none/nvenc/vaapi/qsv) |

---

## Kubernetes

Cluster settings are managed via `deploy/k3s/config-global.yaml`:

```bash
# 1. Copy and edit configuration
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
$EDITOR deploy/k3s/config-global.yaml   # Replace all <...> placeholders

# 2. Render manifests from config
make -C deploy/k3s config

# 3. Deploy
make -C deploy/k3s deploy-cluster
```

Key settings in `config-global.yaml`:

| Setting | Example | Description |
|---------|---------|-------------|
| `cluster.namespace` | `montage-ai` | Kubernetes namespace |
| `cluster.domain` | `cluster.local` | Cluster DNS domain |
| `storage.classes.default` | `nfs-client` | StorageClass for PVCs |
| `storage.nfs.server` | `192.168.1.10` | NFS server IP |
| `registry.host` | `ghcr.io/org` | Container registry |
| `workerMaxReplicas` | `24` | Max worker pod replicas |

After editing, always re-render and re-deploy:

```bash
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster
```

---

## Environment Files

| File | Purpose |
|------|---------|
| `deploy/config.env` | Deployment defaults (registry, namespace, resource limits, service ports) |
| `deploy/k3s/config-global.yaml` | Cluster-specific settings (IPs, storage, scaling) |
| `deploy/k3s/base/cluster-config.env` | Auto-generated from `config-global.yaml` (do not edit manually) |
| `docker-compose.yml` | Docker Compose service definitions and environment variables |

---

## Reference

- [Full Configuration Reference](../docs/configuration.md) — All environment variables and settings
- [Cluster Deployment Guide](../docs/cluster-deploy.md) — Step-by-step K8s setup
- [Performance Tuning](../docs/performance-tuning.md) — Resource optimization
