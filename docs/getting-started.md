# Getting Started

Choose the setup that fits your environment:

| Setup | When to use | Prerequisites |
|-------|-------------|---------------|
| **Docker** (recommended) | Local development, single machine | Docker + Docker Compose v2 |
| **Kubernetes** | Multi-node, production, cloud | kubectl, K3s/K8s cluster |
| **ARM64** | Raspberry Pi, Apple Silicon, Snapdragon | See [Getting Started on ARM](getting-started-arm.md) |

---

## Docker Setup

### Prerequisites

- Docker + Docker Compose v2
- 16 GB RAM recommended (8 GB minimum with `DOCKER_MEMORY_LIMIT=6g QUALITY_PROFILE=preview`)
- 4+ CPU cores recommended
- 10+ GB free disk space

Verify:

```bash
docker --version          # Docker 20+ required
docker compose version    # Compose v2 required
```

### 1. Clone and set up

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
./scripts/setup.sh
```

### 2. Add media files

```bash
# Copy your video clips
cp ~/Videos/*.mp4 data/input/

# Copy a music track
cp ~/Music/track.mp3 data/music/
```

Or generate test media:

```bash
./scripts/ops/create-test-video.sh
```

### 3. Build and run

```bash
docker compose build
docker compose up
```

Open http://localhost:8080 in your browser.

If port 8080 is already in use:

```bash
WEB_PORT=8081 docker compose up
# Then open http://localhost:8081
```

### 4. CLI usage (alternative to Web UI)

```bash
# Full render
docker compose run --rm montage-ai /app/montage-ai.sh run

# Fast preview (lower quality, much faster)
QUALITY_PROFILE=preview docker compose run --rm montage-ai /app/montage-ai.sh run
```

Output files appear in `data/output/`.

---

## Kubernetes

For cluster deployments (K3s, K8s, EKS, GKE), see the dedicated guide:

**[Cluster Deployment Guide](cluster-deploy.md)** — Full walkthrough with prerequisites, storage setup, troubleshooting, and rollback.

Quick summary:

```bash
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
$EDITOR deploy/k3s/config-global.yaml   # Replace all <...> placeholders
make -C deploy/k3s config
make -C deploy/k3s pre-flight
make -C deploy/k3s deploy-cluster
```

For advanced topics (multi-arch builds, KEDA, distributed rendering), see [deploy/k3s/README.md](../deploy/k3s/README.md).

---

## Next Steps

- **[Configuration](configuration.md)** — All environment variables and settings
- **[Features](features.md)** — Styles, effects, timeline export, feature matrix
- **[Optional Dependencies](OPTIONAL_DEPENDENCIES.md)** — SOTA models and AI/ML library options
- **[Troubleshooting](troubleshooting.md)** — Common issues and fixes
- **[Performance Tuning](performance-tuning.md)** — Optimization for your hardware
