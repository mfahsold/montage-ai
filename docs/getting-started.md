# Getting Started

Choose the setup that fits your environment:

| Setup | When to use | Prerequisites |
|-------|-------------|---------------|
| **Docker** (recommended) | Local development, single machine | Docker + Docker Compose v2 |
| **Kubernetes** | Multi-node, production, cloud | kubectl, K3s/K8s cluster |
| **ARM64** | Raspberry Pi, Apple Silicon, Snapdragon | See [Getting Started on ARM](getting-started-arm.md) |

---

> All commands on this page assume you are in the repository root (`montage-ai/`).

## Docker Setup

### Prerequisites

- Docker + Docker Compose v2
- 16 GB RAM recommended (8 GB minimum with `DOCKER_MEMORY_LIMIT=6g QUALITY_PROFILE=preview`)
- 4+ CPU cores recommended
- 10+ GB free disk space

> **ARM64 (Apple Silicon, Raspberry Pi, Snapdragon):** Auto Reframe uses center-crop fallback (MediaPipe unavailable). All other features work normally.

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
docker compose build    # First build takes 5-15 min (downloads dependencies)
docker compose up
```

> **First build:** `docker compose build` downloads base images and installs all dependencies (~1.2 GB). This takes 5-15 minutes depending on your internet speed. Subsequent builds use the Docker cache and are much faster. If the build hangs, try `docker compose build --progress=plain` for verbose output.

> **Resource limits:** Docker defaults to 12 GB memory / 4 CPUs. If your system has more (or less) RAM, override:
> ```bash
> # 32 GB system:
> DOCKER_MEMORY_LIMIT=24g DOCKER_CPU_LIMIT=8 docker compose up
> # 8 GB system:
> DOCKER_MEMORY_LIMIT=6g DOCKER_CPU_LIMIT=2 QUALITY_PROFILE=preview docker compose up
> ```
> See [Performance Tuning](performance-tuning.md) for all options and [Configuration](configuration.md) for all environment variables.

> **GPU acceleration (VAAPI):** If `verify-deployment` reports `/dev/dri/renderD128` as "not readable", set the render group GID:
> ```bash
> RENDER_GID=$(stat -c %g /dev/dri/renderD128) docker compose up
> ```

Open http://localhost:8080 in your browser.

If port 8080 is already in use:

```bash
WEB_PORT=8081 docker compose up
# Then open http://localhost:8081
```

### 3.5. Verify installation

```bash
# Check container is running
docker ps | grep montage-ai

# Verify CLI is accessible
docker compose run --rm montage-ai /app/montage-ai.sh --help

# Check data directories are mounted
docker compose run --rm montage-ai ls /data/input /data/music /data/output
```

For a comprehensive test, see [Installation Test Guide](INSTALLATION_TEST.md).

### 4. CLI usage (alternative to Web UI)

```bash
# Full render
docker compose run --rm montage-ai /app/montage-ai.sh run

# Fast preview (lower quality, much faster)
QUALITY_PROFILE=preview docker compose run --rm montage-ai /app/montage-ai.sh run
```

Output files appear in `data/output/`.

> **See also:** [Features](features.md) for all styles and effects, [CLI Reference](CLI_REFERENCE.md) for all commands.

### 5. Re-running Setup & Idempotency

All setup steps are **idempotent** — safe to re-run without side effects:

| Command | Idempotent? | Notes |
|---------|-------------|-------|
| `./scripts/setup.sh` | Yes | Creates directories only if missing |
| `docker compose build` | Yes | Uses Docker layer cache; fast on repeat runs |
| `docker compose up` | Yes | Starts or resumes containers |
| `docker compose down` | Yes | Stops and removes containers (data volumes preserved) |

**When to rebuild from scratch:**

```bash
# After dependency changes (requirements.txt, pyproject.toml):
docker compose build --no-cache

# If containers won't start or behave unexpectedly:
docker compose down
docker compose build
docker compose up
```

**Full reset** (removes all generated output):

```bash
docker compose down
rm -rf data/output/*
./scripts/setup.sh
docker compose build --no-cache
docker compose up
```

> **Tip:** `docker compose up` without a prior `down` is fine — Docker Compose handles container lifecycle automatically. Use `down` only when you need a clean slate or to free resources.

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
