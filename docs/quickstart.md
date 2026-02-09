# 5-Minute Quick Start

Get Montage AI running in 5 minutes.

---

## Prerequisites

- Docker + Docker Compose v2
- 16 GB RAM recommended (8 GB minimum with preview mode)

Verify:

```bash
docker --version          # Docker 20+ required
docker compose version    # Compose v2 required
```

---

## Steps

### 1. Clone

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
```

### 2. Setup

```bash
./scripts/setup.sh
```

### 3. Add Media

```bash
# Option A: Generate test video
./scripts/ops/create-test-video.sh

# Option B: Use your own footage
cp ~/Videos/*.mp4 data/input/
cp ~/Music/track.mp3 data/music/
```

### 4. Build and Start Web UI

```bash
docker compose build    # First build: 5-15 min (subsequent builds use cache)
docker compose up
```

Open http://localhost:8080 in your browser.

> **Build taking a while?** The first build downloads base images and ~40 Python packages (~1.2 GB). Subsequent builds use Docker cache and are much faster. If the build hangs for >30 minutes, check your network (`ping registry-1.docker.io`) and disk space (`df -h`, needs 3+ GB free). Use `docker compose build --progress=plain` for verbose output.

### 4.5. Verify Installation

Before creating your first montage, verify everything is working:

```bash
# Check container is running
docker ps | grep montage-ai

# Verify CLI is accessible
docker compose run --rm montage-ai /app/montage-ai.sh --help

# Check data directories are mounted
docker compose run --rm montage-ai ls /data/input /data/music /data/output
```

All three commands should succeed. If not, see [Troubleshooting](troubleshooting.md) or the full [Installation Test Guide](INSTALLATION_TEST.md).

### 5. Create Your First Montage

1. Select a style (e.g., "Dynamic" or "Hitchcock")
2. Click "Create Montage"
3. Output appears in `data/output/`

---

## Alternative: CLI Usage

```bash
# Full render
docker compose run --rm montage-ai ./montage-ai.sh run

# Fast preview (360p, much faster)
QUALITY_PROFILE=preview docker compose run --rm montage-ai ./montage-ai.sh run

# Specific style
docker compose run --rm montage-ai ./montage-ai.sh run hitchcock
```

---

## Working Directory

> All commands on this page assume you are in the repository root (`montage-ai/`).

---

## Troubleshooting

**Port 8080 already in use?**

```bash
WEB_PORT=8081 docker compose up
```

**Out of memory?**

```bash
QUALITY_PROFILE=preview docker compose up
```

**Docker won't start?**
Lower the memory limit in `docker-compose.yml` (default is 12g).

---

## Next Steps

| I want to... | Read this |
|--------------|-----------|
| See all features | [Features](features.md) |
| Configure settings | [Configuration](configuration.md) |
| Deploy to Kubernetes | [Cluster Deployment](cluster-deploy.md) |
| Fix an error | [Troubleshooting](troubleshooting.md) |
| Tune performance | [Performance Tuning](performance-tuning.md) |
