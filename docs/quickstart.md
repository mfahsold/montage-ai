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

### 4. Start Web UI

```bash
docker compose up
```

Open http://localhost:8080 in your browser.

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
