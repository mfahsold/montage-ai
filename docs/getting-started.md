# Getting Started

From zero to your first montage in 5 minutes.

---

## Requirements

- **Docker** + Docker Compose v2
- **8 GB RAM** (16 GB for high quality)
- Optional: [cgpu](https://github.com/RohanAdwankar/cgpu) for cloud GPU/LLM

> **Low RAM?** Use `QUALITY_PROFILE=preview` and consider cgpu offload. See
> [performance-tuning.md](performance-tuning.md) and [cgpu-setup.md](cgpu-setup.md).

---

## Installation

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
```

The default flow runs in Docker via `./montage-ai.sh`.
If you prefer a local Python install (pip/uv) or need optional extras, see
[Optional Dependencies](OPTIONAL_DEPENDENCIES.md).

---

## First Montage

### Web UI (Easiest)

```bash
./montage-ai.sh web
```

1. Open **http://localhost:8080** (or your configured `<MONTAGE_WEB_URL>`)
2. Upload video clips
3. Upload music track
4. Pick a style (or natural language prompt)
5. Click **Create Montage**
6. Download video

### Command Line

```bash
# Add media
cp ~/Videos/*.mp4 data/input/
cp ~/Music/track.mp3 data/music/

# Run
./montage-ai.sh run

# Find output
ls data/output/
```

## Test Assets

Sample assets are not bundled in the public repo. Use your own clips, or see
[test-assets.md](test-assets.md) for public-domain sources and synthetic fixtures.

---

## Run Options

### Editing Styles
```bash
./montage-ai.sh run                # dynamic (default)
./montage-ai.sh run hitchcock      # suspense
./montage-ai.sh run mtv            # fast cuts
./montage-ai.sh run documentary    # natural
```

### Quality Modes

```bash
./montage-ai.sh preview hitchcock  # Fast preview (low quality)
./montage-ai.sh run hitchcock      # Normal quality
./montage-ai.sh hq hitchcock       # High quality + stabilization
```

### Custom Prompts

Skip the presets and just describe what you want:

```bash
CREATIVE_PROMPT="edit like a 90s skateboard video" ./montage-ai.sh run
```

---

## Production Deployment

For production environments, we recommend using Redis for session persistence.

### Redis Configuration

Set the following environment variables:

```bash
REDIS_HOST=redis
REDIS_PORT=6379
```

If running in Kubernetes, use the provided manifests in `deploy/k3s`.

## Docker Compose

If you prefer raw Docker:

```bash
# Web UI
docker compose -f docker-compose.web.yml up

# CLI job
docker compose up
```

---

## Kubernetes

For cluster deployments:

```bash
# Render config
make -C deploy/k3s config

# Canonical cluster overlay
make -C deploy/k3s deploy-cluster
```

The canonical cluster overlay is `deploy/k3s/overlays/cluster`.

Check job status:

```bash
python3 -m montage_ai.cli jobs --api-base http://<cluster-service> list
```

---

## Next Steps

- **[Configuration](configuration.md)** — Tweak every setting
- **[Features](features.md)** — Learn about styles, effects, timeline export
- **[Troubleshooting](troubleshooting.md)** — When things go wrong
