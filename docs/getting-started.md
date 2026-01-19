# Getting Started

From zero to your first montage in 5 minutes.

---

## Requirements

- **Docker** + Docker Compose v2
- **8 GB RAM** (16 GB for high quality)
- Optional: [cgpu](https://github.com/RohanAdwankar/cgpu) for cloud GPU/LLM

> **Low RAM?** See [Hybrid Workflow](hybrid-workflow.md) to offload AI tasks to cloud.

---

## Installation

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
```

Everything runs in Docker.

---

## First Montage

### Web UI (Easiest)

```bash
make web
```

1. Open **http://localhost:5001**
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

Included scripts download open-source media (Big Buck Bunny, Sintel, etc.).

```bash
make download-assets
```

Media goes into `data/input/` and `data/music/`.

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
# Base deployment
kubectl apply -k deploy/k3s/base/

# Production overlay
kubectl apply -k deploy/k3s/overlays/production/

# Fast preview mode
kubectl apply -k deploy/k3s/overlays/dev/
```

Check job status:

```bash
kubectl logs -f job/montage-ai-render
```

---

## Next Steps

- **[Configuration](configuration.md)** — Tweak every setting
- **[Features](features.md)** — Learn about styles, effects, timeline export
- **[Troubleshooting](troubleshooting.md)** — When things go wrong
