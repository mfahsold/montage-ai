# Getting Started

Everything you need to go from zero to your first montage.

---

## Requirements

- **Docker** + Docker Compose v2
- **8 GB RAM** minimum (16 GB recommended for high-quality mode)
- Optional: [cgpu](https://github.com/RohanAdwankar/cgpu) for cloud GPU/LLM

> **Laptop User?** Check out the [Hybrid Workflow](hybrid-workflow.md) to run Montage AI on 16GB RAM by offloading AI tasks to the cloud.

---

## Installation

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
```

That's it. Everything runs in Docker.

---

## Your First Montage

### Web UI (recommended)

```bash
make web
```

1. Open **http://localhost:5001**
2. Upload some video clips
3. Upload a music track
4. Pick a style (or type a prompt)
5. Click **Create Montage**
6. Download your video

![Web UI Workflow](images/web-ui-workflow.png)
<!-- TODO: Add screenshot showing the upload and creation process -->

### Command Line

```bash
# 1. Add your media
cp ~/Videos/*.mp4 data/input/
cp ~/Music/track.mp3 data/music/

# 2. Run
./montage-ai.sh run

# 3. Find output
ls data/output/
```

## Test Assets

The project includes scripts to download open-source test material.

**Video (Open Movie Project):**
- `BigBuckBunny.mp4` (151 MB)
- `Sintel.mp4` (182 MB)
- `TearsOfSteel.mp4` (178 MB)

**Audio (SoundHelix):**
- `TestTrack1.mp3`
- `TestTrack2.mp3`
- `TestTrack3.mp3`

These files are stored in `data/input/` and `data/music/`.

---

## Run Options

### Basic Styles

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
CREATIVE_PROMPT="moody and slow, lots of wide shots" ./montage-ai.sh run
```

---

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
make deploy           # Default
make deploy-prod      # Production overlay
make deploy-dev       # Fast preview mode
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
