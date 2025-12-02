# Quick Start Guide

Get Montage AI running in 5 minutes.

---

## Option 1: Web UI (Easiest)

### Start Web Interface

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
make web
```

Open **http://localhost:5000**

### Use It

1. Upload videos and music
2. Choose style (Hitchcock, MTV, etc.)
3. Click "Create Montage"
4. Download result

---

## Option 2: Command Line

### Basic Usage

```bash
# 1. Add your media
cp your_videos/*.mp4 data/input/
cp your_music.mp3 data/music/

# 2. Create montage
./montage-ai.sh run hitchcock

# 3. Output at data/output/
```

### With Options

```bash
# Upscale + Timeline Export
./montage-ai.sh run hitchcock \
  --upscale \
  --export-timeline \
  --cgpu-gpu
```

---

## Option 3: Docker

### Using Docker Compose

```bash
docker-compose up
```

### Manual Docker

```bash
docker build -t montage-ai .

docker run \
  -v $(pwd)/data:/data \
  montage-ai
```

---

## Option 4: Kubernetes

### Deploy to Cluster

```bash
make deploy

# Or with overlay
make deploy-prod  # AMD GPU node
make deploy-dev   # Fast preview
```

### Run Job

```bash
kubectl apply -f deploy/k3s/base/job.yaml
kubectl logs -f job/montage-ai-render
```

---

## Configuration

### Environment Variables

```bash
# Editing
export CUT_STYLE=hitchcock
export CREATIVE_PROMPT="suspenseful editing"

# Enhancement
export STABILIZE=true
export UPSCALE=true
export ENHANCE=true

# Export
export EXPORT_TIMELINE=true
export GENERATE_PROXIES=true

# LLM
export OLLAMA_HOST=http://localhost:11434
export CGPU_ENABLED=true

# Run
./montage-ai.sh run
```

### Style Presets

| Style | Description |
|-------|-------------|
| `dynamic` | Position-aware (default) |
| `hitchcock` | Suspense, slow build |
| `mtv` | Rapid 1-2 beat cuts |
| `action` | Michael Bay fast cuts |
| `documentary` | Natural pacing |
| `minimalist` | Long contemplative takes |
| `wes_anderson` | Symmetric, stylized |

---

## Next Steps

- [Features Documentation](features.md)
- [Configuration Guide](configuration.md)
- [Web UI Guide](web_ui.md)
- [Timeline Export](timeline_export.md)

---

## Troubleshooting

### "No videos found"

```bash
ls data/input/  # Check videos are present
```

### "No music found"

```bash
ls data/music/  # Check music is present
```

### "FFmpeg not found"

```bash
# Install FFmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg
```

### "OpenTimelineIO not installed"

```bash
pip install OpenTimelineIO>=0.16.0
```

### Web UI not accessible

```bash
# Check if running
docker ps | grep web-ui

# Check logs
docker-compose -f docker-compose.web.yml logs
```

---

## Examples

### Travel Video Montage

```bash
# Copy footage from phone
cp ~/DCIM/Camera/*.mp4 data/input/

# Add music
cp ~/Music/vacation_song.mp3 data/music/

# Create montage
./montage-ai.sh run documentary
```

### Music Video (Beat-Synced)

```bash
# Fast-paced MTV style
./montage-ai.sh run mtv

# Or with prompt
CREATIVE_PROMPT="energetic music video" ./montage-ai.sh run
```

### Event Highlights

```bash
# Wedding footage → highlight reel
./montage-ai.sh run wes_anderson
```

### Professional Workflow

```bash
# Export timeline for DaVinci Resolve
./montage-ai.sh run hitchcock \
  --export-timeline \
  --generate-proxies

# Import .otio file into Resolve
# Fine-tune color grading
# Export final video
```

---

## Help

```bash
./montage-ai.sh --help
make help
```

---

**Ready to create?** Start with the Web UI: `make web`
