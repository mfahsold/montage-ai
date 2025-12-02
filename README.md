# Montage AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> Turn hours of raw footage into beat-synchronized montages in minutes.

AI-powered video editor that automatically creates cinematic montages from your clips. Analyzes music beats, detects scenes, and assembles footage using AI-driven creative decisions.

---

## Quick Start

### Web UI (Recommended)

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
make web
```

Open **http://localhost:5001** → Select files → Create Montage → Download

### Command Line

```bash
# Build
./montage-ai.sh build

# Add media
cp /path/to/videos/* data/input/
cp /path/to/music.mp3 data/music/

# Create montage
./montage-ai.sh run              # Default style
./montage-ai.sh run hitchcock    # Suspense style
./montage-ai.sh run mtv          # Fast cuts
./montage-ai.sh hq               # High quality + stabilization
```

Output: `data/output/montage.mp4`

---

## Features

| Feature               | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| 🎵 **Beat Sync**       | Cuts align to music rhythm (librosa)                          |
| 🎬 **Style Templates** | hitchcock, mtv, action, documentary, minimalist, wes_anderson |
| 🤖 **AI Director**     | Natural language → editing parameters                         |
| ⬆️ **AI Upscaling**    | 4x resolution via Real-ESRGAN                                 |
| ☁️ **Cloud GPU**       | Free GPU via [cgpu](https://github.com/RohanAdwankar/cgpu)    |
| 📽️ **Timeline Export** | OTIO/EDL for DaVinci Resolve, Premiere                        |
| 🧠 **Memory Management** | Auto-cleanup, optimized for low-resource systems            |

---

## Configuration

All settings via environment variables. Key options:

```bash
# Style
CUT_STYLE=hitchcock              # Style preset
CREATIVE_PROMPT="tense thriller" # Natural language (overrides style)

# Enhancements
STABILIZE=true                   # Video stabilization
UPSCALE=true                     # AI 4x upscaling
ENHANCE=true                     # Color/sharpness (default)

# AI Backend (choose one)
GOOGLE_API_KEY=xxx               # Google AI (preferred)
OLLAMA_HOST=http://localhost:11434  # Local Ollama

# Cloud GPU (for upscaling)
CGPU_GPU_ENABLED=true            # Use Google Colab T4

# Memory Management (NEW - for stability)
MEMORY_LIMIT_GB=16               # Docker memory limit
AUTO_CLEANUP=true                # Delete temp files automatically
```

→ Full reference: [docs/configuration.md](docs/configuration.md)
→ **Stability improvements:** [STABILITY_IMPROVEMENTS.md](STABILITY_IMPROVEMENTS.md)

---

## Styles

| Style          | Description                          |
| -------------- | ------------------------------------ |
| `dynamic`      | Adapts to music energy (default)     |
| `hitchcock`    | Slow tension build, explosive climax |
| `mtv`          | Rapid 1-2 beat cuts                  |
| `action`       | Michael Bay intensity                |
| `documentary`  | Natural, observational               |
| `minimalist`   | Contemplative long takes             |
| `wes_anderson` | Symmetric, warm, playful             |

Custom styles: Create JSON in `src/montage_ai/styles/` or set `STYLE_PRESET_DIR`

→ Style guide: [docs/features.md](docs/features.md#style-templates-built-in)

---

## Architecture

```
Input Clips + Music
        ↓
┌───────────────────┐
│  Beat Detection   │  ← librosa
│  Scene Detection  │  ← PySceneDetect
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  Creative Director│  ← LLM (Gemini/Ollama)
│  (Style → Params) │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  Footage Manager  │  ← Story arc selection
│  Editor Assembly  │  ← Beat-aligned cuts
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  FFmpeg Render    │  ← Stabilization, upscaling
└───────────────────┘
          ↓
    Final Video
```

→ Details: [docs/architecture.md](docs/architecture.md)

---

## Documentation

| Document                                   | Description                    |
| ------------------------------------------ | ------------------------------ |
| [Quick Start](docs/QUICKSTART.md)          | Fast setup paths               |
| [Configuration](docs/configuration.md)     | Environment variables          |
| [Features & Workflows](docs/features.md)   | Features, styles, Web UI, export|
| [Architecture](docs/architecture.md)       | System design                  |
| [Models](docs/models.md)                   | AI model choices               |
| [Stability Improvements](STABILITY_IMPROVEMENTS.md) | **NEW:** Memory & GPU fixes |

---

## Troubleshooting

### Memory Issues

**Problem:** Jobs crash with "Out of Memory" or container killed

**Solution:**
```bash
# Reduce memory footprint
export MEMORY_LIMIT_GB=12
export MAX_CLIPS_IN_RAM=30
export PARALLEL_ENHANCE=false
export FFMPEG_PRESET=ultrafast

# Enable Cloud GPU instead of local processing
export CGPU_GPU_ENABLED=true
export UPSCALE=true  # Offload upscaling to Colab
```

### Cloud GPU Failures

**Problem:** CUDA operations fail with "PIPELINE_SUCCESS not found"

**Solution:**
```bash
# Check cgpu connection
cgpu status

# Enable detailed logging
export VERBOSE=true

# Check logs for CUDA error diagnosis
docker logs montage-ai | grep "CUDA Error"
```

**Common CUDA errors:**
- `CUDA out of memory` → Video too large, reduce resolution or use smaller clips
- `CUDA not available` → Colab lost GPU, run `cgpu status` to reconnect
- `session expired` → Automatic retry enabled (2 attempts)

### Slow Performance

**Problem:** Rendering takes too long

**Solution:**
```bash
# Optimize for speed
export FFMPEG_PRESET=ultrafast
export PARALLEL_ENHANCE=true
export MAX_PARALLEL_JOBS=8

# Skip expensive operations
export STABILIZE=false
export UPSCALE=false
export DEEP_ANALYSIS=false
```

### Temp Files Fill Disk

**Problem:** `/tmp` runs out of space

**Solution:**
```bash
# Enable automatic cleanup (default since v2.1)
export AUTO_CLEANUP=true

# Manual cleanup if needed
docker exec montage-ai rm -rf /tmp/*.mp4
```

See [STABILITY_IMPROVEMENTS.md](STABILITY_IMPROVEMENTS.md) for detailed diagnostics.

---

## Development

```bash
# Run tests
make test

# Local development
make dev

# Validate K8s manifests
make validate
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## License

MIT — see [LICENSE](LICENSE)
