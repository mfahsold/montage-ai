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

| Feature | Description |
|---------|-------------|
| 🎵 **Beat Sync** | Cuts align to music rhythm (librosa) |
| 🎬 **Style Templates** | hitchcock, mtv, action, documentary, minimalist, wes_anderson |
| 🤖 **AI Director** | Natural language → editing parameters |
| ⬆️ **AI Upscaling** | 4x resolution via Real-ESRGAN |
| ☁️ **Cloud GPU** | Free GPU via [cgpu](https://github.com/RohanAdwankar/cgpu) |
| 📽️ **Timeline Export** | OTIO/EDL for DaVinci Resolve, Premiere |

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
```

→ Full reference: [docs/configuration.md](docs/configuration.md)

---

## Styles

| Style | Description |
|-------|-------------|
| `dynamic` | Adapts to music energy (default) |
| `hitchcock` | Slow tension build, explosive climax |
| `mtv` | Rapid 1-2 beat cuts |
| `action` | Michael Bay intensity |
| `documentary` | Natural, observational |
| `minimalist` | Contemplative long takes |
| `wes_anderson` | Symmetric, warm, playful |

Custom styles: Create JSON in `src/montage_ai/styles/` or set `STYLE_PRESET_DIR`

→ Style guide: [docs/styles.md](docs/styles.md)

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

| Document | Description |
|----------|-------------|
| [Quick Start](docs/QUICKSTART.md) | First-time setup |
| [Installation](docs/INSTALL.md) | Docker, K8s, local |
| [Configuration](docs/configuration.md) | All environment variables |
| [Styles](docs/styles.md) | Style customization |
| [cgpu Setup](docs/CGPU_INTEGRATION.md) | Cloud GPU / Gemini |
| [Web UI](docs/web_ui.md) | Web interface guide |
| [Architecture](docs/architecture.md) | System design |
| [Models](docs/models.md) | AI model choices |

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
