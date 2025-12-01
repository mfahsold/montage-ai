# Montage AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> Turn hours of raw footage into beat-synchronized montages in minutes.

**Montage-AI** is an open-source video editor that automatically creates cinematic montages from your clips. It analyzes music beats, detects scenes, and assembles footage using AI-driven creative decisions.

<!-- TODO: Add demo video link when available -->
<!-- [![Demo](https://img.youtube.com/vi/XXXXX/0.jpg)](https://youtube.com/watch?v=XXXXX) -->

---

## Why Montage-AI?

Creating engaging video montages manually requires:

- **Hours of beat-syncing** — aligning cuts to music rhythm
- **Clip selection expertise** — knowing which shots work together
- **Story arc planning** — structuring intro, build, climax, outro

Montage-AI automates this with proven algorithms:

| Task | Manual | Montage-AI |
|------|--------|------------|
| Beat synchronization | 2-3 hours | Automatic ([librosa](https://librosa.org/)) |
| Scene detection | Manual review | Automatic ([PySceneDetect](https://scenedetect.com/)) |
| Clip selection | Requires experience | AI-driven (LLM) |
| Upscaling | Expensive software | Free ([Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)) |

### vs. Other Tools

| | Adobe Premiere | DaVinci Resolve | Descript | **Montage-AI** |
|-|----------------|-----------------|----------|----------------|
| Beat sync | Manual | Manual | ❌ | ✅ Auto |
| AI editing | ❌ | ❌ | ✅ Transcript | ✅ Creative |
| Cinematic styles | Manual | Manual | ❌ | ✅ Presets |
| Story arc | Manual | Manual | ❌ | ✅ Auto |
| Open source | ❌ | ❌ | ❌ | ✅ MIT |
| Cost | $23/mo | Free | $15/mo | **Free** |

**Best for:** Travel videos, event highlights, music videos, social media content.

---

## Quick Start

\`\`\`bash
# 1. Clone and build
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
./montage-ai.sh build

# 2. Add your media
cp /path/to/videos/* data/input/
cp /path/to/music.mp3 data/music/

# 3. Create montage
./montage-ai.sh run
\`\`\`

Output: \`data/output/montage.mp4\`

→ **[Full Installation Guide](docs/INSTALL.md)** | **[Kubernetes Deployment](deploy/k3s/README.md)**

---

## Features

| Feature | Description |
|---------|-------------|
| 🎵 **Beat Sync** | Cuts align to music rhythm via [librosa](https://librosa.org/) beat detection |
| 🎬 **Style Templates** | Hitchcock, MTV, documentary, action, minimalist presets |
| 🤖 **AI Director** | Natural language → editing parameters via LLM |
| 📖 **Story Arc** | Automatic intro/build/climax/outro structure |
| ⬆️ **AI Upscaling** | 4x resolution via [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| ☁️ **Cloud GPU** | Free cloud processing via [cgpu](https://github.com/RohanAdwankar/cgpu) |
| 🐳 **Deploy Anywhere** | Docker, Kubernetes, local |

---

## Usage

\`\`\`bash
./montage-ai.sh run [STYLE]      # Create montage
./montage-ai.sh preview [STYLE]  # Fast preview (lower quality)
./montage-ai.sh hq [STYLE]       # High quality + stabilization
./montage-ai.sh list             # Show available styles
\`\`\`

### Styles

| Style | Description |
|-------|-------------|
| \`dynamic\` | Position-aware pacing (default) |
| \`hitchcock\` | Suspense — slow build, fast climax |
| \`mtv\` | Rapid 1-2 beat cuts |
| \`action\` | Michael Bay intensity |
| \`documentary\` | Natural, observational |
| \`minimalist\` | Contemplative long takes |

### Options

\`\`\`bash
./montage-ai.sh run --stabilize  # Video stabilization
./montage-ai.sh run --upscale    # AI 4x upscaling
./montage-ai.sh run --cgpu       # Use cloud LLM (faster)
\`\`\`

---

## How It Works

\`\`\`text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Input     │────▶│   Analyze   │────▶│   Arrange   │────▶│   Render    │
│  Clips +    │     │  • Beats    │     │  • Select   │     │  • FFmpeg   │
│  Music      │     │  • Scenes   │     │  • Order    │     │  • Effects  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │                   │
                    ┌─────▼─────┐       ┌─────▼─────┐
                    │  librosa  │       │    LLM    │
                    │PySceneDetect      │ (Ollama/  │
                    └───────────┘       │  Gemini)  │
                                        └───────────┘
\`\`\`

1. **Beat Detection** — [librosa](https://librosa.org/) analyzes music tempo and beat positions
2. **Scene Detection** — [PySceneDetect](https://scenedetect.com/) identifies cut points in clips
3. **Clip Selection** — LLM (Llama 3.1 or Gemini) matches clips to story arc positions
4. **Assembly** — Clips arranged to align cuts with beats
5. **Rendering** — FFmpeg encodes final video with optional enhancements

→ **[Architecture Details](docs/architecture.md)** | **[AI Models Documentation](docs/models.md)**

---

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/INSTALL.md) | Docker, Kubernetes, local setup |
| [Features](docs/features.md) | Detailed feature documentation |
| [Styles](docs/styles.md) | Style templates and customization |
| [Architecture](docs/architecture.md) | System design |
| [AI Models](docs/models.md) | Model choices and rationale |
| [cgpu Integration](docs/CGPU_INTEGRATION.md) | Cloud GPU setup |
| [Configuration](docs/configuration.md) | Environment variables |

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

\`\`\`bash
git checkout -b feature/your-feature
make test
git commit -m "feat: description"
\`\`\`

**Good first issues:** Check [issues labeled "good first issue"](https://github.com/mfahsold/montage-ai/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

---

## Built With

- [librosa](https://librosa.org/) — Audio analysis (beat detection)
- [PySceneDetect](https://scenedetect.com/) — Scene cut detection
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — AI upscaling
- [MoviePy](https://zulko.github.io/moviepy/) — Video composition
- [Ollama](https://ollama.ai/) / [cgpu](https://github.com/RohanAdwankar/cgpu) — LLM backends
- [FFmpeg](https://ffmpeg.org/) — Video encoding

→ **[Why these libraries?](docs/models.md)**

---

## License

MIT — see [LICENSE](LICENSE)
