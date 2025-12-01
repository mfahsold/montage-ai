# Montage AI

AI-powered video montage with beat-synchronized editing.

## Features

- **🎵 Beat Sync** - Automatic cut alignment to music rhythm
- **🎬 Style Templates** - Hitchcock, MTV, documentary, and more
- **🤖 Natural Language** - Describe your vision, AI translates to edits
- **📖 Story Arc** - Intelligent intro/build/climax/outro structure
- **🎨 Enhancement** - Stabilization, AI upscaling, color grading
- **☁️ Cloud GPU** - Free upscaling via [cgpu](https://github.com/RohanAdwankar/cgpu)

**[📚 Full Documentation](docs/README.md)** | **[🔧 Configuration](docs/configuration.md)** | **[🎭 Style Guide](docs/styles.md)**

---

## Quick Start

```bash
# Build once
./montage-ai.sh build

# Run with default style
./montage-ai.sh run

# Or pick a style
./montage-ai.sh run hitchcock
./montage-ai.sh run mtv
./montage-ai.sh run documentary
```

## Commands

| Command | Description |
|---------|-------------|
| `run [STYLE]` | Create montage |
| `preview [STYLE]` | Fast preview |
| `hq [STYLE]` | High quality render |
| `list` | Show available styles |
| `build` | Build Docker image |

## Styles

| Style | Description |
|-------|-------------|
| `hitchcock` | Slow build, explosive climax, high contrast |
| `mtv` | Rapid 1-2 beat cuts, maximum energy |
| `action` | Michael Bay fast cuts, motion preference |
| `documentary` | Natural pacing, longer takes |
| `minimalist` | Contemplative long takes |
| `wes_anderson` | Symmetric framing, warm colors |

→ [Full style documentation](docs/styles.md)

## Options

```bash
./montage-ai.sh run --stabilize              # Stabilization
./montage-ai.sh run --upscale                # AI upscaling
./montage-ai.sh run --variants 3             # Multiple versions
./montage-ai.sh hq hitchcock --stabilize     # HQ + stabilize
```

### Cloud Features (cgpu)

```bash
./montage-ai.sh run --cgpu                      # Gemini LLM
./montage-ai.sh run --cgpu-gpu --upscale        # Cloud GPU upscaling
./montage-ai.sh hq hitchcock --cgpu --cgpu-gpu  # Full cloud mode
```

→ [cgpu setup guide](docs/CGPU_INTEGRATION.md)

## Data Structure

```text
data/
├── input/   # Source clips (any format)
├── music/   # Soundtrack files
├── assets/  # Overlays, logos
└── output/  # Generated videos
```

## Requirements

- Docker & Docker Compose
- [Ollama](https://ollama.ai) (local LLM) or [cgpu](https://github.com/RohanAdwankar/cgpu) (cloud)

## Documentation

| Document | Description |
|----------|-------------|
| [Features](docs/features.md) | Detailed feature documentation |
| [Configuration](docs/configuration.md) | All environment variables |
| [Styles](docs/styles.md) | Style templates and customization |
| [Architecture](docs/architecture.md) | System design |
| [cgpu Integration](docs/CGPU_INTEGRATION.md) | Cloud GPU setup |

## License

MIT
