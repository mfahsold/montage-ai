# Montage AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI-powered video montage with beat-synchronized editing.

## Features

- **🎵 Beat Sync** - Automatic cut alignment to music rhythm
- **🎬 Style Templates** - Hitchcock, MTV, documentary, and more
- **🤖 Natural Language** - Describe your vision, AI translates to edits
- **📖 Story Arc** - Intelligent intro/build/climax/outro structure
- **🎨 Enhancement** - Stabilization, AI upscaling, color grading
- **☁️ Cloud GPU** - Free upscaling via [cgpu](https://github.com/RohanAdwankar/cgpu)

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
./montage-ai.sh build

# Add your media
cp /path/to/videos/* data/input/
cp /path/to/music.mp3 data/music/

# Create montage
./montage-ai.sh run
```

→ **[Full Installation Guide](docs/INSTALL.md)**

## Usage

```bash
# Basic commands
./montage-ai.sh run [STYLE]      # Create montage
./montage-ai.sh preview [STYLE]  # Fast preview
./montage-ai.sh hq [STYLE]       # High quality
./montage-ai.sh list             # Show styles

# With options
./montage-ai.sh run --stabilize  # Add stabilization
./montage-ai.sh run --upscale    # AI upscaling
./montage-ai.sh run --cgpu       # Use cloud LLM
```

## Styles

| Style          | Description                        |
| -------------- | ---------------------------------- |
| `dynamic`      | Position-aware pacing (default)    |
| `hitchcock`    | Suspense - slow build, fast climax |
| `mtv`          | Rapid 1-2 beat cuts                |
| `action`       | Michael Bay fast cuts              |
| `documentary`  | Natural, observational             |
| `minimalist`   | Contemplative long takes           |
| `wes_anderson` | Symmetric, stylized                |

→ [Style documentation](docs/styles.md)

## Deployment

### Docker (Local)

```bash
./montage-ai.sh build && ./montage-ai.sh run
```

### Kubernetes

```bash
kubectl apply -k deploy/k3s/base/
```

→ [Kubernetes guide](deploy/k3s/README.md)

### Development

```bash
make help      # Show all commands
make build     # Build image
make test      # Run tests
make deploy    # Deploy to K8s
```

## Documentation

| Document                                     | Description                     |
| -------------------------------------------- | ------------------------------- |
| **[Installation](docs/INSTALL.md)**          | Setup guide (Docker, K8s, cgpu) |
| [Features](docs/features.md)                 | Detailed feature documentation  |
| [Configuration](docs/configuration.md)       | Environment variables           |
| [Styles](docs/styles.md)                     | Style templates                 |
| [Architecture](docs/architecture.md)         | System design                   |
| [cgpu Integration](docs/CGPU_INTEGRATION.md) | Cloud GPU setup                 |

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development workflow
git checkout -b feature/your-feature
make test
# ... make changes ...
git commit -m "feat: your feature"
```

## License

MIT - see [LICENSE](LICENSE)
