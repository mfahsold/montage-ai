# Quick Reference: Montage AI

## Installation

```bash
# Core only (minimal)
pip install montage-ai

# With AI features (smart reframing, face detection, color matching)
pip install montage-ai[ai]

# With Web UI (Flask + Redis job queue)
pip install montage-ai[web]

# With Cloud GPU support (upscaling, remote analysis)
pip install montage-ai[cloud]

# Everything (development)
pip install montage-ai[all]
```

## Running Montage

### CLI (Fastest)
```bash
# Copy videos to data/input/
cp *.mp4 data/input/

# Run montage with a style
./montage-ai.sh run dynamic
./montage-ai.sh run hitchcock --quality high
./montage-ai.sh run "natural language prompt here"
```

**Styles:** `dynamic`, `hitchcock`, `mtv`, `action`, `documentary`, `minimalist`, `wes_anderson`

### Web UI (Easiest)
```bash
# Start web server
./montage-ai.sh web

# Open in browser
open http://localhost:8080
```

### Docker (Reproducible)
```bash
docker-compose up -d
# Web UI at http://localhost:8080
```

## Configuration

### Environment Variables (Performance Tuning)

**Quick Profiles:**
```bash
# Laptop (low memory)
BATCH_SIZE=2 QUALITY_PROFILE=preview ./montage-ai.sh run

# Workstation (balanced)
BATCH_SIZE=5 ./montage-ai.sh run

# Server (high quality)
BATCH_SIZE=10 QUALITY_PROFILE=high ./montage-ai.sh run
```

**Motion Analysis (Smart Reframing):**
```bash
# Fast (fewer calc intensive ops)
OPTICAL_FLOW_LEVELS=2 OPTICAL_FLOW_WINSIZE=7 OPTICAL_FLOW_ITERATIONS=2 ./montage-ai.sh run

# Precise (more detail)
OPTICAL_FLOW_LEVELS=4 OPTICAL_FLOW_WINSIZE=21 OPTICAL_FLOW_ITERATIONS=5 ./montage-ai.sh run
```

### Key Settings
```bash
QUALITY_PROFILE=preview|standard|high     # Output quality
BATCH_SIZE=2|5|10                         # Parallel processing
LOW_MEMORY_MODE=true|false                # Disk-based processing
SMART_REFRAME_ENABLED=true|false          # Face detection
STABILIZE=true|false                      # Video stabilization
UPSCALE=true|false                        # AI upscaling
CGPU_ENABLED=true|false                   # Cloud GPU jobs
```

### Full Config Reference
See [docs/configuration.md](docs/configuration.md)

## Testing & Validation

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_montage_builder.py -v

# CI pipeline (local)
./scripts/ci.sh

# Check for CVE vulnerabilities
pip-audit --desc
```

## Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with all optional dependencies
pip install -e ".[all]"

# Run tests with coverage
pytest tests/ --cov=src/montage_ai --cov-report=html

# Format code
black src/ tests/

# Type check
mypy src/
```

## Deployment

### K3s Cluster
```bash
# Build & push to registry
./deploy/k3s/build-and-push.sh

# Apply manifests
kubectl apply -k deploy/k3s/base/

# Check status
kubectl get pods -n montage-ai
kubectl logs -n montage-ai -l app=montage-ai -f
```

### Docker Registry
```bash
# Local build
docker build -t montage-ai:v1.0 .

# Tag and push
docker tag montage-ai:v1.0 registry.example.com/montage-ai:v1.0
docker push registry.example.com/montage-ai:v1.0
```

## Troubleshooting

### "No module named 'mediapipe'"
```bash
pip install montage-ai[ai]
```

### "ConnectionError: Can't connect to Redis"
```bash
# Start Redis
redis-server

# Or disable web UI
REDIS_ENABLED=false ./montage-ai.sh run
```

### "ffmpeg not found"
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Docker (included)
```

### Performance is slow
```bash
# Check docs/performance-tuning.md for profiles
# See also: docs/OPTIONAL_DEPENDENCIES.md

# Quick fix: reduce quality
QUALITY_PROFILE=preview ./montage-ai.sh run
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [README.md](README.md) | Project overview |
| [docs/getting-started.md](docs/getting-started.md) | Setup guide |
| [docs/features.md](docs/features.md) | Feature descriptions |
| [docs/architecture.md](docs/architecture.md) | System design |
| [docs/configuration.md](docs/configuration.md) | Environment variables |
| [docs/performance-tuning.md](docs/performance-tuning.md) | Optimization |
| [docs/OPTIONAL_DEPENDENCIES.md](docs/OPTIONAL_DEPENDENCIES.md) | Installation options |
| [docs/DEPENDENCY_AUDIT.md](docs/DEPENDENCY_AUDIT.md) | Dependency report |

## Common Scenarios

### Process folder of videos
```bash
mkdir data/input && cp *.mp4 data/input/
./montage-ai.sh run dynamic --quality high
# Output: data/output/montage_*.mp4
```

### Extract highlights from long video
```bash
cp long_stream.mp4 data/input/
./montage-ai.sh run dynamic --highlights
# Output: Clips with highest engagement potential
```

### Transcript-based editing
```bash
cp video.mp4 data/input/
./montage-ai.sh web
# Use web UI: upload, transcribe, edit by text
```

### Cloud upscaling
```bash
export CGPU_ENABLED=true
export CGPU_API_KEY=sk-...
./montage-ai.sh run dynamic --upscale
```

### Batch processing (multiple jobs)
```bash
for style in dynamic hitchcock mtv; do
  BATCH_SIZE=2 ./montage-ai.sh run $style &
done
wait
```

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/awesome`
3. Make changes and test: `make test`
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## License

PolyForm Noncommercial 1.0.0 — Free for personal use and open-source projects.  
See [LICENSE](LICENSE) for commercial licensing options.

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/mfahsold/montage-ai/issues)
- 💬 [Discussions](https://github.com/mfahsold/montage-ai/discussions)
- 🔒 [Security Policy](SECURITY.md)
