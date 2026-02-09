# Contributing

Hey! Thanks for thinking about contributing. Here's how to get started.

---

## Quick Setup (for Contributors)

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/montage-ai.git
cd montage-ai

# Run tests
./scripts/ci.sh
```

---

## New Developer Setup (First Time)

### Step 1: System Check

Before cloning, verify your machine meets requirements:

```bash
# Check Docker + Compose
docker --version       # >= 20.10
docker compose version # >= v2.0

# Check resources
free -h | grep Mem     # At least 16 GB RAM
nproc                  # At least 4 cores
```

### Step 2: Clone & Local Test

```bash
# Clone and enter directory
git clone https://github.com/YOUR_USERNAME/montage-ai.git
cd montage-ai

# Test Docker setup (builds image, takes ~2-3 min first time)
docker compose build

# Verify build succeeded
docker compose run --rm montage-ai python -c "import montage_ai; print('✅ Ready!')"
```

### Step 3: Run First Montage Locally

```bash
# Add test media (or your own)
mkdir -p data/input data/music
cp test_data/sample_*.mp4 data/input/
cp test_data/sample_music.mp3 data/music/

# Run preview render (fast, low quality)
QUALITY_PROFILE=preview docker compose run --rm montage-ai ./montage-ai.sh run

# Check output
ls -lh data/output/*.mp4
```

### Step 4: Web UI Test (Optional)

```bash
# Start Web UI
docker compose up

# Open http://localhost:8080 in browser
# Upload videos, hit "Create Montage"
```

### Step 5: Run Test Suite

```bash
# Full CI (includes type checks, tests, linting)
./scripts/ci.sh

# Or just unit tests
pytest tests/ -v
```

### Troubleshooting First Setup

**"OCI runtime error" or Docker won't start?**
→ Your memory limit is too high. Edit `docker-compose.yml`, reduce `memory:` to 6g or 8g.

**"No videos found" error?**
→ Verify: `ls data/input/` should show `.mp4` files.

**"No LLM backend available"?**
→ This is normal on first run. LLM is optional. Use Preview mode: `QUALITY_PROFILE=preview`.

**Still stuck?**
→ See [troubleshooting.md](docs/troubleshooting.md) or open an issue on GitHub.

---

## Working on the Go Worker (Advanced)

Montage AI is migrating to **Go for distributed concurrency**. Python stays for LLM/analysis.

### Python Developer? This is for you:

You don't need to write Go code. The worker remains compatible with Python APIs. But if interested:

### Building & Testing the Go Worker

```bash
cd go

# Prerequisites
go version  # Should be 1.22+

# Build
go mod download
go build -o montage-worker ./cmd/worker

# Test locally (requires Redis)
REDIS_HOST=localhost ./montage-worker

# Deploy to Kubernetes
./build-and-push.sh
kubectl apply -f ../deploy/k3s/overlays/cluster/worker-go-canary.yaml
```

### Go Worker Architecture

- **`cmd/worker/main.go`** — Entry point + signal handling
- **`pkg/worker/pool.go`** — Goroutine pool + job processing
- **`pkg/redis/client.go`** — Redis queue integration (RQ-compatible)
- **`pkg/ffmpeg/`** — FFmpeg subprocess orchestration
- **`pkg/python/`** — Python subprocess calls (Creative Director)

### Key Concepts

1. **Goroutine pool:** Cheap concurrency (1000 goroutines ≈ 30MB)
2. **Redis streams:** Shared queue with Python worker
3. **Subprocess integration:** Go calls Python for LLM while handling FFmpeg

### Monitoring the Migration

```bash
# Watch Go worker status
kubectl logs -n montage-ai -l app.kubernetes.io/component=worker-go -f

# Compare resource usage (Python vs. Go)
kubectl top pods -n montage-ai

# Queue length
kubectl exec -n montage-ai redis-0 -- redis-cli LLEN "rq:queue:default"
```

### Further Reading

- [Migration plan](go/MIGRATION.md) — Phase-by-phase rollout
- [Go worker README](go/README.md) — Architecture deep dive
- [Performance targets](go/README.md#performance-targets) — Expected improvements

---

## Making Changes

### 1. Create a branch

```bash
git checkout -b feature/my-cool-thing
# or
git checkout -b fix/that-annoying-bug
```

### 2. Do your thing

- Write focused code with small, testable commits
- Add unit tests and small fixtures in `test_data/` for rendering-related changes
- Update docs when behavior or configuration changes

### 3. Test it

```bash
# Quick test
./montage-ai.sh preview

# Full test suite
./scripts/ci.sh

# If you changed K8s manifests
make -C deploy/k3s validate
```

### 4. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/) style and write clear messages:

```bash
git commit -m "feat: add support for vertical videos"
```

### 5. Update CHANGELOG

Add your change to `CHANGELOG.md` under `[Unreleased]` with a short note.

### 6. Publish docs (optional)

Use the local publish script to publish documentation; do not use GitHub Actions:

- `./scripts/publish-docs.sh` (uses `gh` CLI)
- **Do not use GitHub Actions to publish docs or run CI.** Prefer local `gh` CLI or a vendor-neutral CI.
### 7. Open a PR

Push your branch and open a pull request. Include:
- `./scripts/ci-local.sh` output attached to the PR
- Output of `./scripts/check-hardcoded-registries.sh` (ensure no accidental literals)
- A short note on any increased CI cost or long-running tests

We will review and merge after checks pass. Please address review comments promptly.

---

## Adding a New Style

Styles are just JSON files. Super easy:

1. Create `src/montage_ai/styles/your_style.json`:

```json
{
  "id": "your_style",
  "name": "Your Style",
  "description": "What it does",
  "params": {
    "style": {"name": "your_style", "mood": "chill"},
    "pacing": {"speed": "medium", "variation": "moderate"},
    "transitions": {"type": "crossfade"},
    "effects": {"color_grading": "warm"}
  }
}
```

2. Test it:

```bash
./montage-ai.sh run your_style
```

3. Add it to `docs/features.md`

---

## Code Style

### Python

- Type hints and docstrings for public functions.
- Follow PEP 8.

```python
def process_clip(clip_path: str, style: str = "dynamic") -> VideoClip:
    """Process a video clip with the specified style."""
    ...
```

### YAML

- 2-space indentation
- Comment non-obvious choices

### Programming style & AI assistants

- **Config-first**: Do NOT hardcode config values (IPs, registry URLs, paths, resource limits). Add settings to `deploy/k3s/config-global.yaml` or `config.Settings`.
- Run `./scripts/check-hardcoded-registries.sh` before committing. Install the pre-push hook to enforce hardcoded-registry checks *and* to block accidental `.github/workflows/` pushes:

  ```bash
  cp scripts/hooks/pre-push.sample .git/hooks/pre-push && chmod +x .git/hooks/pre-push
  ```

  The hook runs both the registry scan and a commit scan to prevent introducing GitHub Actions workflows (this repository enforces a no-Actions policy).
- Run `./scripts/ci-local.sh` and attach the logs to PRs.
- When using AI assistants (VS Code Copilot, OpenAI Codex), follow `.github/copilot-instructions.md` and `docs/llm-agents.md`. Our agent persona: **Senior Creative Technologist** — prioritize stability, make small, well-tested changes, and document reasoning in PRs.

---

## Testing

### Local

```bash
./scripts/ci.sh     # Run tests
pytest tests/ -q    # Optional: quick pass
```

### Kubernetes

```bash
make -C deploy/k3s validate           # Check manifests
make -C deploy/k3s deploy-cluster  # Deploy to cluster
kubectl logs -n montage-ai -l app.kubernetes.io/name=montage-ai -f
```
Detailed autoscale smoke workflows are documented in the private docs set.

---

## Questions?

- [Open an issue](https://github.com/mfahsold/montage-ai/issues)
- Check [existing discussions](https://github.com/mfahsold/montage-ai/discussions)

---

## Licensing & Legal

### Your Contributions

By contributing, you agree your code will be licensed under the same [PolyForm Noncommercial](LICENSE) license as the project.

### Third-Party Dependencies

When adding new dependencies, please ensure:

1. **License compatibility**: Check the dependency's license is compatible with our project
2. **Document the license**: Add an entry to [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
3. **LGPL compliance**: If the dependency is LGPL (like FFmpeg), ensure we're using dynamic linking

**Acceptable licenses:**
- MIT, BSD, Apache 2.0, ISC — ✅ Always OK
- LGPL — ✅ OK with dynamic linking
- GPL — ❌ Not compatible (copyleft)
- Proprietary — ❌ Not acceptable

**Adding a new dependency:**

1. Add to `requirements.txt` or `pyproject.toml`
2. Add entry to `THIRD_PARTY_LICENSES.md`:
   ```markdown
   ## Package Name
   - **Version**: X.Y.Z
   - **License**: MIT
   - **URL**: https://github.com/org/package
   - **Usage**: Brief description of how we use it
   ```
3. Update `NOTICE` if the license requires attribution

### Model Weights

For AI model weights (Real-ESRGAN, Whisper, etc.):
- Check the model's license before inclusion
- Document in `THIRD_PARTY_LICENSES.md` under "Model Weights"
- Most OpenAI/Meta models are MIT/Apache 2.0 licensed

---

## Where to Contribute

Check the internal backlog via the maintainers (private docs set):

- 🔥 **High priority**: Epics marked "In Progress" or "Planned"
- 🐛 **Bug fixes**: Issues labeled `bug` 
- 📝 **Documentation**: Always welcome
- 🧪 **Tests**: We can always use more coverage

See [docs/STRATEGY.md](docs/STRATEGY.md) for product vision and [docs/roadmap/ROADMAP_2026.md](docs/roadmap/ROADMAP_2026.md) for the development plan.
