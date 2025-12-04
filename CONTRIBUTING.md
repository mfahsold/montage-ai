# Contributing

Hey! Thanks for thinking about contributing. Here's how to get started.

---

## Quick Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/montage-ai.git
cd montage-ai

# Build
make build

# Run tests
make test
```

---

## Making Changes

### 1. Create a branch

```bash
git checkout -b feature/my-cool-thing
# or
git checkout -b fix/that-annoying-bug
```

### 2. Do your thing

- Write some code
- Add tests if it's a new feature
- Update docs if needed

### 3. Test it

```bash
# Quick test
./montage-ai.sh preview

# Full test suite
make test

# If you changed K8s manifests
make validate
```

### 4. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/) style:

```bash
git commit -m "feat: add support for vertical videos"
git commit -m "fix: beat detection crash on short clips"
git commit -m "docs: update configuration examples"
```

### 5. Update CHANGELOG

Add your change to `CHANGELOG.md` under `[Unreleased]`:

```markdown
## [Unreleased]

### Added
- Support for vertical videos
```

### 6. Open a PR

Push your branch and open a pull request. We'll review it as soon as we can.

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

Nothing fancy:

- Type hints are nice
- Docstrings for public functions
- PEP 8 (your editor probably handles this)

```python
def process_clip(clip_path: str, style: str = "dynamic") -> VideoClip:
    """Process a video clip with the specified style."""
    ...
```

### YAML

- 2-space indentation
- Comments for anything non-obvious

---

## Testing

### Local

```bash
make build          # Build image
make test           # Run tests
make shell          # Debug inside container
```

### Kubernetes

```bash
make validate       # Check manifests
make deploy         # Deploy to cluster
make logs           # Watch job logs
```

---

## Questions?

- [Open an issue](https://github.com/mfahsold/montage-ai/issues)
- Check [existing discussions](https://github.com/mfahsold/montage-ai/discussions)

---

## License Note

By contributing, you agree your code will be licensed under the same [PolyForm Noncommercial](LICENSE) license as the project.
