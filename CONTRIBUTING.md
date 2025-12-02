# Contributing to Montage AI

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/montage-ai.git
cd montage-ai

# Build and test
make build
make test
```

## Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow existing code style
- Add tests for new features
- Update documentation

### 3. Test Your Changes

```bash
# Local Docker test
make test-local

# Kubernetes manifest validation
make validate

# Full test suite
make test
```

### 4. Commit with Clear Messages

```bash
git commit -m "feat: add new style template for music videos"
# or
git commit -m "fix: resolve beat detection timing issue"
```

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Tests
- `chore:` Maintenance

### 5. Update CHANGELOG

Add your changes to `CHANGELOG.md` under `[Unreleased]`:

```markdown
## [Unreleased]

### Added
- Your new feature description

### Fixed
- Your bug fix description
```

### 6. Submit Pull Request

- Push your branch
- Open a PR against `main`
- Fill out the PR template
- Wait for review

## Code Style

### Python

- Use type hints
- Docstrings for public functions
- Follow PEP 8

```python
def process_clip(clip_path: str, style: str = "dynamic") -> VideoClip:
    """Process a video clip with the specified style.
    
    Args:
        clip_path: Path to the video file.
        style: Processing style name.
        
    Returns:
        Processed VideoClip object.
    """
    pass
```

### YAML (Kubernetes)

- Use 2-space indentation
- Include comments for non-obvious settings
- Follow Kubernetes naming conventions

## Testing

### Local Testing

```bash
# Build image
make build

# Run with test footage
./montage-ai.sh preview

# Interactive debugging
make shell
```

### Kubernetes Testing

```bash
# Validate manifests
make validate

# Dry-run deployment
kubectl apply -k deploy/k3s/base/ --dry-run=client

# Test on local cluster
make deploy
make job
make logs
```

## Adding a New Style

1. Create JSON in `src/montage_ai/styles/`:

```json
{
  "id": "your_style",
  "name": "Your Style",
  "description": "Brief description",
  "params": {
    "style": {"name": "your_style", "mood": "energetic"},
    "pacing": {"speed": "fast", "variation": "high"},
    "transitions": {"type": "hard_cuts"},
    "effects": {"color_grading": "neutral"}
  }
}
```

2. Test with:

```bash
./montage-ai.sh run your_style
```

3. Add to documentation in `docs/features.md`

## Release Process

Releases are managed by maintainers:

```bash
# Create release
make release VERSION=v1.0.0
```

This builds multi-arch images and pushes to GHCR.

## Getting Help

- [Open an issue](https://github.com/mfahsold/montage-ai/issues)
- Check existing [discussions](https://github.com/mfahsold/montage-ai/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
