# AI Agent Guidelines for Montage AI

This file provides essential context for AI coding agents working on this codebase. It should be read in conjunction with `docs/llm-agents.md`.

---

## Project Overview

**Montage AI** is a local-first, post-production video assistant. We "polish" existing footage (beat/scene analysis, edit planning, FFmpeg-based rendering), not generate new video.

**Architecture:**
```
User Prompt → Creative Director (LLM) → JSON → MontageBuilder → SegmentWriter → FFmpeg → /data/output/
```

---

## Build, Lint, and Test Commands

### Running Tests

```bash
# Full local CI (preferred over GitHub Actions)
./scripts/ci-local.sh

# Run all unit tests
pytest -m "not integration and not slow and not scale"

# Run specific test file
pytest tests/test_auto_reframe.py

# Run specific test by name
pytest tests/test_auto_reframe.py::TestAutoReframe::test_calculate_crop_dims

# Run tests with verbose output
pytest -v tests/test_auto_reframe.py

# Run with pytest flags
pytest -q --maxfail=1 --disable-warnings
```

### Code Quality

```bash
# Syntax check (dry run)
DRY_RUN=1 ./scripts/ci-local.sh

# Dead code detection
vulture src --min-confidence 50

# Run full CI locally
make ci-local
```

### Installation

```bash
# Install with uv (recommended)
uv sync --dev --extra test

# Or with pip
pip install -e ".[test]"
```

---

## Code Style Guidelines

### General Principles

- **Pythonic**: Write idiomatic Python. Avoid over-abstraction.
- **Typed**: Use type hints where beneficial. Prefer explicit over implicit.
- **Documented**: Document public APIs. Complex logic needs comments.
- **Concise**: Keep responses and code brief unless detail is required.

### Imports

```python
# Standard library first, then third-party, then local
from __future__ import annotations
import os
import subprocess
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from montage_ai.config import get_settings
from montage_ai.logger import logger

# Use TYPE_CHECKING for imports only needed for type hints
if TYPE_CHECKING:
    from montage_ai.core import hardware
```

- Use absolute imports from the package: `from montage_ai.ffmpeg_config import get_config`
- Group imports: stdlib, third-party, local
- Avoid wildcard imports (`from x import *`)

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `audio_analysis.py` |
| Classes | PascalCase | `MontageBuilder` |
| Functions | snake_case | `get_beat_times()` |
| Constants | UPPER_SNAKE | `STANDARD_FPS = 30` |
| Variables | snake_case | `clip_metadata` |

### Types

- Use `typing` module for complex types: `List[str]`, `Optional[Dict]`, `Tuple[int, int]`
- Use `from __future__ import annotations` for forward references
- Use `TYPE_CHECKING` guard for imports only needed by type checkers

### Paths and Files

- Use `pathlib.Path` for path handling
- Use `/data/...` paths (Docker-compatible)
- Never hardcode absolute paths or registry URLs

### Configuration

- **NEVER hardcode configuration values** - Always use environment variables or centralized config
- Use `deploy/config.env` for deployment settings (registry, namespace, storage, resource limits)
- Use `src/montage_ai/config.py` for runtime defaults via `get_settings()` or `config.Settings`
- Use `FFmpegConfig` singleton via `get_config()` for FFmpeg arguments

```python
# Correct: use centralized config
from montage_ai.ffmpeg_config import get_config
cfg = get_config()
params = cfg.get_preview_video_params()

# Wrong: hardcoded flags
params = ["-c:v", "libx264", "-preset", "fast"]  # Don't do this
```

### Error Handling

- Use custom exceptions from `montage_ai.exceptions` when appropriate
- Guard heavy ML imports with try/except: `try: import mediapipe except ImportError: ...`
- Log errors with appropriate level: `logger.error()` vs `logger.warning()`
- Fail fast with clear error messages

### Logging

- Use `logger.info()`, `logger.warning()`, `logger.error()` - not `print()`
- Avoid `tqdm` in CI logs
- Keep log messages concise and informative

### Testing Patterns

```python
import unittest
from unittest.mock import MagicMock, patch

class TestAutoReframe(unittest.TestCase):
    def setUp(self):
        # Patch heavy imports at module level
        self.mp_patcher = patch('montage_ai.auto_reframe.mp')
        self.mock_mp = self.mp_patcher.start()
        
    def tearDown(self):
        self.mp_patcher.stop()
        
    def test_calculate_crop_dims(self):
        # Test implementation
        pass
```

- Place tests in `tests/` directory
- Mirror source structure: `tests/test_xxx.py` corresponds to `src/montage_ai/xxx.py`
- Use `unittest` or `pytest` fixtures
- Mock heavy dependencies (mediapipe, cv2, etc.)

### Heavy ML Dependencies

Guard optional ML libraries with try/except blocks to keep CLI startup fast:

```python
def get_face_detector():
    try:
        import mediapipe as mp
        return mp.solutions.face_detection
    except ImportError:
        logger.warning("mediapipe not available, using fallback")
        return None
```

---

## Key Modules Reference

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `core/montage_builder.py` | Orchestration | `MontageBuilder` |
| `ffmpeg_config.py` | FFmpeg args | `get_config()`, `FFmpegConfig` |
| `audio_analysis.py` | Beat detection | `get_beat_times()` |
| `scene_analysis.py` | Scene detection | `SceneDetector` |
| `segment_writer.py` | Progressive rendering | `SegmentWriter` |
| `auto_reframe.py` | 9:16 reframing | `AutoReframeEngine` |
| `web_ui/` | Flask backend | `app.py` |

---

## Common Commands

```bash
# Preview run
QUALITY_PROFILE=preview ./montage-ai.sh run dynamic

# Web UI
./montage-ai.sh web

# Styles: dynamic, hitchcock, mtv, action, documentary, minimalist, wes_anderson
```

---

## Data Paths

- Input: `/data/input/`
- Music: `/data/music/`
- Output: `/data/output/`

---

## Pre-Push Validation

Before committing, verify:

1. No hardcoded registries/URLs: `./scripts/check-hardcoded-registries.sh`
2. No secrets in git: `git diff --cached --name-only | grep -E "(\.env|secrets)"`
3. Syntax checks pass: `DRY_RUN=1 ./scripts/ci-local.sh`

---

## Additional Resources

- Full agent guidelines: `docs/llm-agents.md`
- Architecture details: `docs/architecture.md`
- Copilot instructions: `.github/copilot-instructions.md`
