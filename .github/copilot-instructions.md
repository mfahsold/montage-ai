# Montage AI - Copilot Instructions

AI-powered video montage creation with beat-synchronized editing.

## Upcoming: cgpu Integration

**Priority task:** Integrate cgpu (github.com/RohanAdwankar/cgpu) for free cloud GPU and LLM access.

**cgpu provides two capabilities:**

1. **Free Cloud GPU** (`cgpu connect/run`) - Access Google Colab GPUs from terminal for CUDA workloads
   - Use for: Real-ESRGAN upscaling, GPU-accelerated video processing
   - `cgpu run nvidia-smi` - Run commands on cloud GPU
   - `cgpu connect` - Persistent terminal session with GPU

2. **Free LLM API** (`cgpu serve`) - OpenAI-compatible server proxying Google Gemini
   - Use for: Creative Director NLP → editing parameters, clip analysis, story arc generation
   - Requires: [gemini-cli](https://github.com/google-gemini/gemini-cli) installed

**What could move to cgpu cloud GPU:** Real-ESRGAN upscaling (currently CPU-bound in Docker)

**What stays local:** Video stabilization, FFmpeg encoding, beat detection (librosa)

**Key files to modify:**
| File | Change |
|------|--------|
| `requirements.txt` | Add `openai>=1.0.0` |
| `src/montage_ai/creative_director.py` | Add cgpu/Gemini backend option |
| `docker-compose.yml` | Add `cgpu serve` as sidecar service |
| `montage-ai.sh` | Auto-start `cgpu serve`, optional `cgpu run` for upscaling |

**New env vars:** `CGPU_ENABLED`, `CGPU_HOST`, `CGPU_PORT`, `CGPU_MODEL=gemini-2.0-flash`, `CGPU_GPU_ENABLED`

See `docs/CGPU_INTEGRATION.md` for full implementation plan.

## Architecture Overview

```
User Prompt → Creative Director (LLM) → JSON Instructions → Editor → FFmpeg/MoviePy → Final Video
```

**Core flow:** Natural language prompts are interpreted by `creative_director.py` into structured JSON editing instructions that drive `editor.py`.

### Key Components

| Module | Purpose |
|--------|---------|
| `editor.py` | Main orchestrator - beat detection (librosa), scene detection, clip assembly, FFmpeg rendering |
| `creative_director.py` | LLM-to-JSON translation via Ollama (llama3.1:70b) |
| `styles/*.json` | Predefined style presets (overridable via env) |
| `footage_manager.py` | Story arc-aware clip selection, variety scoring, "use once" clip tracking |
| `monitoring.py` | Real-time decision logging and performance metrics |

**Experimental (WIP):** `timeline_exporter.py` (OTIO/EDL), `footage_analyzer.py` (deep visual analysis)

## Development Workflow

```bash
# Build and run (Docker-based)
./montage-ai.sh build
./montage-ai.sh run [STYLE]         # dynamic, hitchcock, mtv, action, documentary, minimalist
./montage-ai.sh preview [STYLE]     # Fast preview
./montage-ai.sh hq [STYLE]          # High quality + stabilization
```

All execution happens in Docker. The `montage-ai.sh` script wraps `docker compose run` with environment variables.

## Data Paths (Container)

```
/data/input/   # Source video clips (mounted read-only)
/data/music/   # Audio tracks
/data/assets/  # Overlays, logos
/data/output/  # Generated videos
```

## Configuration Pattern

All behavior is controlled via environment variables in `docker-compose.yml`:

```yaml
# Style/Creative
CUT_STYLE: dynamic|hitchcock|mtv|...
CREATIVE_PROMPT: "Natural language editing instructions"

# AI
OLLAMA_HOST: http://host.docker.internal:11434
DIRECTOR_MODEL: llama3.1:70b

# Enhancement (optional, slow)
STABILIZE: true|false
UPSCALE: true|false
```

## Style Presets Structure

Style definitions live in JSON (no code changes required). Each file can contain one or many presets:

```json
{
  "id": "example_style",
  "name": "Example Style",
  "description": "One-line intent",
  "params": {
    "style": {"name": "example_style", "mood": "energetic"},
    "pacing": {"speed": "fast", "variation": "moderate"},
    "cinematography": {"prefer_wide_shots": false},
    "transitions": {"type": "hard_cuts"},
    "effects": {"color_grading": "neutral"}
  }
}
```

Override shipped presets by setting `STYLE_PRESET_DIR` (folder of `*.json`) or `STYLE_PRESET_PATH` (single file). Later files override earlier ones when IDs match.

## Footage Manager Concepts

`footage_manager.py` implements professional editing workflows:
- **UsageStatus:** Clips are consumed once (`UNUSED → USED`)
- **StoryPhase:** Timeline position maps to INTRO→BUILD→CLIMAX→SUSTAIN→OUTRO
- **SceneType:** Clips categorized as ESTABLISHING, ACTION, DETAIL, PORTRAIT, SCENIC

## Code Conventions

- **Imports:** Use relative imports within `montage_ai` package (e.g., `from .style_templates import ...`)
- **Feature flags:** Use `try/except ImportError` pattern for optional features:
  ```python
  try:
      from .creative_director import CreativeDirector
      CREATIVE_DIRECTOR_AVAILABLE = True
  except ImportError:
      CREATIVE_DIRECTOR_AVAILABLE = False
  ```
- **Dataclasses:** Preferred for structured data (see `footage_manager.py`, `footage_analyzer.py`)
- **Enums:** Use for fixed categorizations (SceneType, StoryPhase, MoodType)

## Change Documentation

**All code changes MUST be documented in `CHANGELOG.md`** using the following structure:

```markdown
## [Unreleased]

### Added
- New features or capabilities

### Changed
- Changes to existing functionality

### Fixed
- Bug fixes

### Removed
- Removed features or deprecated code

### Technical
- Internal refactoring, dependency updates, build changes
```

**Guidelines:**
- Group related changes under a single bullet point
- Reference affected files: `Updated \`editor.py\` to support...`
- Include context for WHY changes were made, not just WHAT
- Use imperative mood: "Add feature" not "Added feature"
- Link to issues/PRs when applicable: `(#123)`

**Example entry:**
```markdown
### Added
- cgpu cloud GPU support for Real-ESRGAN upscaling (`cgpu_upscaler.py`)
  - Offloads AI upscaling to free Google Colab GPUs via cgpu
  - Falls back to local Vulkan/CPU when cgpu unavailable
  - New env vars: `CGPU_GPU_ENABLED`, `CGPU_TIMEOUT`
```

## Testing & Debugging

- Run with `VERBOSE=true` for detailed decision logging via `monitoring.py`
- Monitor logs show: clip selection reasons, beat alignment, energy scores
- No test framework currently configured - manual testing via Docker

## External Dependencies

- **Ollama:** Required for Creative Director LLM (runs on host, accessed via `host.docker.internal:11434`)
- **FFmpeg:** Video encoding (installed in Docker image)
- **Real-ESRGAN:** Optional AI upscaling (built from source in Dockerfile for ARM64)
