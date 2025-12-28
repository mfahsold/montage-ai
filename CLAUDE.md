# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Agent Guidelines:** See [`docs/llm-agents.md`](docs/llm-agents.md) for persona, coding principles, and DRY/KISS patterns shared across all AI assistants.

## Project Overview

Montage AI is an **AI Post-Production Assistant** that enhances, organizes, cuts, and polishes existing footage. The philosophy: "We do not generate pixels; we polish them."

**Core Pillars:**
- **AI Editing & Storytelling**: Intelligent cutting based on beat, mood, and narrative arc
- **Enhancement & Restoration**: Upscaling, stabilization, color grading via Cloud GPU
- **Professional Workflow**: Subtitles, B-roll planning, export to NLEs (DaVinci Resolve, Premiere)
- **Cloud GPU Integration**: Heavy compute (upscaling, transcoding, Whisper) offloaded to `cgpu`

## Common Commands

### Development

```bash
make build          # Build Docker image (local arch)
make test           # Run all tests (validate + test-local + test-unit)
make test-unit      # Run pytest unit tests only
make shell          # Interactive shell in container
```

### Running the Application

```bash
./montage-ai.sh run [STYLE]       # Create montage (styles: dynamic, hitchcock, mtv, action, documentary, minimalist, wes_anderson)
./montage-ai.sh preview           # Fast preview render
./montage-ai.sh hq [STYLE]        # High-quality render with stabilization + upscaling
./montage-ai.sh web               # Start Web UI (http://localhost:8080)
./montage-ai.sh list              # List available styles
```

### Testing

```bash
pytest tests/ -v                  # Run unit tests
pytest tests/test_editor_basic.py -v  # Run single test file
make validate                     # Validate Kubernetes manifests
```

### Kubernetes Deployment

```bash
make deploy                       # Deploy base resources
make deploy-prod                  # Production overlay (AMD GPU)
make deploy-dev                   # Dev overlay (fast preview)
make logs                         # View job logs
```

## Architecture

### Three-Stage Pipeline

```
1. Ingest & Analysis → 2. Creative Direction & Edit → 3. Finishing & Export
       (Local+Cloud)              (Local LLM)              (Cloud GPU Heavy)
```

**Stage 1 - Ingest & Analysis:**
- `footage_manager.py`: Scans input directories
- `video_agent.py`: Semantic analysis via CLIP/SigLIP embeddings
- `broll_planner.py`: Script-to-clip matching (uses video_agent)
- `transcriber.py`: Whisper transcription via cgpu

**Stage 2 - Creative Direction & Edit:**
- `creative_director.py`: Translates prompts to JSON Edit Decision Lists
- `editor.py`: Assembly, beat detection, scene detection, clip selection

**Stage 3 - Finishing & Export:**
- `cgpu_upscaler.py`: Real-ESRGAN 4x upscaling via cgpu
- `timeline_exporter.py`: OTIO/EDL export for NLEs
- FFmpeg: Stabilization, color grading, final render

### Core Modules (`src/montage_ai/`)

| Module | Purpose |
|--------|---------|
| `editor.py` | Main orchestrator: beat detection, scene detection, clip assembly, rendering |
| `creative_director.py` | LLM interface: translates natural language prompts to editing parameters |
| `footage_manager.py` | Clip selection with story arc awareness (INTRO→BUILD→CLIMAX→SUSTAIN→OUTRO) |
| `video_agent.py` | Semantic clip analysis for B-roll planning and keyword search |
| `broll_planner.py` | Script-to-clip matching: finds footage for script segments |
| `transcriber.py` | Audio transcription (Whisper) via cgpu for subtitles |
| `style_templates.py` | Loads/validates JSON style presets from `styles/` |
| `segment_writer.py` | Memory-efficient progressive rendering (batches clips to prevent OOM) |
| `ffmpeg_config.py` | Centralized FFmpeg configuration with GPU encoder detection |
| `cgpu_upscaler.py` | Cloud GPU offloading for AI upscaling via cgpu |
| `timeline_exporter.py` | Export to OTIO/XML for DaVinci Resolve, Premiere |

### LLM Backend Priority

The system tries LLM backends in this order:
1. OpenAI-compatible API (`OPENAI_API_BASE`)
2. Google AI (`GOOGLE_API_KEY`)
3. cgpu/Gemini (`CGPU_ENABLED=true`)
4. Ollama (`OLLAMA_HOST`)

### Key Data Classes (footage_manager.py)

- `FootageClip`: Clip metadata (path, duration, energy, scene type)
- `UsageStatus`: UNUSED, USED, RESERVED
- `SceneType`: ESTABLISHING, ACTION, DETAIL, PORTRAIT, SCENIC
- `StoryPhase`: INTRO, BUILD, CLIMAX, SUSTAIN, OUTRO
- `StoryArcController`: Maps timeline position to required energy/scene type

## Configuration

Key environment variables (see `docs/configuration.md` for full reference):

```bash
CUT_STYLE=dynamic              # Editing style preset
CREATIVE_PROMPT="..."          # Natural language override
CGPU_ENABLED=true              # Enable cloud GPU via cgpu
FFMPEG_HWACCEL=auto            # GPU encoding: auto, nvenc, vaapi, qsv, none
STABILIZE=true                 # Enable video stabilization
UPSCALE=true                   # Enable AI upscaling
```

## Adding a New Style

Create `src/montage_ai/styles/your_style.json`:

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

Test with: `./montage-ai.sh run your_style`

## Commit Style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add support for vertical videos"
git commit -m "fix: beat detection crash on short clips"
git commit -m "docs: update configuration examples"
```

Update `CHANGELOG.md` under `[Unreleased]` when making changes.
