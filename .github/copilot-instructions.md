# Copilot Instructions

See [`docs/llm-agents.md`](../docs/llm-agents.md) for agent persona and coding principles.

## Project summary
**Important:** **Do NOT hardcode configuration values.** Always add deployment or runtime settings to `deploy/config.env` or the project's `config.Settings` and reference them via environment variables or centralized config helpers. This helps keep deployments reproducible and secure.

Montage AI is a local-first, post-production assistant: we "polish" existing footage (beat/scene analysis, edit planning, and FFmpeg-based rendering), not generate new video.

## Architecture

```
User Prompt → Creative Director (LLM) → JSON → MontageBuilder → SegmentWriter → FFmpeg → /data/output/
```

| Module | Purpose |
|--------|---------|
| `core/montage_builder.py` | Central orchestrator — pipeline phases (analyze → plan → enhance → render) |
| `editor.py` | CLI entry point — initializes MontageBuilder, handles CLI args |
| `creative_director.py` | LLM-to-JSON translation (Ollama/Gemini/OpenAI) |
| `audio_analysis.py` | Beat detection, tempo extraction, energy profiling (FFmpeg astats/tempo) |
| `scene_analysis.py` | Scene detection, content analysis, visual similarity (SceneDetector class) |
| `segment_writer.py` | Disk-based progressive rendering (SegmentWriter) |
| `auto_reframe.py` | 9:16 smart reframing (AutoReframeEngine, convex optimization) |
| `audio_enhancer.py` | Voice isolation, auto-ducking, loudness normalization |
| `ffmpeg_config.py` | GPU encoder detection, encoding parameters (`get_config()` / `FFmpegConfig`) |
| `broll_planner.py` | Script-to-clip matching via semantic search |
| `footage_manager.py` | Clip pool management with story arc awareness |
| `transcriber.py` | Whisper transcription via cgpu |
| `cgpu_upscaler.py` | AI upscaling via cloud GPU |

## Essentials & quick refs
- `src/montage_ai/core/montage_builder.py` — orchestration
- `src/montage_ai/audio_analysis.py` — beat detection, energy analysis
- `src/montage_ai/scene_analysis.py` — scene detection (`SceneDetector`)
- `src/montage_ai/ffmpeg_config.py` — use `get_config()` / `FFmpegConfig`
- `src/montage_ai/segment_writer.py` — disk-based segment writing
- `src/montage_ai/auto_reframe.py`, `src/montage_ai/audio_enhancer.py`
- `src/montage_ai/web_ui/` — Flask backend & templates
- Read: `docs/llm-agents.md`, `docs/architecture.md`, `docs/cgpu-setup.md`

## Common commands
- `make ci-local` — run full local CI (attach output to PR)
- `make test-unit` / `pytest tests/<target>.py`
- `./montage-ai.sh run [STYLE]` (use `QUALITY_PROFILE=preview` for preview)
- `./montage-ai.sh web`

## Conventions & CI rules
- **Never hardcode configuration values.** Add new settings to `deploy/config.env` or the project `config.Settings`, and reference them from code via environment variables or centralized config helpers. Use `deploy/config.env` for deployment values (registry, namespace, storage, resource limits) and `src/*/config` or `settings` for runtime defaults.
- Use `get_config()` for FFmpeg args; do not hardcode flags.
- Guard heavy ML imports with `try/except`; add to `requirements.txt`.
- Use `ClipMetadata` for clip state and `pathlib` for paths (`/data/...`).
- Logging: use `logger.info()`; avoid `tqdm` in CI logs.
- **Do not use GitHub Actions.** Run CI locally with `./scripts/ci.sh` or use a vendor-neutral CI (Jenkins/Drone/Buildkite). Any GitHub Actions workflows are deprecated and will be removed; local CI is the canonical source for running tests and smoke checks.

> Tip: Run `./scripts/check-hardcoded-registries.sh` and a quick grep for literal IPs/paths before committing; the pre-push hook also checks for new hardcoded values.

## Tests & validation
- Add small fixtures in `test_data/` for rendering tests.
- Reproduce failing tests locally: `pytest -q` and `make ci-local`.
- Validate `SegmentWriter` behaviour after rendering changes.

## Examples
- FFmpeg config:
```python
from montage_ai.ffmpeg_config import get_config
cfg = get_config()
params = cfg.get_preview_video_params()
```
- Audio analysis:
```python
from montage_ai.audio_analysis import get_beat_times, calculate_dynamic_cut_length
beat_info = get_beat_times("/data/music/track.mp3")
```
- Scene detection:
```python
from montage_ai.scene_analysis import SceneDetector, detect_scenes
detector = SceneDetector()
scenes = detector.detect(video_path)
```
- Preview run:
```bash
QUALITY_PROFILE=preview ./montage-ai.sh run dynamic
```

For more detail, read `docs/llm-agents.md` and `docs/architecture.md`.


## Styles & environment
- Styles: `dynamic`, `hitchcock`, `mtv`, `action`, `documentary`, `minimalist`, `wes_anderson`
- Environment examples:
```
CUT_STYLE: dynamic
CGPU_ENABLED: true|false
STABILIZE: true|false
UPSCALE: true|false
```

## Project-specific conventions (brief)
- Use `get_config()` for FFmpeg args; avoid hardcoding ffmpeg flags.
- Guard heavy ML imports with `try/except` and document optional deps in `requirements.txt`.
- Prefer `ClipMetadata` for clip state and `pathlib` for path handling (`/data/...`).
- Logging: `logger.info()` over `print()`; avoid `tqdm` in CI logs.
- Rendering: write segments to disk and concatenate—avoid holding full video in RAM.

## Data paths
`/data/input/`, `/data/music/`, `/data/output/`
