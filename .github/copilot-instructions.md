# Copilot Instructions

See [`docs/llm-agents.md`](../docs/llm-agents.md) for agent persona and coding principles.

## Project summary
**Important:** **Do NOT hardcode configuration values.** Always add deployment or runtime settings to `deploy/config.env` or the project's `config.Settings` and reference them via environment variables or centralized config helpers. This helps keep deployments reproducible and secure.

Montage AI is a local-first, post-production assistant: we "polish" existing footage (beat/scene analysis, edit planning, and FFmpeg-based rendering), not generate new video.

## Architecture

```
User Prompt → Creative Director (LLM) → JSON Instructions → Editor → FFmpeg → Final Video
```

| Module | Purpose |
|--------|---------|
| `editor.py` | Main orchestrator - beat detection, scene detection, clip assembly |
| `creative_director.py` | LLM-to-JSON translation (Ollama/Gemini/OpenAI) |
| `broll_planner.py` | Script-to-clip matching via semantic search |
| `transcriber.py` | Whisper transcription via cgpu |
| `cgpu_upscaler.py` | AI upscaling via cloud GPU |

## Quick architecture
User Prompt → Creative Director (LLM) → JSON → MontageBuilder → SegmentWriter → FFmpeg → /data/output/

## Essentials & quick refs
- `src/montage_ai/core/montage_builder.py` — orchestration
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
