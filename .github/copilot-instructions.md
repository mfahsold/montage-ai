# Copilot Instructions

See [`docs/llm-agents.md`](../docs/llm-agents.md) for agent persona and coding principles.

## Project: Montage AI

AI post-production assistant - "We do not generate pixels; we polish them."

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

## Commands

```bash
./montage-ai.sh run [STYLE]    # Create montage
./montage-ai.sh web            # Web UI
make test                      # Run tests
```

## Styles

`dynamic`, `hitchcock`, `mtv`, `action`, `documentary`, `minimalist`, `wes_anderson`

## Environment

```yaml
CUT_STYLE: dynamic
CREATIVE_PROMPT: "Natural language instructions"
CGPU_ENABLED: true|false
STABILIZE: true|false
UPSCALE: true|false
```

## Data Paths (Container)

```
/data/input/   # Source clips
/data/music/   # Audio tracks
/data/output/  # Generated videos
```
