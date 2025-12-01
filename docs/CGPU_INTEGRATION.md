---
title: "cgpu Integration"
summary: "cgpu integration for free cloud GPU and Gemini LLM"
updated: 2025-12-01
status: implemented
---

# cgpu Integration for Montage-AI

## Status: ✅ Implemented

cgpu (github.com/RohanAdwankar/cgpu) is now integrated, providing:
1. **Free LLM API** via `cgpu serve` (Google Gemini proxy)
2. **Free Cloud GPU** via `cgpu run` (Google Colab GPUs for Real-ESRGAN)

## Quick Start

```bash
# Install cgpu (one-time)
npm i -g cgpu

# Install gemini-cli (required for cgpu serve)
# See: https://github.com/google-gemini/gemini-cli

# First-time setup (interactive wizard)
cgpu connect

# Option 1: Use Gemini for Creative Director
./montage-ai.sh run hitchcock --cgpu

# Option 2: Use cloud GPU for upscaling
./montage-ai.sh run hitchcock --cgpu --cgpu-gpu --upscale

# Option 3: Full cgpu mode (LLM + GPU)
./montage-ai.sh hq hitchcock --cgpu --cgpu-gpu
```

## How It Works

cgpu provides two main features:

1. **LLM API** - `cgpu serve` proxies to Google Gemini (used by Creative Director)
2. **Cloud GPU** - `cgpu run` executes code on Google Colab GPUs (used for AI upscaling)

See [Architecture](architecture.md) for full system design.

## Environment Variables

| Variable           | Default                | Description                              |
| ------------------ | ---------------------- | ---------------------------------------- |
| `CGPU_ENABLED`     | `false`                | Enable cgpu/Gemini for Creative Director |
| `CGPU_HOST`        | `host.docker.internal` | cgpu serve host                          |
| `CGPU_PORT`        | `8080`                 | cgpu serve port                          |
| `CGPU_MODEL`       | `gemini-2.0-flash`     | Gemini model to use                      |
| `CGPU_GPU_ENABLED` | `false`                | Enable cloud GPU for upscaling           |
| `CGPU_TIMEOUT`     | `600`                  | Cloud GPU operation timeout (seconds)    |

## CLI Commands

```bash
# cgpu management
./montage-ai.sh cgpu-start    # Start cgpu serve (Gemini API)
./montage-ai.sh cgpu-stop     # Stop cgpu serve
./montage-ai.sh cgpu-status   # Check cgpu installation/status

# Running with cgpu
./montage-ai.sh run --cgpu              # Use Gemini LLM
./montage-ai.sh run --cgpu-gpu          # Use cloud GPU for upscaling
./montage-ai.sh run --cgpu --cgpu-gpu   # Both features
```

## Files Modified

| File                                  | Changes                              |
| ------------------------------------- | ------------------------------------ |
| `requirements.txt`                    | Added `openai>=1.0.0`                |
| `src/montage_ai/creative_director.py` | Added cgpu/Gemini backend            |
| `src/montage_ai/cgpu_upscaler.py`     | **NEW** - Cloud GPU upscaling module |
| `src/montage_ai/editor.py`            | Integrated cgpu upscaler             |
| `docker-compose.yml`                  | Added cgpu environment variables     |
| `montage-ai.sh`                       | Added cgpu commands and flags        |

## Implementation Details

### Creative Director (Gemini LLM)

When `CGPU_ENABLED=true`:

1. `cgpu serve` runs locally, proxying to Google Gemini
2. Creative Director uses OpenAI-compatible API at `http://localhost:8080/v1`
3. Falls back to Ollama if cgpu is unavailable

```python
# In creative_director.py
if self.use_cgpu and self.cgpu_client:
    return self._query_cgpu(user_prompt)  # Uses Gemini
else:
    return self._query_ollama(user_prompt)  # Local fallback
```

### Cloud GPU Upscaling

When `CGPU_GPU_ENABLED=true` and `UPSCALE=true`:

1. Video frames extracted locally
2. Frames uploaded to Google Colab via cgpu
3. Real-ESRGAN runs on T4/A100 GPU
4. Upscaled frames downloaded
5. Video reassembled locally

Priority order:

1. cgpu Cloud GPU (if enabled)
2. Local Vulkan GPU (if available)
3. FFmpeg Lanczos (CPU fallback)

## Limitations

1. **Internet required** - Both features need connectivity
2. **Rate limits** - Google Gemini has usage limits
3. **Colab quotas** - Cloud GPU usage is limited
4. **Latency** - Cloud operations add network overhead
5. **gemini-cli required** - Must install separately for `cgpu serve`

## Fallback Behavior

The system gracefully falls back when cgpu is unavailable:

- **LLM**: Falls back to Ollama (local)
- **Upscaling**: Falls back to local Vulkan GPU or FFmpeg

## Testing

```bash
# Check cgpu status
./montage-ai.sh cgpu-status

# Test Gemini integration
CGPU_ENABLED=true python -c "
from montage_ai.creative_director import CreativeDirector
d = CreativeDirector()
print(d.interpret_prompt('Edit like Hitchcock'))
"

# Test cloud GPU
CGPU_GPU_ENABLED=true python -c "
from montage_ai.cgpu_upscaler import is_cgpu_available, check_cgpu_gpu
print(f'Available: {is_cgpu_available()}')
print(f'GPU: {check_cgpu_gpu()}')
"
```

## References

- cgpu GitHub: <https://github.com/RohanAdwankar/cgpu>
- Gemini CLI: <https://github.com/google-gemini/gemini-cli>
- OpenAI Python Client: <https://github.com/openai/openai-python>
    ports:
