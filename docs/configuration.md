# Configuration Reference

Complete reference for all environment variables and settings.

---

## Core Settings

| Variable          | Default       | Description                                                                         |
| ----------------- | ------------- | ----------------------------------------------------------------------------------- |
| `CUT_STYLE`       | `dynamic`     | Editing style: `dynamic`, `hitchcock`, `mtv`, `action`, `documentary`, `minimalist`, `viral`, `wes_anderson` |
| `CREATIVE_PROMPT` | *(empty)*     | Natural language editing instructions (overrides `CUT_STYLE`)                       |
| `NUM_VARIANTS`    | `1`           | Number of output variants to generate                                               |
| `JOB_ID`          | *(timestamp)* | Unique identifier for parallel runs                                                 |
| `JOB_ID_STRATEGY` | `timestamp`   | Job ID mode: `timestamp` (default) or `hash` (deterministic from inputs/settings)   |
| `TARGET_DURATION` | `0`           | Target video duration in seconds (`0` = use full music length)                      |
| `MUSIC_START`     | `0`           | Music start time in seconds (trim beginning)                                        |
| `MUSIC_END`       | `0`           | Music end time in seconds (`0` = use full track, auto-derived from `TARGET_DURATION`) |

---

## Creative Loop (Agentic Refinement)

The Creative Loop enables iterative LLM-powered refinement of montage quality.

| Variable                       | Default | Description                                        |
| ------------------------------ | ------- | -------------------------------------------------- |
| `CREATIVE_LOOP`                | `false` | Enable agentic creative feedback loop              |
| `CREATIVE_LOOP_MAX_ITERATIONS` | `3`     | Maximum refinement iterations before auto-approval |

**How it works:**
1. First cut is built with initial editing instructions
2. LLM evaluates the cut (pacing, variety, energy, transitions)
3. If score < 0.8, adjustments are suggested and applied
4. Process repeats until approved or max iterations reached

**Example:**
```bash
CREATIVE_LOOP=true \
CREATIVE_LOOP_MAX_ITERATIONS=3 \
./montage-ai.sh run hitchcock
```

---

## AI / LLM Settings

LLM backends are used in **priority order**: OpenAI-compatible > Google AI > cgpu > Ollama

| Variable      | Default | Description                         |
| ------------- | ------- | ----------------------------------- |
| `LLM_TIMEOUT` | `60`    | Request timeout in seconds for LLMs |

### OpenAI-Compatible API (Recommended for Kubernetes)

For KubeAI, vLLM, LocalAI, or any OpenAI-compatible endpoint:

| Variable              | Default      | Description                                                      |
| --------------------- | ------------ | ---------------------------------------------------------------- |
| `OPENAI_API_BASE`     | *(empty)*    | API base URL (e.g., `http://kubeai.svc.local/openai/v1`)         |
| `OPENAI_API_KEY`      | `not-needed` | API key (KubeAI ignores this)                                    |
| `OPENAI_MODEL`        | *(empty)*    | Creative Director model (e.g., `gemma3-4b`, `qwen2-5-32b`)       |
| `OPENAI_VISION_MODEL` | *(empty)*    | Vision model for scene analysis (e.g., `moondream2`, `llava-7b`) |

**Example (KubeAI cluster):**

```bash
OPENAI_API_BASE=http://kubeai.kubeai-system.svc.cluster.local/openai/v1 \
OPENAI_MODEL=gemma3-4b \
OPENAI_VISION_MODEL=moondream2 \
./montage-ai.sh run
```

**Model Selection:**
- **Creative Director**: Use small, fast models (gemma3-4b, llama3-8b)
- **Vision Analysis**: Use vision models (moondream2, llava-7b)
- **Complex Tasks**: Use larger models on-demand (qwen2-5-32b, llama3-70b)

### Ollama (Local LLM)

| Variable           | Default                             | Description                    |
| ------------------ | ----------------------------------- | ------------------------------ |
| `OLLAMA_HOST`      | `http://host.docker.internal:11434` | Ollama API endpoint            |
| `OLLAMA_MODEL`     | `llava`                             | Model for scene analysis       |
| `DIRECTOR_MODEL`   | `llama3.1:70b`                      | Model for Creative Director    |
| `ENABLE_AI_FILTER` | `false`                             | Enable AI-based clip filtering |

### Google AI (Direct API)

| Variable          | Default            | Description                |
| ----------------- | ------------------ | -------------------------- |
| `GOOGLE_API_KEY`  | *(empty)*          | Google AI API key          |
| `GOOGLE_AI_MODEL` | `gemini-2.0-flash` | Gemini model for LLM tasks |

### cgpu / Gemini (Cloud LLM via cgpu serve)

| Variable           | Default                | Description                                      |
| ------------------ | ---------------------- | ------------------------------------------------ |
| `CGPU_ENABLED`     | `false`                | Enable cgpu/Gemini backend for Creative Director |
| `CGPU_HOST`        | `host.docker.internal` | cgpu serve hostname                              |
| `CGPU_PORT`        | `8080`                 | cgpu serve port                                  |
| `CGPU_MODEL`       | `gemini-2.0-flash`     | Gemini model to use                              |
| `CGPU_GPU_ENABLED` | `false`                | Enable cloud GPU for upscaling                   |
| `CGPU_TIMEOUT`     | `1200`                 | Cloud operation timeout (seconds)                |
| `CGPU_MAX_CONCURRENCY` | `1`               | Max parallel cgpu commands (advanced)            |

---

## Cloud & Pro Features (Monetization)

Settings for Montage Cloud integration (Pro tier).

| Variable                 | Default                   | Description                                      |
| ------------------------ | ------------------------- | ------------------------------------------------ |
| `MONTAGE_CLOUD_ENABLED`  | `false`                   | Enable Montage Cloud integration                 |
| `MONTAGE_CLOUD_API_KEY`  | *(empty)*                 | API Key for Pro features                         |
| `MONTAGE_CLOUD_ENDPOINT` | `https://api.montage.ai`  | API Endpoint                                     |
| `TRACK_USAGE`            | `true`                    | Enable usage tracking for billing                |

---

## Duration Controls

Fine-tune the length and timing of your montage.

| Variable          | Default | Description                                      |
| ----------------- | ------- | ------------------------------------------------ |
| `TARGET_DURATION` | `0`     | Target length in seconds (0 = auto/music length) |
| `MUSIC_START`     | `0`     | Start time of music track (seconds)              |
| `MUSIC_END`       | `0`     | End time of music track (0 = end of file)        |

---

## Visual Enhancement

| Variable          | Default | Description                                   |
| ----------------- | ------- | --------------------------------------------- |
| `STABILIZE`       | `false` | Enable video stabilization                    |
| `UPSCALE`         | `false` | Enable AI upscaling (Real-ESRGAN)             |
| `ENHANCE`         | `true`  | Enable color/sharpness enhancement            |
| `PRESERVE_ASPECT` | `false` | Letterbox instead of crop when aspect differs |
| `DEEP_ANALYSIS`   | `false` | Enable deep footage analysis (experimental)   |
| `COLORLEVELS`     | `true`  | Apply broadcast-safe levels (16-235)          |
| `LUMA_NORMALIZE`  | `true`  | Normalize luma for consistency across clips   |

### Upscaling Controls

| Variable               | Default               | Description                                                           |
| ---------------------- | --------------------- | --------------------------------------------------------------------- |
| `UPSCALE_MODEL`        | `realesrgan-x4plus`    | Real-ESRGAN model (`realesrgan-x4plus`, `realesr-animevideov3`, etc.)  |
| `UPSCALE_SCALE`        | `2`                   | Upscale factor (2 or 4)                                               |
| `UPSCALE_FRAME_FORMAT` | `jpg`                 | Frame cache format (`jpg` = faster, `png` = lossless)                 |
| `UPSCALE_TILE_SIZE`    | `512`                 | Tile size for GPU upscaling (lower = safer, slower)                   |
| `UPSCALE_CRF`          | `FINAL_CRF` or `18`    | CRF for upscaled encode (lower = better quality, larger files)        |

### Aspect Ratio Handling

When the input footage has a different aspect ratio than the target output:

| `PRESERVE_ASPECT` | Behavior                                              |
| ----------------- | ----------------------------------------------------- |
| `false` (default) | **Crop to fill** - cuts edges to fill entire frame    |
| `true`            | **Letterbox/Pillarbox** - adds black bars to preserve |

Use `PRESERVE_ASPECT=true` when:

- Horizontal clips should keep full content in vertical (9:16) output
- You want to avoid cutting important content at frame edges

---

## Quick-Start Configurations

### KubeAI Cluster with GPU Encoding

```bash
# .env file or docker-compose environment
OPENAI_API_BASE=http://kubeai.kubeai-system.svc.cluster.local/openai/v1
OPENAI_MODEL=gemma3-4b
OPENAI_VISION_MODEL=moondream2
ENABLE_AI_FILTER=true
FFMPEG_HWACCEL=auto
PRESERVE_ASPECT=false
```

### Local Development (Ollama + CPU)

```bash
OLLAMA_HOST=http://localhost:11434
DIRECTOR_MODEL=llama3.1:8b
OLLAMA_MODEL=llava
FFMPEG_HWACCEL=none
```

### Cloud GPU Processing (cgpu)

```bash
CGPU_ENABLED=true
CGPU_GPU_ENABLED=true
CGPU_MODEL=gemini-2.0-flash
UPSCALE=true
```

---

## Performance / Hardware

| Variable            | Default           | Description                                                       |
| ------------------- | ----------------- | ----------------------------------------------------------------- |
| `USE_GPU`           | `auto`            | GPU mode: `auto`, `vulkan`, `v4l2`, `none`                        |
| `FFMPEG_HWACCEL`    | `auto`            | Video encoding GPU: `auto`, `nvenc`, `vaapi`, `qsv`, `none`       |
| `FFMPEG_THREADS`    | `0`               | FFmpeg thread count (`0` = auto)                                  |
| `FFMPEG_PRESET`     | `medium`          | Encoding speed: `ultrafast`, `fast`, `medium`, `slow`, `veryslow` |
| `PARALLEL_ENHANCE`  | `true`            | Enable parallel clip enhancement                                  |
| `MAX_PARALLEL_JOBS` | *(CPU cores - 2)* | Maximum parallel workers                                          |

### GPU Hardware Acceleration

`FFMPEG_HWACCEL` enables hardware-accelerated video **encoding**:

| Value          | GPU Type          | Notes                                   |
| -------------- | ----------------- | --------------------------------------- |
| `auto`         | Auto-detect       | Uses best available GPU encoder         |
| `nvenc`        | NVIDIA            | Requires NVIDIA GPU with NVENC support  |
| `vaapi`        | AMD/Intel (Linux) | Requires `/dev/dri` access in container |
| `qsv`          | Intel QuickSync   | Requires Intel GPU with QSV support     |
| `videotoolbox` | macOS             | Apple Silicon or Intel Mac with GPU     |
| `none`         | CPU only          | Software encoding (libx264/libx265)     |

**Quality parameters are automatically adjusted per encoder:**
- **NVENC:** `-cq` (constant quality, similar to CRF)
- **VAAPI:** `-qp` (quantization parameter)
- **QSV:** `-global_quality` (quality scale)
- **CPU/VideoToolbox:** `-crf` (constant rate factor)

**Performance comparison (1080p encoding):**

| Encoder    | Speed       | Quality   | Power Usage |
| ---------- | ----------- | --------- | ----------- |
| CPU (x264) | 1x baseline | Excellent | High        |
| NVENC      | 5-10x       | Very Good | Low         |
| VAAPI      | 3-6x        | Good      | Low         |
| QSV        | 4-8x        | Very Good | Low         |

**Current Limitations:**
- Hardware-accelerated **decoding** is not supported (MoviePy limitation)
- Only encoding benefits from GPU acceleration
- Decoding remains CPU-based for maximum compatibility

---

## Memory Management

Settings for large projects and constrained hardware.

| Variable           | Default | Description                                                   |
| ------------------ | ------- | ------------------------------------------------------------- |
| `MEMORY_LIMIT_GB`  | `16`    | Docker container memory limit (GB) - match docker-compose.yml |
| `MAX_CLIPS_IN_RAM` | `50`    | Maximum clips to keep in RAM simultaneously                   |
| `AUTO_CLEANUP`     | `true`  | Automatically delete temp files after rendering               |

**Hardware-specific recommendations:**

- **8-16GB RAM:** `MEMORY_LIMIT_GB=12`, `MAX_CLIPS_IN_RAM=30`, `CGPU_GPU_ENABLED=true`
- **16-32GB RAM:** Default settings are optimal
- **32GB+ RAM:** `MEMORY_LIMIT_GB=48`, `MAX_CLIPS_IN_RAM=100`

See [archive/OPERATIONS_LOG.md](archive/OPERATIONS_LOG.md) for stability notes.

---

## Idempotency & Timeouts

| Variable                | Default | Description                                                      |
| ----------------------- | ------- | ---------------------------------------------------------------- |
| `SKIP_EXISTING_OUTPUTS` | `true`  | Reuse existing outputs when present                              |
| `FORCE_REPROCESS`       | `false` | Ignore cached outputs and recompute                              |
| `MIN_OUTPUT_BYTES`      | `1024`  | Minimum file size to treat an output as valid                    |

| Variable              | Default | Description                                   |
| --------------------- | ------- | --------------------------------------------- |
| `FFPROBE_TIMEOUT`     | `10`    | Timeout for ffprobe metadata calls (seconds)  |
| `FFMPEG_SHORT_TIMEOUT`| `30`    | Timeout for quick ffmpeg checks (seconds)     |
| `FFMPEG_TIMEOUT`      | `120`   | Timeout for standard ffmpeg operations        |
| `FFMPEG_LONG_TIMEOUT` | `600`   | Timeout for long ffmpeg operations (seconds)  |
| `RENDER_TIMEOUT`      | `3600`  | Timeout for final render/concat (seconds)     |
| `ANALYSIS_TIMEOUT`    | `120`   | Timeout for audio/analysis pipelines          |

**Notes:**
- For deterministic reruns, use `JOB_ID_STRATEGY=hash` with `SKIP_EXISTING_OUTPUTS=true`.
- Set `FORCE_REPROCESS=true` to override cached outputs.

---

## Output / Export

| Variable           | Default | Description                                                                               |
| ------------------ | ------- | ----------------------------------------------------------------------------------------- |
| `EXPORT_TIMELINE`  | `false` | Export OTIO/EDL timeline (experimental)                                                   |
| `GENERATE_PROXIES` | `false` | Generate proxy files for NLE (experimental)                                               |
| `QUALITY_PROFILE`  | `standard` | Preset defaults for CRF/preset/codec/pix_fmt (`preview`, `standard`, `high`, `master`) |
| `OUTPUT_CODEC`     | *auto*  | `libx264` by default; switches to `libx265` when most footage is HEVC (override to force) |
| `OUTPUT_PROFILE`   | *auto*  | FFmpeg profile passed to encoder (usually `high` for H.264, `main` for HEVC)              |
| `OUTPUT_LEVEL`     | *auto*  | Encoder level; raised automatically for 4K/high‑FPS projects                              |

*Resolution and FPS are inferred from the dominant input footage (orientation, median size, and common framerate) to minimize re-encoding and quality loss.*

**Quality profile notes:**
- `preview`: ultrafast + CRF 28
- `standard`: medium + CRF 18
- `high`: slow + CRF 17
- `master`: slow + CRF 16, `libx265` + `yuv420p10le` + `main10` (10-bit output)

---

## Paths (Container)

| Variable     | Default        | Description        |
| ------------ | -------------- | ------------------ |
| `INPUT_DIR`  | `/data/input`  | Source video clips |
| `MUSIC_DIR`  | `/data/music`  | Audio tracks       |
| `ASSETS_DIR` | `/data/assets` | Overlays, logos    |
| `OUTPUT_DIR` | `/data/output` | Generated videos   |

---

## Style Presets

| Variable            | Default      | Description                                 |
| ------------------- | ------------ | ------------------------------------------- |
| `STYLE_PRESET_DIR`  | *(built-in)* | Directory containing `*.json` style presets |
| `STYLE_PRESET_PATH` | *(none)*     | Single JSON file with style preset(s)       |

---

## Debugging

| Variable  | Default | Description                 |
| --------- | ------- | --------------------------- |
| `VERBOSE` | `true`  | Show detailed analysis logs |

---

## Example Configurations

### Fast Preview

```bash
FFMPEG_PRESET=ultrafast \
ENHANCE=false \
STABILIZE=false \
./montage-ai.sh preview
```

### High Quality Render

```bash
FFMPEG_PRESET=slow \
STABILIZE=true \
UPSCALE=true \
CGPU_GPU_ENABLED=true \
./montage-ai.sh hq hitchcock
```

### Cloud-Powered (cgpu)

```bash
CGPU_ENABLED=true \
CGPU_GPU_ENABLED=true \
CREATIVE_PROMPT="Cinematic thriller with slow reveals" \
./montage-ai.sh run
```

### Custom Style with Variants

```bash
CUT_STYLE=mtv \
NUM_VARIANTS=5 \
./montage-ai.sh run
```
