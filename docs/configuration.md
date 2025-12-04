# Configuration Reference

Complete reference for all environment variables and settings.

---

## Core Settings

| Variable          | Default       | Description                                                                         |
| ----------------- | ------------- | ----------------------------------------------------------------------------------- |
| `CUT_STYLE`       | `dynamic`     | Editing style: `dynamic`, `hitchcock`, `mtv`, `action`, `documentary`, `minimalist` |
| `CREATIVE_PROMPT` | *(empty)*     | Natural language editing instructions (overrides `CUT_STYLE`)                       |
| `NUM_VARIANTS`    | `1`           | Number of output variants to generate                                               |
| `JOB_ID`          | *(timestamp)* | Unique identifier for parallel runs                                                 |

---

## AI / LLM Settings

LLM backends are used in **priority order**: OpenAI-compatible > Google AI > cgpu > Ollama

### OpenAI-Compatible API (Recommended for Kubernetes)

For KubeAI, vLLM, LocalAI, or any OpenAI-compatible endpoint:

| Variable          | Default      | Description                                              |
| ----------------- | ------------ | -------------------------------------------------------- |
| `OPENAI_API_BASE` | *(empty)*    | API base URL (e.g., `http://kubeai.svc.local/openai/v1`) |
| `OPENAI_API_KEY`  | `not-needed` | API key (KubeAI ignores this)                            |
| `OPENAI_MODEL`    | *(empty)*    | Model name as configured in cluster (e.g., `gemma3-4b`)  |

**Example (KubeAI cluster):**

```bash
OPENAI_API_BASE=http://kubeai.kubeai-system.svc.cluster.local/openai/v1 \
OPENAI_MODEL=gemma3-4b \
./montage-ai.sh run
```

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

---

## Visual Enhancement

| Variable          | Default | Description                                   |
| ----------------- | ------- | --------------------------------------------- |
| `STABILIZE`       | `false` | Enable video stabilization                    |
| `UPSCALE`         | `false` | Enable AI upscaling (Real-ESRGAN)             |
| `ENHANCE`         | `true`  | Enable color/sharpness enhancement            |
| `PRESERVE_ASPECT` | `false` | Letterbox instead of crop when aspect differs |
| `DEEP_ANALYSIS`   | `false` | Enable deep footage analysis (experimental)   |

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

## Output / Export

| Variable           | Default | Description                                                                               |
| ------------------ | ------- | ----------------------------------------------------------------------------------------- |
| `EXPORT_TIMELINE`  | `false` | Export OTIO/EDL timeline (experimental)                                                   |
| `GENERATE_PROXIES` | `false` | Generate proxy files for NLE (experimental)                                               |
| `OUTPUT_CODEC`     | *auto*  | `libx264` by default; switches to `libx265` when most footage is HEVC (override to force) |
| `OUTPUT_PROFILE`   | *auto*  | FFmpeg profile passed to encoder (usually `high` for H.264, `main` for HEVC)              |
| `OUTPUT_LEVEL`     | *auto*  | Encoder level; raised automatically for 4K/high‑FPS projects                              |

*Resolution and FPS are inferred from the dominant input footage (orientation, median size, and common framerate) to minimize re-encoding and quality loss.*

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
