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

### Ollama (Local LLM)

| Variable           | Default                             | Description                    |
| ------------------ | ----------------------------------- | ------------------------------ |
| `OLLAMA_HOST`      | `http://host.docker.internal:11434` | Ollama API endpoint            |
| `OLLAMA_MODEL`     | `llava`                             | Model for scene analysis       |
| `DIRECTOR_MODEL`   | `llama3.1:70b`                      | Model for Creative Director    |
| `ENABLE_AI_FILTER` | `false`                             | Enable AI-based clip filtering |

### cgpu / Gemini (Cloud LLM)

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

| Variable        | Default | Description                                 |
| --------------- | ------- | ------------------------------------------- |
| `STABILIZE`     | `false` | Enable video stabilization                  |
| `UPSCALE`       | `false` | Enable AI upscaling (Real-ESRGAN)           |
| `ENHANCE`       | `true`  | Enable color/sharpness enhancement          |
| `DEEP_ANALYSIS` | `false` | Enable deep footage analysis (experimental) |

---

## Performance / Hardware

| Variable            | Default           | Description                                                       |
| ------------------- | ----------------- | ----------------------------------------------------------------- |
| `USE_GPU`           | `auto`            | GPU mode: `auto`, `vulkan`, `v4l2`, `none`                        |
| `FFMPEG_THREADS`    | `0`               | FFmpeg thread count (`0` = auto)                                  |
| `FFMPEG_PRESET`     | `medium`          | Encoding speed: `ultrafast`, `fast`, `medium`, `slow`, `veryslow` |
| `PARALLEL_ENHANCE`  | `true`            | Enable parallel clip enhancement                                  |
| `MAX_PARALLEL_JOBS` | *(CPU cores - 2)* | Maximum parallel workers                                          |

---

## Output / Export

| Variable           | Default | Description                                 |
| ------------------ | ------- | ------------------------------------------- |
| `EXPORT_TIMELINE`  | `false` | Export OTIO/EDL timeline (experimental)     |
| `GENERATE_PROXIES` | `false` | Generate proxy files for NLE (experimental) |

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
