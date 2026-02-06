# Configuration Reference

Complete reference for all environment variables and settings.
For an audited snapshot of code-aligned defaults, see `docs/config-defaults.md`.

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

## Supported Media Formats

- **Video (containers/codecs via FFmpeg):** mp4, mov, avi, mkv, webm, m4v, mxf, mts, m2ts, ts
- **Audio:** mp3, wav, flac, aac, m4a, ogg
- **Images:** png, jpg, jpeg, gif, webp, bmp, svg (SVG overlays require `rsvg-convert`, `inkscape`, or ImageMagick.)
- **Notes:** ProRes/DNxHR/DNxHD typically arrive in mov/mxf; AVCHD/HDV arrive as mts/m2ts/ts. Formats rely on the FFmpeg build shipped with the container; verify with `ffprobe sample.mxf` if in doubt. Upload validation uses `FileTypeConfig` from `src/montage_ai/config.py`, shared by the web UI and API handlers.

---

## Queue Routing

- `QUEUE_FAST_NAME` (default: `default`) – short/preview + transcript jobs
- `QUEUE_HEAVY_NAME` (default: `heavy`) – volle Renders, Upscale/Enhance, Nicht-Preview
- Preview-Jobs werden automatisch in die Fast-Queue geleitet; Standard/High landet in Heavy.

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

For LiteLLM, vLLM, LocalAI, or any OpenAI-compatible endpoint:

| Variable              | Default      | Description                                                      |
| --------------------- | ------------ | ---------------------------------------------------------------- |
| `OPENAI_API_BASE`     | *(empty)*    | API base URL (e.g., `http://<OPENAI_BASE_HOST>:<PORT>/v1`) |
| `OPENAI_API_KEY`      | *(empty)*    | API key for OpenAI-compatible proxy (set via secret/env)          |
| `OPENAI_MODEL`        | *(empty)*    | Model ID or `auto` to discover via `/v1/models`                  |
| `OPENAI_VISION_MODEL` | *(empty)*    | Vision model ID or `auto` (optional)                             |

**Example (Cluster / LiteLLM):**

```bash
OPENAI_API_BASE=http://litellm.<LLM_NAMESPACE>.svc.cluster.local:4000/v1 \
OPENAI_API_KEY=<OPENAI_API_KEY> \
OPENAI_MODEL=auto \
./montage-ai.sh run
```

**Example (Together AI / Claude via OpenAI-compatible API):**

```bash
# point Montage AI at Together's OpenAI-compatible endpoint
OPENAI_API_BASE=https://api.together.xyz/v1 \
OPENAI_API_KEY=YOUR_TOGETHER_API_KEY \
OPENAI_MODEL=auto \
./montage-ai.sh run
```

### Claude Code Router (musistudio/claude-code-router) — recommended on org/user level

If you want to route Claude/"Together" through a single, OpenAI‑compatible proxy (recommended for auditability, rate‑limits and central key management), run the `musistudio/claude-code-router` at the org/user level and point `OPENAI_API_BASE` at it.

Benefits:
- Central API key stored in one place (GitHub org secret, K8s Secret or cloud secret manager) ✅
- Single endpoint for all CI runners / cluster nodes (easier rotation & logging) ✅

Quick local example (router runs locally and exposes OpenAI‑compatible /v1):

```bash
# run router locally and point Montage at it
# (do NOT commit your API key; keep it in env / secret manager)
TOGETHER_API_KEY="<redacted>" docker compose -f deploy/claude-code-router/docker-compose.claude-router.yml up -d
export OPENAI_API_BASE="http://localhost:8080/v1"
export OPENAI_API_KEY="${TOGETHER_API_KEY}"
./montage-ai.sh run
```

Security & deployment — store the real API key at the org/user level (do NOT commit):

1) GitHub (org or repo) secrets — use `OPENAI_API_BASE` + `OPENAI_API_KEY` in workflows.
2) Kubernetes — create a `Secret` (example below) and reference it from the Deployment.
3) Cloud secret managers (GCP Secret Manager / AWS Secrets Manager) — inject at deploy time.

**Model Selection (DRY):**

- Do not hardcode model IDs. Use `OPENAI_MODEL=auto` and let the app query `/v1/models`.
- For vision, set `OPENAI_VISION_MODEL=auto` if a vision model is available.

### Ollama (Local LLM)

| Variable           | Default                             | Description                    |
| ------------------ | ----------------------------------- | ------------------------------ |
| `OLLAMA_HOST`      | `http://host.docker.internal:11434` | Ollama API endpoint            |
| `OLLAMA_MODEL`     | `llava`                             | Model for scene analysis       |
| `DIRECTOR_MODEL`   | `llama3.1:70b`                      | Model for Creative Director    |
| `ENABLE_AI_FILTER` | `false`                             | Enable AI-based clip filtering |

**Cluster note:** If you run Ollama in-cluster, set `OLLAMA_HOST` explicitly (or set
`CLUSTER_NAMESPACE` so service DNS can be derived).

### Cluster storage (PVC split)

Distributed jobs can use split PVCs for input/output/music/assets. Configure
these in `deploy/k3s/config-global.yaml` under `storage.pvc` so the rendered
`cluster-config.env` exports:

- `PVC_INPUT_NAME`
- `PVC_OUTPUT_NAME`
- `PVC_MUSIC_NAME`
- `PVC_ASSETS_NAME`

---

## Cluster Performance Tuning

Recommended for large inputs (4K/AV1) and distributed runs:

| Variable | Default | Description |
| --- | --- | --- |
| `CLUSTER_PARALLELISM` | `4` | Shard count for distributed scene detection + render. |
| `SCENE_DETECT_TIER` | `medium` | Resource tier for scene detection jobs (`small`, `medium`, `large`). |
| `SCENE_CACHE_DIR` | `$OUTPUT_DIR/scene_cache` | Shared scene-cache directory for shard outputs. |
| `PROXY_CACHE_DIR` | *(empty)* | Shared proxy cache for analysis (set to a RW PVC to avoid re-generation). |
| `PROXY_HEIGHT_LARGE` | `720` | Proxy height for large files (lower = faster decode). |

**Example (K8s):**
```bash
SCENE_DETECT_TIER=large \
SCENE_CACHE_DIR=/data/output/scene_cache \
PROXY_CACHE_DIR=/data/output/proxy_cache
```

Notes:
- Scene detection is CPU‑bound; for AV1/4K, set `SCENE_DETECT_TIER=large`.
- If you set `PROXY_CACHE_DIR`, keep it on a shared RW PVC so shards can reuse proxies.

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

| Variable          | Default       | Description                                   |
| ----------------- | ------------- | --------------------------------------------- |
| `STABILIZE`       | `false`       | Enable video stabilization                    |
| `UPSCALE`         | `false`       | Enable AI upscaling (Real-ESRGAN)             |
| `ENHANCE`         | `true`        | Enable color/sharpness enhancement            |
| `PRESERVE_ASPECT` | `false`       | Letterbox instead of crop when aspect differs |
| `DEEP_ANALYSIS`   | `false`       | Enable deep footage analysis (experimental)   |
| `COLORLEVELS`     | `true`        | Apply broadcast-safe levels (16-235)          |
| `LUMA_NORMALIZE`  | `true`        | Normalize luma for consistency across clips   |

### Color Grading

| Variable          | Default       | Description                                                             |
| ----------------- | ------------- | ----------------------------------------------------------------------- |
| `COLOR_GRADING`   | `auto`        | Color grading preset (overrides style default)                          |
| `COLOR_INTENSITY` | `1.0`         | Color grading strength (0.0-1.0). Values of 0.5-0.7 often look natural. |

**Available Color Grading Presets:**

| Preset          | Description                                          |
| --------------- | ---------------------------------------------------- |
| `auto`          | Use style template's default color grade             |
| `none`          | No color grading applied                             |
| `neutral`       | Broadcast-safe levels (Rec.709 compliant)            |
| `natural`       | True-to-source with minimal adjustment               |
| `warm`          | Golden warmth (highlight red/yellow)                 |
| `cool`          | Blue/teal tone (shadows/midtones)                    |
| `golden_hour`   | Sunset/sunrise warmth                                |
| `blue_hour`     | Dawn/dusk cool tones                                 |
| `teal_orange`   | Hollywood blockbuster look (orange skin, teal bg)    |
| `cinematic`     | S-curve contrast with slight desaturation            |
| `blockbuster`   | Action movie: strong teal/orange + high contrast     |
| `vibrant`       | Punchy, saturated colors                             |
| `desaturated`   | Muted, filmic look                                   |
| `high_contrast` | Strong black & white with midtone emphasis           |
| `vintage`       | Faded film with lifted blacks                        |
| `filmic_warm`   | Classic warm film stock                              |
| `noir`          | High contrast with desaturation                      |
| `documentary`   | Natural with sharpness emphasis                      |

**Example:**
```bash
COLOR_GRADING=teal_orange \
COLOR_INTENSITY=0.7 \
./montage-ai.sh run hitchcock
```

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

### Fluxibri Cluster with GPU Encoding

```bash
# .env file or docker-compose environment
OPENAI_API_BASE=http://litellm.<LLM_NAMESPACE>.svc.cluster.local:4000/v1
OPENAI_API_KEY=<OPENAI_API_KEY>
OPENAI_MODEL=auto
ENABLE_AI_FILTER=true
FFMPEG_HWACCEL=auto
PRESERVE_ASPECT=false
```

### Local Development (Ollama + CPU)

```bash
OLLAMA_HOST=http://<OLLAMA_HOST>:11434
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
| `USE_FFMPEG_MCP`    | `false`           | Enable remote ffmpeg MCP service for encoding                     |
| `FFMPEG_MCP_ENDPOINT` | *(empty)*       | Full MCP URL (overrides host/port)                                |
| `FFMPEG_MCP_HOST`   | *(empty)*         | MCP host (used when endpoint not set)                             |
| `FFMPEG_MCP_PORT`   | `8080`            | MCP port                                                          |
| `FFMPEG_THREADS`    | `0`               | FFmpeg thread count (`0` = auto)                                  |
| `FFMPEG_PRESET`     | `medium`          | Encoding speed: `ultrafast`, `fast`, `medium`, `slow`, `veryslow` |
| `PARALLEL_ENHANCE`  | `true`            | Enable parallel clip enhancement                                  |
| `MAX_PARALLEL_JOBS` | *(effective CPU cores)* | Maximum parallel workers                                     |
| `MONTAGE_PREVIEW_CPU_LIMIT` | *(unset)* | Optional clamp for effective CPU count in preview benchmarks     |

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

<!-- markdownlint-disable MD060 MD032 -->
## Memory Management

Centralized in `resources` configuration and exported as environment variables for subprocess compatibility.

| Variable                      | Default | Description                                                     |
| ----------------------------- | ------- | --------------------------------------------------------------- |
| `MEMORY_LIMIT_GB`             | `8`     | Effective memory limit for adaptive memory manager (in GB)      |
| `MEMORY_WARNING_THRESHOLD`    | `0.75`  | Fraction of limit that triggers proactive cleanup               |
| `MEMORY_CRITICAL_THRESHOLD`   | `0.90`  | Fraction of limit that indicates critical pressure              |
| `MEMORY_SAFETY_MARGIN_MB`     | `500`   | Reserved headroom when estimating safe batch sizes              |

These map to `settings.resources.memory_limit_gb`, `memory_warning_threshold`, and `memory_critical_threshold`.

**Hardware-specific recommendations:**

- 8–16GB RAM: Increase `MEMORY_LIMIT_GB` to match container limit if higher.
- 16–32GB RAM: Defaults are generally safe.
- 32GB+ RAM: Consider `MEMORY_LIMIT_GB=48` for long-form edits.

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

## Caching

Caching is unified across analysis and metadata subsystems.

| Variable                     | Default | Description                                        |
| ---------------------------- | ------- | -------------------------------------------------- |
| `CACHE_INVALIDATION_HOURS`   | `24`    | TTL for cache entries (hours)                      |
| `DISABLE_ANALYSIS_CACHE`     | `false` | Disable analysis cache when `true`                 |
| `CACHE_VERSION`              | `1.0`   | Global cache version; bump to invalidate all caches|

### Cache Locations

| Variable                | Default                                              | Description |
| ----------------------- | ---------------------------------------------------- | ----------- |
| `METADATA_CACHE_DIR`    | `${XDG_CACHE_HOME:-~/.cache}/montage_ai/metadata`   | Centralized cache directory for metadata and analysis when set. If unset, analysis results default to sidecar files located next to each input clip to avoid permission issues outside containers. |
| `TENSION_METADATA_DIR`  | `${XDG_CACHE_HOME:-~/.cache}/montage_ai/tension`    | Directory for derived metrics (e.g., tension/energy) used across runs. |

Notes:
- Defaults follow the OS XDG cache convention to avoid writing to `/data` on non-container hosts.
- Sidecar caching is the default for analysis results unless `METADATA_CACHE_DIR` is explicitly provided.
- Both locations are created on demand; respect `CACHE_INVALIDATION_HOURS` and `CACHE_VERSION`.

## Monitoring

Centralized in `monitoring` configuration with environment exports for compatibility.

| Variable                | Default | Description                                 |
| ----------------------- | ------- | ------------------------------------------- |
| `MONITOR_MEM_INTERVAL`  | `30.0`  | Memory telemetry log interval (seconds)     |
| `LOG_FILE`              | *(auto)*| Path to tee log file (defaults to output dir)|
| `PREVIEW_TIME_TARGET`   | `180`   | KPI target for Time-to-First-Preview (s)    |


These map to `settings.cache.analysis_ttl_hours`, `settings.cache.metadata_ttl_hours`, and `settings.cache.version`.

Notes:
- Metadata cache shares the same TTL unless overridden.
- Both analysis and metadata caches use the same `CACHE_VERSION`.

<!-- markdownlint-enable MD060 MD032 -->

<!-- markdownlint-disable MD060 MD032 -->
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

### Encoding Overrides

- `FORCE_CGPU_ENCODING`: When `true`, routes encoding to the CGPU backend regardless of local GPU availability. Maps to `settings.gpu.force_cgpu_encoding` and is exported to subprocess env for compatibility.

---

## Preview Profile

Preview generation (Transcript/Shorts previews, analysis proxies) uses fast, low‑latency defaults that are now configurable centrally via `settings.preview`.

| Variable             | Default     | Description                                               |
| -------------------- | ----------- | --------------------------------------------------------- |
| `PREVIEW_WIDTH`      | `640`       | Preview width (pixels)                                    |
| `PREVIEW_HEIGHT`     | `360`       | Preview height (pixels)                                   |
| `PREVIEW_CRF`        | `28`        | Quality for previews (lower = higher quality, larger)     |
| `PREVIEW_PRESET`     | `ultrafast` | Encoder preset for previews                               |
| `PREVIEW_MAX_DURATION` | `30.0`    | Max duration for previews (seconds)                       |

Where it’s used:
- `ffmpeg_config`: preview params and constants now mirror `settings.preview`.
- `scene_analysis`: proxy generation uses preview preset/CRF instead of literals.
- `preview_generator`: duration and encoder params come from preview config.
- `text_editor`: preview resolution and params are read via `ffmpeg_config`.

Notes:
- `QUALITY_PROFILE=preview` still maps encoder `preset` and `crf` for the final render pipeline. The preview profile above controls dedicated fast previews and analysis proxies.

Orientation and sizing:
- Previews preserve the source orientation. Defaults (640x360) apply to landscape; portrait sources will map to 360x640 automatically.
- Proxy generation preserves aspect ratio and typically limits by height (e.g., `scale=-2:PREVIEW_HEIGHT`), so final dimensions may be rounded by encoder constraints.

### On-cluster preview SLO benchmark (self-hosted)

For canonical SLOs, metrics, and benchmark steps, see `docs/operations/preview-slo.md`.
Keep the detailed workflow there to avoid duplication and drift.


## Export Settings

Control the format and specifications of the final video output.

| Variable          | Default | Description                                                                 |
| ----------------- | ------- | --------------------------------------------------------------------------- |
| `EXPORT_WIDTH`    | `1920`  | Output video width (pixels)                                                 |
| `EXPORT_HEIGHT`   | `1080`  | Output video height (pixels)                                                |
| `EXPORT_FPS`      | `30.0`  | Output frame rate                                                           |
| `EXPORT_TIMELINE` | `false` | Export timeline files (`.otio`, `.edl`, `.csv`) alongside video             |
| `GENERATE_PROXIES`| `false` | Generate low-res proxy files for NLE import                                 |

---

### Resolution Overrides

- When both `EXPORT_WIDTH` and `EXPORT_HEIGHT` are set, they force an explicit output resolution and orientation.
- When not set, output resolution and orientation are inferred from the dominant input footage (orientation, median size, common FPS) to minimize re-encoding.
- Explicit overrides are only applied when the variables are present; otherwise inference remains in effect.

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

<!-- markdownlint-enable MD060 MD032 -->
