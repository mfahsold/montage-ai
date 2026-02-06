# Configuration Defaults (Code-Aligned)

This snapshot was audited against `src/montage_ai/config.py` and
`src/montage_ai/config_pools.py` on 2026-02-05.

Scope: high-impact environment variables and computed defaults. This is not
exhaustive; the source files above are the full reference.

## Computed Defaults (CPU / cgroup aware)

| Variable | Default | Notes |
| --- | --- | --- |
| `MAX_PARALLEL_JOBS` | effective CPU count | Effective CPU count respects cgroup/affinity and can be clamped by `MONTAGE_PREVIEW_CPU_LIMIT`. |
| `MAX_CONCURRENT_JOBS` | `max(4, effective_cpu_count - 1)` | Concurrency guard for job-level parallelism. |
| `MAX_SCENE_WORKERS` | `os.cpu_count()` | Used to cap scene detection workers. |
| `PROCESS_WORKERS` | `max(2, cpu_count // 2)` | From `config_pools`; capped to `cpu_count`. |
| `THREAD_WORKERS` | `min(cpu_count, 8)` | From `config_pools`. |
| `HTTP_CONNECTIONS` | `min(cpu_count * 2, 32)` | From `config_pools`. |
| `HTTP_MAXSIZE` | `min(cpu_count * 4, 64)` | From `config_pools`. |
| `QUEUE_SIZE` | `max(10, cpu_count * 2)` | From `config_pools` (web UI SSE queue). |

Note: pool defaults use `os.cpu_count()` and do not apply cgroup limits.

## Quality Profile (Encoding Defaults)

| QUALITY_PROFILE | FFMPEG_PRESET | FINAL_CRF | Notes |
| --- | --- | --- | --- |
| `preview` | `ultrafast` | `28` | Fast, low-latency. |
| `standard` | `medium` | `18` | Default profile. |
| `high` | `slow` | `17` | Higher quality. |
| `master` | `slow` | `16` | Also switches to `libx265` + `yuv420p10le` unless overridden. |

## Preview Fast Path

| Variable | Default | Notes |
| --- | --- | --- |
| `PREVIEW_WIDTH` | `640` | Preview resolution width. |
| `PREVIEW_HEIGHT` | `360` | Preview resolution height. |
| `PREVIEW_CRF` | `28` | Preview quality. |
| `PREVIEW_PRESET` | `ultrafast` | Preview encoding speed preset. |
| `PREVIEW_MAX_DURATION` | `30.0` | Seconds. |
| `PREVIEW_MAX_INPUT_SIZE_MB` | `200` | Per-file limit for preview mode. |
| `PREVIEW_MAX_FILES` | `3` | Max files in preview fast-path. |
| `PREVIEW_JOB_TIMEOUT` | `300` | Seconds. |
| `PREVIEW_TIME_TARGET` | `180` | KPI target in seconds. |

## Core Feature Flags (Common)

| Variable | Default | Notes |
| --- | --- | --- |
| `ENHANCE` | `true` | Global enhancement toggle. |
| `STABILIZE` | `false` | Stabilization off by default. |
| `UPSCALE` | `false` | Upscaling off by default. |
| `PARALLEL_ENHANCE` | `true` | Parallel clip enhancement. |
| `LLM_CLIP_SELECTION` | `true` | LLM-based clip selection enabled. |
| `DEEP_ANALYSIS` | `false` | Optional heavier analysis. |
| `LOW_MEMORY_MODE` | `false` | Enables more conservative limits. |

## Thresholds (Detection)

| Variable | Default | Notes |
| --- | --- | --- |
| `SCENE_THRESHOLD` | `30.0` | PySceneDetect ContentDetector threshold. |
| `SPEECH_THRESHOLD` | `0.5` | VAD threshold. |
| `SPEECH_MIN_DURATION` | `250` | ms. |
| `SPEECH_MIN_SILENCE` | `100` | ms. |
| `SILENCE_THRESHOLD` | `-35` | dB (stored without unit for silencedetect). |
| `SILENCE_DURATION` | `0.5` | Seconds. |
| `FACE_CONFIDENCE` | `0.6` | MediaPipe face confidence. |
| `DUCKING_CORE_THRESHOLD` | `0.1` | Strong ducking. |
| `DUCKING_SOFT_THRESHOLD` | `0.03` | Gentle ducking. |
| `MUSIC_MIN_DURATION` | `5.0` | Seconds. |
| `BLUR_DETECTION_VARIANCE_THRESHOLD` | `1000.0` | Laplacian variance cutoff. |
| `MOTION_DIRECTION_THRESHOLD` | `0.5` | Optical flow direction threshold. |

## Proxy / Analysis Cache

| Variable | Default | Notes |
| --- | --- | --- |
| `ENABLE_PROXY_ANALYSIS` | `true` | Auto-enable proxies for large inputs. |
| `PROXY_HEIGHT` | `720` | Base proxy height. |
| `ADAPTIVE_PROXY_HEIGHT` | `true` | Enables size-based heights. |
| `PROXY_HEIGHT_SMALL` | `360` | Small proxy tier. |
| `PROXY_HEIGHT_MEDIUM` | `540` | Medium proxy tier. |
| `PROXY_HEIGHT_LARGE` | `720` | Large proxy tier. |
| `AUTO_PROXY_DURATION_THRESHOLD` | `600.0` | Seconds. |
| `AUTO_PROXY_RESOLUTION_THRESHOLD` | `2000000` | Pixels. |
| `CACHE_PROXIES` | `true` | Reuse proxies across passes. |
| `PROXY_CACHE_TTL_SECONDS` | `86400` | 24h. |
| `PROXY_CACHE_MAX_BYTES` | `1073741824` | 1 GiB. |
| `PROXY_CACHE_MIN_AGE_SECONDS` | `60` | Avoids thrash. |
| `PROXY_LOCK_TIMEOUT_SECONDS` | `900` | File lock timeout. |
| `PREFER_ANALYSIS_PROXY_FOR_PREVIEW` | `true` | Use lighter proxy in preview. |

## Scene Cache (Distributed)

| Variable | Default | Notes |
| --- | --- | --- |
| `SCENE_CACHE_DIR` | `$OUTPUT_DIR/scene_cache` | Shard outputs for distributed scene detection. |

## Cluster (Distributed Jobs)

| Variable | Default | Notes |
| --- | --- | --- |
| `SCENE_DETECT_TIER` | `medium` | Resource tier for scene detection jobs. |

## Timeouts (Seconds)

| Variable | Default | Notes |
| --- | --- | --- |
| `ANALYSIS_TIMEOUT` | `120` | Scene/analysis task timeout. |
| `FFPROBE_TIMEOUT` | `10` | Metadata extraction timeout. |
| `FFMPEG_SHORT_TIMEOUT` | `30` | Short operation timeout. |
| `FFMPEG_TIMEOUT` | `120` | Standard FFmpeg timeout. |
| `FFMPEG_LONG_TIMEOUT` | `600` | Long FFmpeg timeout. |
| `RENDER_TIMEOUT` | `3600` | Full render timeout. |
| `JOB_TIMEOUT` | `1800` | Default job timeout. |

## Export Defaults

| Variable | Default | Notes |
| --- | --- | --- |
| `EXPORT_WIDTH` | `1080` | Timeline export width. |
| `EXPORT_HEIGHT` | `1920` | Timeline export height. |
| `EXPORT_FPS` | `30.0` | Frames per second. |
| `EXPORT_PROJECT_NAME` | `fluxibri_montage` | Default timeline project name. |
