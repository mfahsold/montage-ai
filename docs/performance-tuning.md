# Performance Tuning Guide

This guide helps you optimize Montage AI for your hardware and workflow, avoiding hardcoded limits and enabling fine-grained control.

## Quick Performance Profiles

### Laptop / Low-Power (4 GB RAM, 4 CPU)

```bash
export LOW_MEMORY_MODE=true
export BATCH_SIZE=2
export QUALITY_PROFILE=preview
export MOTION_SAMPLING_MODE=adaptive
export HISTOGRAM_BINS=32
# Trade precision for speed
export CGPU_ENABLED=true
# Offload heavy lifting
./montage-ai.sh run dynamic --cgpu
```

### Workstation / Standard (16 GB RAM, 8+ CPU)

```bash
export BATCH_SIZE=5
export QUALITY_PROFILE=standard
export MOTION_SAMPLING_MODE=full
export HISTOGRAM_BINS=64
export STABILIZE=true
export UPSCALE=false
./montage-ai.sh run dynamic --stabilize
```

### Server / High-Performance (32+ GB RAM, 16+ CPU, GPU)

```bash
export BATCH_SIZE=10
export QUALITY_PROFILE=high
export MOTION_SAMPLING_MODE=full
export HISTOGRAM_BINS=128
# Maximum detail
export UPSCALE=true
export STABILIZE=true
export FFMPEG_HWACCEL=auto
# Auto-detect GPU
./montage-ai.sh run dynamic --upscale --stabilize
```

---

## Motion Analysis Tuning

Fine-tune optical flow (motion detection) for your use case:

### Fast Motion Detection (Real-Time Preview)

```bash
export OPTICAL_FLOW_LEVELS=2
export OPTICAL_FLOW_WINSIZE=7
export OPTICAL_FLOW_ITERATIONS=2
export HISTOGRAM_BINS=32
```

### Balanced (Default)

```bash
export OPTICAL_FLOW_LEVELS=3
export OPTICAL_FLOW_WINSIZE=15
export OPTICAL_FLOW_ITERATIONS=3
export HISTOGRAM_BINS=64
```

### Precise Motion Tracking (High-Quality Output)

```bash
export OPTICAL_FLOW_LEVELS=4
export OPTICAL_FLOW_WINSIZE=21
export OPTICAL_FLOW_ITERATIONS=5
export HISTOGRAM_BINS=128
export MOTION_SAMPLING_MODE=full
```

**Optical Flow Parameters:**
- `OPTICAL_FLOW_LEVELS`: Pyramid levels (1-4; higher = slower, more accurate)
- `OPTICAL_FLOW_WINSIZE`: Search window size (7-31; higher = more context, slower)
- `OPTICAL_FLOW_ITERATIONS`: Refine steps per level (1-5; higher = more accurate, slower)
- `OPTICAL_FLOW_PYR_SCALE`: Image scale between levels (0.5-1.0; affects multi-scale speed)

---

## Blur & Focus Detection

Control how strictly Montage AI filters low-focus clips:

### Lenient (Keep More Clips)

```bash
export BLUR_DETECTION_VARIANCE_THRESHOLD=500.0
```

### Standard (Balanced)

```bash
export BLUR_DETECTION_VARIANCE_THRESHOLD=1000.0
```

### Strict (High-Quality Only)

```bash
export BLUR_DETECTION_VARIANCE_THRESHOLD=1500.0
```

**Hint:** If montage contains too many soft-focus shots, increase threshold. If it's excluding good clips, decrease threshold.

---

## Histogram & Color Matching

Trade precision for speed in visual similarity detection:

```bash
# Speed: 32 bins (fast color matching)
export HISTOGRAM_BINS=32

# Balanced: 64 bins (default)
export HISTOGRAM_BINS=64

# Precision: 128 bins (slow, very precise)
export HISTOGRAM_BINS=128
```

---

## Batch Processing

### Adaptive Batching (Auto-Adjust for Resolution)
By default, Montage AI adapts batch size to input resolution:
- **1080p:** `BATCH_SIZE=5`
- **4K:** `BATCH_SIZE=2`
- **6K+:** `BATCH_SIZE=1`

Override manually:
```bash
export BATCH_SIZE=10  # Force larger batches (use with caution on high-res)
```

---

## Memory Management

### Low-Memory Mode (Laptop/Containers)
Enables sequential processing and smaller buffers:
```bash
export LOW_MEMORY_MODE=true
export BATCH_SIZE=2
export QUALITY_PROFILE=preview
```

### High-Concurrency Mode (Server)
Process multiple clips in parallel:
```bash
export LOW_MEMORY_MODE=false
export BATCH_SIZE=10
export MAX_CONCURRENT_JOBS=4
```

---

## GPU Acceleration

### Auto-Detect GPU
```bash
export FFMPEG_HWACCEL=auto
```

### Force Specific GPU
```bash
export FFMPEG_HWACCEL=nvenc  # NVIDIA
export FFMPEG_HWACCEL=vaapi  # AMD/Intel
export FFMPEG_HWACCEL=qsv    # Intel QuickSync
```

### Cloud GPU (Colab)
```bash
export CGPU_ENABLED=true
export CGPU_GPU_ENABLED=true
export CGPU_TIMEOUT=1200
```

---

## Quality Profiles

### Preview (Fastest, Lowest Quality)
```bash
export QUALITY_PROFILE=preview
export FFMPEG_PRESET=ultrafast
export FINAL_CRF=28
```

### Standard (Balanced)
```bash
export QUALITY_PROFILE=standard
export FFMPEG_PRESET=medium
export FINAL_CRF=18
```

### High (Slower, Better Quality)
```bash
export QUALITY_PROFILE=high
export FFMPEG_PRESET=slow
export FINAL_CRF=17
```

### Master (Highest Quality)
```bash
export QUALITY_PROFILE=master
export FFMPEG_PRESET=veryslow
export FINAL_CRF=16
```

---

## Profiling & Benchmarking

Enable detailed timing:
```bash
export VERBOSE=true
./montage-ai.sh run dynamic 2>&1 | tee profile.log
```

Check which phase is slowest:
```bash
grep -E "Phase|Duration|seconds" profile.log
```

Adjust parameters targeting the slowest phase:
- **Audio Analysis slow?** → Reduce `BATCH_SIZE`
- **Motion detection slow?** → Reduce `OPTICAL_FLOW_LEVELS` or `HISTOGRAM_BINS`
- **Rendering slow?** → Enable `FFMPEG_HWACCEL`, increase `FFMPEG_PRESET`
- **Memory spikes?** → Enable `LOW_MEMORY_MODE`

---

## Environment Variable Cheat Sheet

| Variable | Default | Range | Impact |
| ------- | ------- | ----- | ------ |
| `BATCH_SIZE` | 5 | 1-20 | Higher = faster but more memory |
| `QUALITY_PROFILE` | standard | preview/standard/high/master | Output quality & speed |
| `FFMPEG_PRESET` | medium | ultrafast-veryslow | Encoding speed |
| `LOW_MEMORY_MODE` | false | true/false | Sequential vs parallel |
| `OPTICAL_FLOW_LEVELS` | 3 | 1-4 | Motion accuracy vs speed |
| `OPTICAL_FLOW_WINSIZE` | 15 | 7-31 | Motion context vs speed |
| `HISTOGRAM_BINS` | 64 | 16-256 | Color precision vs speed |
| `BLUR_DETECTION_VARIANCE_THRESHOLD` | 1000 | 100-2000 | Clip quality strictness |
| `FFMPEG_HWACCEL` | auto | auto/nvenc/vaapi/qsv/none | GPU acceleration |
| `CGPU_ENABLED` | false | true/false | Cloud GPU offload |

---

## Common Scenarios

**Scenario:** Slow motion detection on 4K footage
```bash
export OPTICAL_FLOW_LEVELS=2
export OPTICAL_FLOW_WINSIZE=11
export HISTOGRAM_BINS=32
export MOTION_SAMPLING_MODE=adaptive
```

**Scenario:** Laptop running out of memory
```bash
export LOW_MEMORY_MODE=true
export BATCH_SIZE=1
export QUALITY_PROFILE=preview
```

**Scenario:** Too many blurry clips in output
```bash
export BLUR_DETECTION_VARIANCE_THRESHOLD=1500.0  # Stricter
```

**Scenario:** Missing good clips (too strict)
```bash
export BLUR_DETECTION_VARIANCE_THRESHOLD=500.0  # Lenient
```

**Scenario:** Want fastest possible output
```bash
export QUALITY_PROFILE=preview
export MOTION_SAMPLING_MODE=adaptive
export OPTICAL_FLOW_LEVELS=1
```

---

For more details, see [Configuration Reference](configuration.md) and [Architecture](architecture.md).
