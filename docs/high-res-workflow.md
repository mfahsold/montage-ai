# High-Resolution Workflow Guide

Working with 4K, 6K, 8K, and RAW footage in Montage AI using proxy-based editing workflows.

---

## Overview

High-resolution footage (4K+) requires special handling to keep analysis and editing fast. Montage AI uses a **proxy workflow**: lightweight H.264 proxies are generated for analysis and editing, while the final render uses the original full-resolution files.

```
Original 8K → Generate Proxies (720p) → Analyze & Edit → Render from Originals → Final Output
```

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 16 GB | 32 GB+ |
| CPU | 4 cores | 8+ cores |
| Disk | 50 GB free | 100 GB+ |
| GPU | Optional | NVENC/VAAPI recommended |

> **Tip:** For 8K footage, GPU encoding (`FFMPEG_HWACCEL=auto`) reduces render time by 3-5x compared to CPU-only.

---

## Step 1: Generate Proxies

Use the built-in proxy generator to create lightweight editing proxies:

```bash
# Generate 720p H.264 proxies for all footage in data/input/
./montage-ai.sh generate-proxies

# Custom proxy resolution (default: 720p)
PROXY_HEIGHT=540 ./montage-ai.sh generate-proxies

# Specify input directory
./montage-ai.sh generate-proxies --input /path/to/8k-footage
```

**Proxy settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_HEIGHT` | `720` | Proxy height in pixels (width auto-calculated) |
| `PROXY_HEIGHT_LARGE` | `720` | Override for footage > 4K |
| `GENERATE_PROXIES` | `true` | Auto-generate proxies during analysis |

Proxies are saved alongside originals with a `_proxy` suffix.

---

## Step 2: Analyze and Edit

Run the montage pipeline as normal. Montage AI automatically uses proxies for analysis when available:

```bash
# Standard analysis using proxies
./montage-ai.sh run dynamic

# With story engine for narrative structure
./montage-ai.sh run documentary --story-engine
```

Analysis parameters optimized for high-res:

```bash
# Recommended for 6K+ footage
export OPTICAL_FLOW_LEVELS=4        # More precise motion detection
export OPTICAL_FLOW_WINSIZE=21      # Larger search context
export BATCH_SIZE=2                 # Smaller batches to fit in memory
export MOTION_SAMPLING_MODE=full    # Full precision
```

---

## Step 3: Final Render

### Quality Profiles for High-Res

| Profile | Resolution | Codec | Use Case |
|---------|-----------|-------|----------|
| `preview` | 360p | H.264 | Quick review |
| `standard` | Source | H.264 | General use |
| `high` | Source | H.264 | Delivery |
| `master` | Source | H.265 10-bit | Archival / DCI |

```bash
# High-quality 4K render
QUALITY_PROFILE=high ./montage-ai.sh hq

# Master quality (H.265, 10-bit, slow preset)
QUALITY_PROFILE=master ./montage-ai.sh hq

# 4K with GPU acceleration
QUALITY_PROFILE=high FFMPEG_HWACCEL=auto ./montage-ai.sh hq
```

### ProRes for NLE Roundtrip

For DaVinci Resolve or Final Cut Pro conform:

```bash
# ProRes 422 output (large files, best NLE compatibility)
OUTPUT_CODEC=prores ./montage-ai.sh hq
```

---

## Proxy Conform in NLE

After creating your rough cut, export a timeline and conform in your NLE:

### DaVinci Resolve

1. Export timeline from Montage AI:
   ```bash
   ./montage-ai.sh export-to-nle --manifest /data/output/manifest.json --formats otio edl
   ```

2. In DaVinci Resolve:
   - **File > Import Timeline** -- select the `.edl` or `.otio` file
   - **Right-click Media Pool > Relink Media** -- point to original 8K files
   - Resolve automatically re-links proxy clips to full-resolution originals

### Final Cut Pro

1. Export an XML timeline:
   ```bash
   ./montage-ai.sh export-to-nle --manifest /data/output/manifest.json --formats premiere
   ```

2. In Final Cut Pro:
   - **File > Import > XML** -- select the exported file
   - **File > Relink Files** -- select the original media directory

### Premiere Pro

1. Export timeline:
   ```bash
   ./montage-ai.sh export-to-nle --manifest /data/output/manifest.json --formats premiere edl
   ```

2. In Premiere Pro:
   - **File > Import** -- select the `.xml` file
   - **Clip > Link Media** -- relink to original files

---

## CLI Examples

```bash
# Full 8K proxy workflow
PROXY_HEIGHT=720 ./montage-ai.sh generate-proxies
QUALITY_PROFILE=high ./montage-ai.sh run dynamic --export
./montage-ai.sh export-to-nle --manifest /data/output/manifest.json

# 6K with GPU + story engine
FFMPEG_HWACCEL=auto ./montage-ai.sh run documentary --story-engine --stabilize

# Quick 8K preview (proxies only)
QUALITY_PROFILE=preview ./montage-ai.sh preview

# Master-quality archival render
QUALITY_PROFILE=master FFMPEG_HWACCEL=auto ./montage-ai.sh hq --stabilize --upscale
```

---

## Performance Benchmarks

Approximate render times on different hardware (30-minute 8K source, 3-minute output):

| Hardware | Preview | Standard | High | Master |
|----------|---------|----------|------|--------|
| **Laptop** (16 GB, i7, no GPU) | ~3 min | ~12 min | ~25 min | ~45 min |
| **Workstation** (32 GB, Ryzen 9, RTX 4070) | ~1 min | ~4 min | ~8 min | ~15 min |
| **Server** (64 GB, Xeon, A4000) | ~30s | ~2 min | ~4 min | ~8 min |
| **Cloud GPU** (cgpu, A100) | ~20s | ~1 min | ~2 min | ~5 min |

> Times vary with source codec, complexity, and enabled effects (stabilize, upscale, denoise).

---

## Tips

- **Always generate proxies first** for footage > 4K. Analysis on full-res 8K is 5-10x slower.
- **Use `BATCH_SIZE=1` or `2`** for 8K to avoid memory pressure.
- **GPU encoding** (`FFMPEG_HWACCEL=auto`) is strongly recommended for high-res final renders.
- **Cloud GPU** (`--cgpu-gpu`) can offload upscaling and denoising for systems without a discrete GPU.
- **Export timelines** for final color grading and sound design in a dedicated NLE.

---

## Related Documentation

- [Performance Tuning](performance-tuning.md) -- all tuning parameters
- [Configuration: Quality Profiles](configuration.md#quality-profiles) -- profile details
- [Configuration: GPU Acceleration](configuration.md#gpu--hardware-acceleration) -- GPU setup
- [Features](features.md) -- full feature matrix
