# Getting Started on ARM64 (Snapdragon, Apple Silicon, Raspberry Pi)

This guide covers ARM64-specific setup. For general setup, see the [README](../README.md).

## ARM64 Limitations

> **MediaPipe not available on ARM64.** Auto-reframe uses center-crop fallback instead of face detection. This works well for most content. The `[WARN] MediaPipe not installed` log message is expected and safe to ignore. See [Optional Dependencies](OPTIONAL_DEPENDENCIES.md) for details.

## Prerequisites

- Docker + Docker Compose v2 ([general requirements](../README.md#system-requirements))
- USB-C SSD recommended (not SD card: 10x faster I/O)
- 64-bit OS (Ubuntu 22.04, Raspberry Pi OS 64-bit, macOS, or Windows on Arm)

## Setup

### Step 1: Install Docker

```bash
# On Raspberry Pi OS:
curl -sSL https://get.docker.com | sh

# Add your user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker compose version
```

### Step 2: Clone & Setup

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai

mkdir -p data/input data/music data/output data/assets

./scripts/validate-onboarding.sh
```

### Step 3: Adjust docker-compose.yml

```yaml
montage-ai:
  deploy:
    resources:
      limits:
        memory: 6g      # Leave 2GB for OS
        cpus: 4         # All 4 cores
```

### Step 4: Build & Run Preview Only

```bash
# Will take longer (10-15 min build on first run)
docker compose build

# Test (preview render will take 2-3 minutes)
QUALITY_PROFILE=preview docker compose run --rm montage-ai /app/montage-ai.sh run dynamic
```

---

## ARM64 Specific Troubleshooting

### "OCI runtime error: exec format error"

**Problem:** Running x86 binaries on ARM (or vice versa)

**Solution:** 
```bash
# Check what architecture Docker is using
docker run --rm alpine uname -m

# Should show "aarch64" on ARM systems

# Rebuild to ensure ARM64 image
docker compose build --no-cache
```

### "ffmpeg: not found" in container

**Problem:** FFmpeg not installed for ARM

**Solution:**
```bash
# Verify ffmpeg is in container
docker compose run --rm montage-ai ffmpeg -version

# If missing, rebuild:
docker compose build --no-cache
```

### Slow rendering on Apple Silicon

**Problem:** M1/M2 video encoding is slower than expected

**Solution:**
```bash
# Check if VideoToolbox is being used
docker compose run --rm montage-ai ffmpeg -hide_banner -encoders | grep hevc_videotoolbox

# If available, enable hardware encoding:
FFMPEG_PRESET=fast docker compose run --rm montage-ai /app/montage-ai.sh run

# Or use Preview mode (2x faster):
QUALITY_PROFILE=preview docker compose run --rm montage-ai /app/montage-ai.sh run
```

### Memory usage spikes & container crashes

**Problem:** Docker memory limit set too low

**Solution:**
```bash
# Check current limit in docker-compose.yml
cat docker-compose.yml | grep memory

# Increase it (e.g., from 8g to 10g) and rebuild:
docker compose up
```

---

## ARM64 Hardware Acceleration

### Apple Silicon VideoToolbox

Apple Silicon has native video encoding hardware (VideoToolbox).

```bash
# Check available (M1/M2/M3 should have these):
docker compose run --rm montage-ai ffmpeg -hide_banner -encoders | grep -E "hevc_videotoolbox|h264_videotoolbox"

# Use in FFmpeg pipeline (automatic if available)
```

### Snapdragon (Windows on Arm)

Snapdragon X chips have Qualcomm Hexagon DSP and Adreno GPU.

```bash
# Check supported hardware
docker compose run --rm montage-ai ffmpeg -hide_banner -hwaccels
```

**Note:** Most standard FFmpeg builds don't include Snapdragon-specific acc. We recommend using Preview mode for now.

---

## Performance Expectations (ARM64)

### Render Times (Approximate)

| Device | Mode | Duration | Time |
|--------|------|----------|------|
| **Apple M2** | Preview | 5 min input | 2-3 min |
| **Apple M2** | Normal | 5 min input | 8-12 min |
| **Snapdragon X** | Preview | 5 min input | 2-3 min |
| **Snapdragon X** | Normal | 5 min input | 10-15 min |
| **Raspberry Pi 5** | Preview | 5 min input | 30-60 min |

**Tip:** For faster testing, use 1-2 minute input clips and `QUALITY_PROFILE=preview`.

---

## Validation Script

Run this to verify everything is set up correctly:

```bash
./scripts/validate-onboarding.sh
```

This script checks:
- ✅ Docker & Docker Compose versions
- ✅ Hardware (RAM, CPU, disk)
- ✅ Architecture (will show `aarch64` for ARM)
- ✅ Docker build for ARM64
- ✅ Python imports
- ✅ FFmpeg availability
- ✅ First montage render (preview)

---

## Next Steps

1. **Follow [docs/getting-started.md](getting-started.md)** for common workflows
2. **Read [docs/performance-tuning.md](performance-tuning.md)** for rendering optimization
3. **Check [docs/troubleshooting.md](troubleshooting.md)** if issues arise

---

## ARM Docker Image Notes

The Dockerfile automatically builds the correct ARM64 image using:

```dockerfile
ARG TARGETARCH

# Base image (supports both amd64 and arm64)
FROM python:3.10-slim-bookworm

# Install ARM64-compatible packages
RUN [...] ffmpeg libsndfile1 [...]

# Skip Intel QSV drivers on ARM
RUN if [ "$TARGETARCH" = "amd64" ]; then
    apt-get install -y intel-media-va-driver-non-free
fi
```

**No special action needed** — Docker automatically selects the correct base image and layers for your architecture.

---

## Still Having Issues?

**On Snapdragon:**
- Verify Windows on Arm is properly installed: `wsl --list --verbose`
- Check Docker Desktop resource allocation (Settings → Resources)
- Try `docker run hello-world` to test Docker

**On Apple Silicon:**
- Verify native installation: `docker run --rm alpine uname -m` should show `aarch64`
- Check Docker Desktop → Preferences → General → Use native Docker socket

**On Raspberry Pi:**
- Use SSD, not SD card (SD card is too slow)
- Monitor temperature: `vcgencmd measure_temp` (should stay <60°C)
- Use preview mode only: `QUALITY_PROFILE=preview`

**Still stuck?**
- Run `./scripts/validate-onboarding.sh` and share output
- Check `docs/troubleshooting.md`
- Open issue on GitHub with `[ARM]` in title
