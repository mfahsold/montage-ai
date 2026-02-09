# Getting Started on ARM (Snapdragon & Apple Silicon)

This guide covers installation on **ARM64 architecture**, including Windows on Arm (Snapdragon), Apple Silicon (M1/M2/M3), and Raspberry Pi 5+.

---

## Quick Check: What Architecture Am I?

```bash
# Check your CPU architecture
uname -m

# You should see one of:
#   aarch64      ← ARM64 (Snapdragon, Apple Silicon, Raspberry Pi)
#   x86_64       ← Intel/AMD (standard laptop)
#   arm7l        ← ARM 32-bit (older Raspberry Pi)
```

---

## System Requirements (ARM64)

### Minimum Specs
- **RAM:** 16 GB (8 GB for preview mode)
- **CPU:** 4+ cores (ARM processors are efficient, so fewer cores is OK)
- **Disk:** 20 GB free (videos + processing space)
- **Docker:** 20.10+ with ARM64 support
- **Docker Compose:** v2.0+

### ARM64 Specific Notes

| Device | RAM | CPU Cores | Status | Notes |
|--------|-----|-----------|--------|-------|
| **Windows on Arm (Snapdragon X)** | 8-12 GB | 8-10 | ✅ Tested | P-cores are very fast; E-cores sufficient for preview |
| **Apple Silicon (M1/M2/M3)** | 8-32 GB | 8-10 | ✅ Tested | Native ARM64 support; fastest on ARM |
| **Raspberry Pi 5** | 4-8 GB | 4 | ⚠️ Slow | Preview only; recommend 8GB + SSD |

### Docker Resource Allocation (ARM64)

Since ARM is efficiency-focused, you can often use lower memory limits than x86:

```toml
# In docker-compose.yml (adjust to your system):

# Windows on Arm (Snapdragon, 12GB RAM)
memory: 8g          # Leave 4GB for OS
cpus: 8             # Most Snapdragon chips have 8-10 cores

# Apple Silicon (M2, 16GB RAM)
memory: 12g         # Leave 4GB for OS
cpus: 8             # 8 performance + efficiency cores

# Raspberry Pi 5 (8GB)
memory: 6g          # Leave 2GB for OS
cpus: 4             # All 4 cores at reduced clock
```

---

## Installation on Snapdragon (Windows on Arm)

### Step 1: Install Docker Desktop for Windows

1. Download [Docker Desktop for Windows on Arm](https://docs.docker.com/desktop/install/windows-install/)
2. Run installer and complete setup
3. Restart Windows

**Verify installation:**

```powershell
# In PowerShell:
docker --version
docker compose version

# Should both work without errors
```

### Step 2: Allocate Resources in Docker Desktop

1. Open **Docker Desktop**
2. Go to **Settings** → **Resources**
3. Set:
   - **Memory:** 8-10 GB (leave 2-4 GB for Windows)
   - **CPU:** 8 cores (all P-cores if available)
   - **Disk image size:** 50 GB

4. Click **Apply & Restart**

### Step 3: Clone & Setup Repository

```powershell
# Clone repo
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai

# Create data directories
mkdir -p data\input data\music data\output data\assets

# Run validation
.\scripts\validate-onboarding.sh
```

### Step 4: Build & Test

```powershell
# Build image (will use ARM64 base image automatically)
docker compose build

# Verify it works
docker compose run --rm montage-ai python -c "import montage_ai; print('✅ Ready')"
```

### Step 5: Create First Montage

```powershell
# Web UI (easiest)
docker compose up

# Then open http://localhost:8080 in Edge/Chrome

# OR via CLI
$env:QUALITY_PROFILE = "preview"
docker compose run --rm montage-ai .\montage-ai.sh run
```

---

## Installation on Apple Silicon (M1/M2/M3)

### Step 1: Install Docker Desktop for Mac

1. Download [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/) (ARM64 version)
2. Run `.dmg` file
3. Restart Mac

**Verify installation:**

```bash
docker --version
docker compose version
uname -m          # Should show "arm64"
```

### Step 2: Allocate Resources in Docker Desktop

1. Click **Docker** menu → **Settings**
2. Go to **Resources**
3. Set:
   - **Memory:** 10-12 GB (leave 4 GB for macOS)
   - **CPU:** 8 cores (usually all are P-cores)

4. Click **Apply & Restart**

### Step 3: Clone & Setup Repository

```bash
# Clone repo
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai

# Create data directories
mkdir -p data/input data/music data/output data/assets

# Run validation
./scripts/validate-onboarding.sh
```

### Step 4: Build & Test

```bash
# Build image (auto-detects ARM64 architecture)
docker compose build

# Verify it works
docker compose run --rm montage-ai python -c "import montage_ai; print('✅ Ready')"

# Test FFmpeg on ARM
docker compose run --rm montage-ai ffmpeg -f lavfi -i testsrc=s=320x240:d=1 -f null -
```

### Step 5: Create First Montage

```bash
# Web UI (recommended)
docker compose up &
# Open http://localhost:8080

# OR CLI
QUALITY_PROFILE=preview docker compose run --rm montage-ai ./montage-ai.sh run preview hitchcock
```

---

## Installation on Raspberry Pi 5

⚠️ **Note:** Raspberry Pi rendering will be **very slow**. Recommended for development/testing only.

### Requirements

- Raspberry Pi 5 with 8GB RAM
- USB-C SSD (not SD card: 10x faster I/O)
- 64-bit OS (Ubuntu 22.04 or Raspberry Pi OS 64-bit)

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
QUALITY_PROFILE=preview docker compose run --rm montage-ai ./montage-ai.sh run dynamic
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
FFMPEG_PRESET=fast docker compose run --rm montage-ai ./montage-ai.sh run

# Or use Preview mode (2x faster):
QUALITY_PROFILE=preview docker compose run --rm montage-ai ./montage-ai.sh run
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
