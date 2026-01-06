# Hardware Encoding Strategy - Heterogeneous Cluster

**Status:** Active Analysis
**Date:** 2026-01-06
**Priority:** High (Performance Critical)

---

## Executive Summary

Montage AI läuft in einem heterogenen K3s Cluster mit verschiedenen GPU-Typen:
- **AMD VAAPI** (codeai-fluxibriserver)
- **NVIDIA NVMPI** (Jetson)
- **Tesla T4** (CGPU Cloud)
- **CPU-only** (ARM Workers)

**Problem:** Hardware-Encoding funktioniert derzeit nicht zuverlässig:
1. QSV außerhalb Container: MFX Session Error -9
2. Docker-Image ohne GPU-Bibliotheken kompiliert
3. CGPU nur für Upscaling genutzt, nicht für Encoding
4. VAAPI-Erkennung fragil (Driver-Hints fehlen oft)

---

## Cluster Hardware Matrix

| Node | Arch | GPU | Encoder | Status | Priority |
|------|------|-----|---------|--------|----------|
| codeai-fluxibriserver | x86_64 | AMD RDNA | VAAPI | ⚠️ Fragil | 1 |
| codeai-fluxibriserver | x86_64 | Intel iGPU | QSV | ❌ MFX Error | 2 |
| codeaijetson-desktop | ARM64 | Tegra | NVMPI | ⚠️ Priority Issue | 3 |
| CGPU (Cloud) | Remote | Tesla T4 | NVENC | ✅ Verfügbar, nicht genutzt | 1 |
| raspillm8850 | ARM | - | CPU | ✅ Fallback | 4 |
| pi-worker-1 | ARM | - | CPU | ✅ Fallback | 4 |

---

## Root Cause Analysis

### 1. QSV Failure (MFX Session Error -9)

```
[h264_qsv @ 0x...] Error creating a MFX session: -9
Device creation failed: -1313558101
No device available for decoder: device type qsv needed for codec h264_qsv
```

**Ursache:** Intel Media SDK nicht korrekt initialisiert.

**Mögliche Gründe:**
- `LIBMFX_PLUGINS_PATH` nicht gesetzt
- `/dev/dri/renderD128` Permissions (außerhalb Container)
- iGPU von AMD GPU überlagert (Multi-GPU Konflikt)
- Intel Media SDK Libraries fehlen im System

**Diagnostik:**
```bash
# Check Intel GPU
ls -la /dev/dri/
vainfo --display drm --device /dev/dri/renderD128
# Check QSV
ffmpeg -init_hw_device qsv=hw -filter_hw_device hw -f lavfi -i color=black:s=64x64:d=0.1 -c:v h264_qsv -f null -
```

### 2. Docker Image ohne GPU Support

Das aktuelle `Dockerfile` basiert auf `python:3.10-slim-bookworm`:
- Kein CUDA Toolkit
- Kein ROCm (AMD)
- Kein Intel Media SDK
- FFmpeg ohne Hardware-Encoder kompiliert

**Lösung:** Multi-Stage Build mit GPU Libraries.

### 3. CGPU Underutilization

CGPU (Tesla T4) wird nur für:
- Real-ESRGAN Upscaling
- Whisper Transcription
- Gemini LLM

**Nicht genutzt für:**
- Video Encoding (H.264/HEVC NVENC)
- Beat Detection (GPU-accelerated)
- Scene Detection (GPU-accelerated)

---

## Lösungsstrategie

### Phase 1: CGPU Encoding Offload (Kurzfristig)

**Ziel:** Tesla T4 für schwere Encoding-Jobs nutzen.

**Architektur:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Web UI Job    │────▶│   Job Router    │────▶│  CGPU Encoder   │
│   (K3s Pod)     │     │   (Decision)    │     │  (Tesla T4)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  Local Fallback │
                        │  (VAAPI/CPU)    │
                        └─────────────────┘
```

**Implementation:**

```python
# src/montage_ai/cgpu_jobs/encoding.py (NEU)
class VideoEncodingJob(CGPUJob):
    """Offload video encoding to Cloud GPU."""

    def encode(self, input_path: str, output_path: str,
               codec: str = "h264", quality: int = 18) -> str:
        """
        Encode video on Tesla T4 with NVENC.

        Falls back to local if CGPU unavailable.
        """
        script = f'''
import subprocess
import sys

cmd = [
    "ffmpeg", "-y",
    "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
    "-i", "{input_path}",
    "-c:v", "h264_nvenc",
    "-preset", "p4",  # Quality preset
    "-cq", "{quality}",  # Constant quality
    "-c:a", "aac", "-b:a", "192k",
    "{output_path}"
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"NVENC failed: {{result.stderr}}", file=sys.stderr)
    sys.exit(1)
print(f"Encoded: {output_path}")
'''
        return self._run_cgpu_script(script)
```

**Routing Logic:**
```python
def should_use_cgpu_encoding(file_size_mb: float, duration_sec: float) -> bool:
    """Decide if CGPU encoding is worth the upload overhead."""
    # CGPU macht Sinn bei:
    # - Großen Dateien (>100MB) wo GPU-Speed den Upload amortisiert
    # - Langen Videos (>60s) mit komplexen Filtern
    # - Wenn lokale GPU nicht verfügbar

    local_gpu = get_best_gpu_encoder()
    if local_gpu and local_gpu != "cpu":
        return False  # Lokale GPU bevorzugen

    # Upload ~10MB/s, NVENC ~5x schneller als CPU
    upload_time = file_size_mb / 10
    cpu_encode_time = duration_sec * 2  # ~0.5x realtime
    nvenc_encode_time = duration_sec * 0.1  # ~10x realtime

    return (upload_time + nvenc_encode_time) < cpu_encode_time
```

### Phase 2: GPU-Enabled Docker Image (Mittelfristig)

**Ziel:** Docker Image mit nativer GPU-Unterstützung.

**Multi-Arch GPU Dockerfile:**

```dockerfile
# Dockerfile.gpu
ARG TARGETARCH=amd64

# === AMD64 Base (VAAPI + QSV) ===
FROM ubuntu:22.04 AS base-amd64
RUN apt-get update && apt-get install -y \
    ffmpeg \
    intel-media-va-driver-non-free \
    libva-dev \
    vainfo \
    && rm -rf /var/lib/apt/lists/*
ENV LIBVA_DRIVER_NAME=iHD

# === ARM64 Base (NVMPI für Jetson) ===
FROM nvcr.io/nvidia/l4t-base:r35.3.1 AS base-arm64
RUN apt-get update && apt-get install -y \
    ffmpeg \
    nvidia-l4t-multimedia \
    && rm -rf /var/lib/apt/lists/*

# === Final Stage ===
FROM base-${TARGETARCH} AS final
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/src/
```

**K3s Kustomize Overlay:**
```yaml
# deploy/k3s/overlays/gpu-native/kustomization.yaml
resources:
  - ../../base
patches:
  - target:
      kind: Deployment
      name: montage-ai-web
    patch: |
      - op: replace
        path: /spec/template/spec/containers/0/image
        value: 192.168.1.12:5000/montage-ai:gpu-native
      - op: add
        path: /spec/template/spec/containers/0/securityContext
        value:
          privileged: true  # For /dev/dri access
```

### Phase 3: Distributed Encoding Queue (Langfristig)

**Ziel:** Intelligente Job-Verteilung basierend auf Node-Capabilities.

**Architektur:**
```
┌──────────────┐
│   Web UI     │
│   Submit     │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌─────────────────────────────────────┐
│  Redis Queue │────▶│         Job Scheduler               │
│  (montage-ai)│     │  ┌─────────┬─────────┬───────────┐  │
└──────────────┘     │  │  AMD    │ Jetson  │   CGPU    │  │
                     │  │ Worker  │ Worker  │  Worker   │  │
                     │  │(VAAPI)  │(NVMPI)  │ (NVENC)   │  │
                     │  └─────────┴─────────┴───────────┘  │
                     └─────────────────────────────────────┘
```

**Node Capability Labels:**
```yaml
# kubectl label node codeai-fluxibriserver montage.ai/encoder=vaapi
# kubectl label node codeaijetson-desktop montage.ai/encoder=nvmpi
apiVersion: v1
kind: ConfigMap
metadata:
  name: encoder-capabilities
data:
  codeai-fluxibriserver: |
    encoder: vaapi
    codec: h264_vaapi
    speed: 3x  # vs realtime
    quality: good
  codeaijetson-desktop: |
    encoder: nvmpi
    codec: h264_nvmpi
    speed: 2x
    quality: good
  cgpu: |
    encoder: nvenc
    codec: h264_nvenc
    speed: 10x
    quality: excellent
    cost: cloud  # Upload overhead
```

**Job Scheduler Logic:**
```python
class EncoderScheduler:
    """Route encoding jobs to best available encoder."""

    def select_encoder(self, job: EncodingJob) -> str:
        """
        Select best encoder based on:
        1. Local GPU availability (lowest latency)
        2. Job size (CGPU for large jobs)
        3. Quality requirements (CGPU for master quality)
        4. Queue depth (load balancing)
        """
        capabilities = self.get_node_capabilities()

        # Priorität: Lokal > CGPU > CPU
        if job.quality_profile == "master":
            return "cgpu"  # Beste Qualität

        if self.local_gpu_available():
            return self.local_encoder_type

        if job.file_size_mb > 100 and self.cgpu_available():
            return "cgpu"

        return "cpu"  # Fallback
```

---

## Immediate Action Items

### 1. CGPU Encoding aktivieren

```bash
# Neues Modul erstellen
touch src/montage_ai/cgpu_jobs/encoding.py

# In MontageBuilder integrieren
# Bei final render: if should_use_cgpu_encoding() -> cgpu_encode()
```

### 2. VAAPI-Erkennung verbessern

```python
# hardware.py: Robustere Driver-Erkennung
def _has_vaapi() -> bool:
    """Mit explizitem Timeout und besseren Hints."""
    drivers = ["radeonsi", "iHD", "i965"]

    for driver in drivers:
        env = {**os.environ, "LIBVA_DRIVER_NAME": driver}
        try:
            result = subprocess.run(
                ["vainfo", "--display", "drm"],
                env=env,
                capture_output=True,
                timeout=5  # Timeout hinzufügen
            )
            if result.returncode == 0:
                os.environ["LIBVA_DRIVER_NAME"] = driver
                logger.info(f"VAAPI: Using {driver} driver")
                return True
        except subprocess.TimeoutExpired:
            continue

    return False
```

### 3. QSV-Diagnose

```bash
# Prüfen ob Intel GPU vorhanden
lspci | grep -i intel | grep -i vga

# libmfx installieren
sudo apt install libmfx1 libmfx-tools

# QSV testen
ffmpeg -init_hw_device qsv=hw -f lavfi -i testsrc=duration=1:size=64x64 -c:v h264_qsv -f null -
```

---

## Success Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| GPU Encoding Success Rate | ~20% | >90% | CGPU Fallback |
| Encoding Speed (1080p) | 0.5x RT | >3x RT | GPU Offload |
| Job Failure Rate | ~30% | <5% | Smart Routing |
| Cloud Cost | $0 | <$10/mo | Selective CGPU |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| CGPU Rate Limits | High | Local fallback, batch jobs |
| Docker Image Size (+2GB) | Medium | Multi-stage builds, layer caching |
| Network Latency | Medium | Threshold-based routing |
| Driver Incompatibility | Low | Test matrix, CI/CD validation |

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 (CGPU) | 1-2 Tage | `cgpu_jobs/encoding.py` |
| Phase 2 (Docker) | 3-5 Tage | `Dockerfile.gpu`, multi-arch |
| Phase 3 (Queue) | 2-3 Wochen | Redis Queue, Scheduler |

---

## References

- [FFmpeg Hardware Encoding](https://trac.ffmpeg.org/wiki/HWAccelIntro)
- [NVIDIA NVENC SDK](https://developer.nvidia.com/video-codec-sdk)
- [Intel Media SDK](https://www.intel.com/content/www/us/en/developer/tools/vpl/overview.html)
- [VAAPI on AMD](https://wiki.archlinux.org/title/Hardware_video_acceleration)
