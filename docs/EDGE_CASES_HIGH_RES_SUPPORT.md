# Edge Cases & High-Resolution Support

## Executive Summary

Montage AI ist aktuell für **Standard-Workflows (1080p-4K)** optimiert. Für extreme Szenarien wie **6K/8K RAW-Material** sind zusätzliche Maßnahmen erforderlich.

**Status:**
- ✅ **1080p-4K**: Vollständig unterstützt
- ⚠️ **6K (6144x3160)**: Funktioniert, aber Memory-intensive
- ❌ **8K+ RAW**: Erfordert Optimierungen

---

## 1. Aktuelle Limits & Constraints

### 1.1 Resolution Detection

**Datei:** `src/montage_ai/video_metadata.py`

```python
@dataclass
class VideoMetadata:
    width: int      # ✅ Keine hartcodierte Obergrenze
    height: int     # ✅ Keine hartcodierte Obergrenze
    fps: float
    duration: float
    codec: str      # ⚠️ Codec-spezifische Limits
    pix_fmt: str
    bitrate: int
```

**Status:**
- ✅ Metadata-Extraktion: Unterstützt **beliebige** Auflösungen
- ✅ `probe_metadata()`: Verwendet ffprobe (keine Resolution-Limits)
- ⚠️ `VideoMetadata.long_side`: Funktioniert bis 16K (65536 Pixel)

**Test-Coverage:**
```python
# tests/test_video_metadata.py
# ❌ Fehlt: Test für 6K/8K Resolutions
```

---

### 1.2 FFmpeg Configuration

**Datei:** `src/montage_ai/ffmpeg_config.py`

**Aktuelle Presets:**
```python
# Standard resolutions (hardcoded)
STANDARD_WIDTH_HORIZONTAL = 1920   # 1080p
STANDARD_HEIGHT_HORIZONTAL = 1080
STANDARD_WIDTH_VERTICAL = 1080
STANDARD_HEIGHT_VERTICAL = 1920

# Preview (fast feedback)
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360  # 360p
```

**Probleme:**
1. **Output-Profile hartkodiert auf 1080p/4K**
   - `determine_output_profile()` schnappt zu Standard-Resolutions
   - 6K-Input → 1080p Output (unnötige Downscaling)

2. **Keine 6K/8K Output-Presets**
   ```python
   # ❌ Fehlt in ffmpeg_config.py:
   STANDARD_WIDTH_6K = 6144
   STANDARD_HEIGHT_6K = 3160
   STANDARD_WIDTH_8K = 7680
   STANDARD_HEIGHT_8K = 4320
   ```

3. **H.264 Level 4.1 limitiert auf 4K30**
   ```python
   STANDARD_LEVEL = "4.1"  # Max 1080p60 or 4K30
   # ❌ 6K benötigt Level 5.2
   # ❌ 8K benötigt Level 6.2 (nur H.265/HEVC)
   ```

---

### 1.3 Memory Management

**Datei:** `src/montage_ai/config.py`

```python
class ProcessingSettings:
    batch_size: int = 5              # Default: 5 clips parallel
    # ⚠️ Für 6K RAW: 1 Frame = ~100MB → 5 Frames = 500MB
    # ❌ 8K RAW: 1 Frame = ~200MB → 5 Frames = 1GB (OOM risk)
    
    def get_adaptive_batch_size(self, low_memory: bool) -> int:
        """Adaptive batch sizing for low-memory systems."""
        if low_memory:
            return max(1, self.batch_size // 2)
        return self.batch_size
```

**Probleme:**
1. **Keine automatische Resolution-basierte Anpassung**
   ```python
   # ❌ Fehlt:
   def get_adaptive_batch_size_for_resolution(self, width: int, height: int) -> int:
       pixels = width * height
       if pixels > 33_177_600:  # 6K (6144x5400)
           return 1  # Process einzeln
       elif pixels > 8_294_400:  # 4K (3840x2160)
           return 2
       return self.batch_size
   ```

2. **Scene Detection Cache wächst mit Auflösung**
   - Histogramme: `192 floats * num_scenes * num_clips`
   - 6K: ~10x mehr Scenes als 1080p (höhere Detail-Dichte)

---

### 1.4 Codec Support

**Unterstützte Codecs:**

| Codec | Resolution | Bit Depth | RAW | Status |
|-------|------------|-----------|-----|--------|
| H.264 | ≤4K | 8-bit | ❌ | ✅ Standard |
| H.265/HEVC | ≤8K | 10-bit | ❌ | ✅ Unterstützt |
| ProRes 422 | ≤8K | 10-bit | ⚠️ Semi-RAW | ⚠️ Proxy-Generator |
| ProRes RAW | ≤8K | 12-bit | ✅ | ❌ Nicht getestet |
| DNxHD/DNxHR | ≤4K/8K | 8/10-bit | ❌ | ⚠️ Proxy-Generator |
| CinemaDNG | ≤8K | 16-bit | ✅ | ❌ Nicht unterstützt |
| BRAW | ≤8K | 12-bit | ✅ | ❌ Nicht unterstützt |

**Datei:** `src/montage_ai/proxy_generator.py`

```python
def generate_proxy(input_path: str, format: str = "h264", ...):
    """
    Formats: 'h264' (default), 'prores', 'dnxhr'
    ⚠️ ProRes/DNxHR nur für Proxy-Generierung, nicht für finale Montage
    """
    if format == "prores":
        # ProRes Proxy (profile 0) - bis 8K
        v_codec = ["-c:v", "prores_ks", "-profile:v", "proxy"]
    elif format == "dnxhr":
        # DNxHR LB - bis 4K
        v_codec = ["-c:v", "dnxhd", "-profile:v", "dnxhr_lb"]
```

**Probleme:**
1. **RAW-Formate ungetestet:**
   - ProRes RAW, BRAW, CinemaDNG nicht in Tests
   - Keine Debayering-Pipeline
   
2. **10-bit/12-bit Workflows fehlen:**
   - `pix_fmt` hartkodiert auf `yuv420p` (8-bit)
   - Keine HDR-Unterstützung (yuv420p10le, p010le)

---

## 2. Kritische Edge Cases

### 2.1 Extreme Resolutions

#### Test Case: 6K Landscape (6144x3160)
```python
# Input: RED Komodo 6K (6144x3160 @ 24fps)
metadata = probe_metadata("red_komodo_6k.r3d")
# Expected:
# - width: 6144
# - height: 3160
# - long_side: 6144
# - codec: "redcode"  # ⚠️ RED proprietary

# Problem 1: ffprobe unterstützt .r3d nicht direkt
# → Benötigt REDline SDK oder FFmpeg mit RED-Plugin

# Problem 2: Output-Profile schnappt zu 4K
profile = determine_output_profile([metadata])
# ❌ Actual: 3840x2160 (4K) statt 6144x3160
# → Unnötiges Downsampling
```

#### Test Case: 8K Portrait (4320x7680)
```python
# Input: 8K vertical TikTok challenge
metadata = probe_metadata("8k_vertical.mp4")
# Expected:
# - width: 4320
# - height: 7680
# - orientation: "vertical"

# Problem 1: H.264 Level 4.1 unterstützt max 4K30
# ❌ FFmpeg error: "Specified level is not supported by encoder"

# Problem 2: Memory explosion
# 8K frame (4320x7680) @ yuv420p = ~100MB uncompressed
# Scene detection: 1 scene = 1 histogram (192 floats) + 1 frame snapshot
# 100 scenes = ~10GB RAM für Histogramme allein
```

---

### 2.2 Exotic Codecs

#### Test Case: ProRes RAW
```python
# Input: ARRI Alexa ProRes RAW (4K @ 12-bit)
metadata = probe_metadata("alexa_prores_raw.mov")
# Expected:
# - codec: "prores_raw"
# - pix_fmt: "rgb48le" (16-bit RGB)

# Problem: FFmpeg benötigt --enable-libprores_raw
# ❌ Standard-Builds unterstützen ProRes RAW nicht
```

#### Test Case: Blackmagic RAW (BRAW)
```python
# Input: BMPCC 6K Pro (6144x3456 @ 12-bit)
metadata = probe_metadata("bmpcc_6k.braw")
# Problem: ffprobe erkennt .braw nicht
# → Benötigt Blackmagic RAW SDK
```

---

### 2.3 High Frame Rates

#### Test Case: 6K @ 120fps
```python
# Input: Slow-Motion (6K @ 120fps)
metadata = probe_metadata("6k_120fps.mp4")
# Expected:
# - fps: 120.0
# - duration: 10.0 (Echtzeit)

# Problem 1: Beat-Detection (librosa) skaliert mit FPS
# 120fps @ 10s = 1200 frames → 2x langsamer als 60fps

# Problem 2: Scene Detection
# PySceneDetect analysiert ALLE Frames (kein Keyframe-Skip bei HFR)
# 120fps = 2x mehr Frames als 60fps → 2x langsamer
```

---

### 2.4 Variable Frame Rate (VFR)

#### Test Case: Smartphone-Video mit VFR
```python
# Input: iPhone 14 Pro (1080p VFR 24-60fps)
metadata = probe_metadata("iphone_vfr.mov")
# Expected:
# - fps: 30.0 (average)
# - ⚠️ Actual FPS schwankt 24-60fps

# Problem: FFmpeg benötigt -vsync vfr oder fps=fps=30 Filter
# → Sonst Frame-Drops oder Duplikate
```

---

### 2.5 Extreme Aspect Ratios

#### Test Case: Anamorphic 2.39:1 (Cinemascope)
```python
# Input: 6K Anamorphic (6144x2571)
metadata = probe_metadata("6k_anamorphic.mp4")
# Expected:
# - aspect_ratio: 2.39
# - orientation: "horizontal"

# Problem: _snap_aspect_ratio() hat keine 2.39:1 Preset
# → Snapt zu 16:9 (fehlerhaft)
```

#### Test Case: Ultra-Wide 32:9
```python
# Input: Multi-Monitor-Setup (5120x1440)
metadata = probe_metadata("ultrawide_32_9.mp4")
# Expected:
# - aspect_ratio: 3.56
# - orientation: "horizontal"

# Problem: Smart Reframing (16:9 → 9:16) nicht für 32:9 getestet
```

---

## 3. Empfohlene Fixes

### 3.1 Resolution Constraints hinzufügen

**Datei:** `src/montage_ai/config.py`

```python
@dataclass
class ProcessingSettings:
    # Neue Felder:
    max_input_resolution: int = 8294400  # 4K (3840x2160)
    warn_threshold_resolution: int = 33177600  # 6K
    
    # ⚠️ Bei Überschreitung:
    # - Warnung loggen
    # - Batch-Size reduzieren
    # - Empfehlung: Proxy-Workflow
    
    def validate_resolution(self, width: int, height: int) -> bool:
        pixels = width * height
        if pixels > self.warn_threshold_resolution:
            logger.warning(
                f"⚠️ High resolution detected: {width}x{height} ({pixels/1e6:.1f}MP). "
                "Consider using proxy workflow for better performance."
            )
        if pixels > self.max_input_resolution * 2:  # 8K
            logger.error(
                f"❌ Resolution {width}x{height} exceeds safe limits. "
                "Use proxy workflow or reduce resolution."
            )
            return False
        return True
```

---

### 3.2 Adaptive Batch-Sizing basierend auf Resolution

**Datei:** `src/montage_ai/config.py`

```python
class ProcessingSettings:
    def get_adaptive_batch_size_for_resolution(
        self, width: int, height: int, low_memory: bool = False
    ) -> int:
        """
        Adaptive batch sizing basierend auf Input-Resolution.
        
        Resolution Brackets:
        - 1080p (2MP): batch_size = 5
        - 4K (8MP): batch_size = 2
        - 6K (19MP): batch_size = 1
        - 8K+ (33MP): ❌ Fehler (Proxy erforderlich)
        """
        pixels = width * height
        
        # 8K+ (33MP+): Nicht unterstützt
        if pixels > 33_177_600:
            raise ValueError(
                f"Resolution {width}x{height} ({pixels/1e6:.1f}MP) exceeds 8K limit. "
                "Please generate proxies first using: "
                "python -m montage_ai.proxy_generator --format h264 --scale 1920:-1"
            )
        
        # 6K (19-33MP): Einzelne Verarbeitung
        elif pixels > 19_660_800:
            return 1
        
        # 4K (8-19MP): Halbe Batch-Size
        elif pixels > 8_294_400:
            return max(1, self.batch_size // 2)
        
        # 1080p/2K: Standard
        else:
            if low_memory:
                return max(1, self.batch_size // 2)
            return self.batch_size
```

---

### 3.3 6K/8K Output-Presets

**Datei:** `src/montage_ai/ffmpeg_config.py`

```python
# =============================================================================
# High-Resolution Presets (6K/8K)
# =============================================================================

# 6K Presets
STANDARD_WIDTH_6K_HORIZONTAL = 6144
STANDARD_HEIGHT_6K_HORIZONTAL = 3160
STANDARD_WIDTH_6K_VERTICAL = 3160
STANDARD_HEIGHT_6K_VERTICAL = 6144

# 8K Presets (nur HEVC/H.265)
STANDARD_WIDTH_8K_HORIZONTAL = 7680
STANDARD_HEIGHT_8K_HORIZONTAL = 4320
STANDARD_WIDTH_8K_VERTICAL = 4320
STANDARD_HEIGHT_8K_VERTICAL = 7680

# H.265 Levels für High-Res
LEVEL_6K = "5.2"  # Max 6K60
LEVEL_8K = "6.2"  # Max 8K60 (nur HEVC)

@dataclass
class FFmpegConfig:
    def get_level_for_resolution(self, width: int, height: int, fps: float) -> str:
        """Determine H.264/H.265 level based on resolution and FPS."""
        pixels = width * height
        
        # 8K (7680x4320): Level 6.2 (nur HEVC)
        if pixels > 33_177_600:
            if "265" not in self.codec and "hevc" not in self.codec:
                raise ValueError(
                    f"8K resolution ({width}x{height}) requires HEVC/H.265. "
                    f"Current codec: {self.codec}"
                )
            return "6.2"
        
        # 6K (6144x3160): Level 5.2
        elif pixels > 19_660_800:
            return "5.2"
        
        # 4K (3840x2160): Level 5.1 (4K60) or 5.0 (4K30)
        elif pixels > 8_294_400:
            return "5.1" if fps > 30 else "5.0"
        
        # 1080p: Level 4.1 (1080p60) or 4.0 (1080p30)
        elif pixels > 2_073_600:
            return "4.1" if fps > 30 else "4.0"
        
        # SD/720p: Level 3.1
        else:
            return "3.1"
```

---

### 3.4 RAW Codec Warnings

**Datei:** `src/montage_ai/video_metadata.py`

```python
# Liste unterstützter RAW-Codecs
RAW_CODECS = {
    "prores_raw": "ProRes RAW (requires FFmpeg with --enable-libprores_raw)",
    "braw": "Blackmagic RAW (requires Blackmagic RAW SDK)",
    "redcode": "RED RAW (requires REDline SDK or FFmpeg plugin)",
    "cinemadng": "CinemaDNG (requires FFmpeg with --enable-libraw)",
}

def probe_metadata(video_path: str, ...) -> Optional[VideoMetadata]:
    """..."""
    codec = (stream.get("codec_name") or "unknown").lower()
    
    # Warnung bei RAW-Codecs
    if codec in RAW_CODECS:
        logger.warning(
            f"⚠️ RAW codec detected: {codec}\n"
            f"   {RAW_CODECS[codec]}\n"
            f"   Consider generating H.264/H.265 proxies for better compatibility."
        )
    
    # ...
```

---

### 3.5 Memory-Optimized Scene Detection

**Datei:** `src/montage_ai/scene_analysis.py`

```python
def detect_scenes(
    video_path: str,
    threshold: float = 30.0,
    max_resolution: Optional[int] = None  # Neu
) -> List[Tuple[float, float]]:
    """
    Detect scenes with optional resolution downscaling.
    
    Args:
        max_resolution: Max pixels für Analyse (z.B. 2073600 = 1080p)
                       6K/8K Videos werden automatisch skaliert
    """
    from scenedetect import detect, ContentDetector, AdaptiveDetector
    from scenedetect.video_stream import VideoStream
    
    # Probe video resolution
    metadata = probe_metadata(video_path)
    if not metadata:
        return []
    
    pixels = metadata.width * metadata.height
    
    # Automatisches Downscaling für 6K+
    if max_resolution and pixels > max_resolution:
        scale_factor = (max_resolution / pixels) ** 0.5
        target_width = int(metadata.width * scale_factor)
        
        logger.info(
            f"   📉 Downscaling {metadata.width}x{metadata.height} → "
            f"{target_width}x? for scene detection"
        )
        
        # FFmpeg scale filter
        video = VideoStream(
            video_path,
            scale=target_width  # PySceneDetect unterstützt scale parameter
        )
    else:
        video = VideoStream(video_path)
    
    # Rest der Implementierung...
```

---

## 4. Tests für Edge Cases

**Datei:** `tests/test_video_metadata_edge_cases.py` (NEU)

```python
"""
Edge Case Tests für Video Metadata - High-Res & Exotic Codecs
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from src.montage_ai.video_metadata import probe_metadata, determine_output_profile
from src.montage_ai.ffmpeg_config import FFmpegConfig


class TestHighResolutionSupport:
    """Tests für 6K/8K Resolutions."""
    
    @patch('src.montage_ai.video_metadata.subprocess.run')
    def test_6k_landscape_detection(self, mock_run):
        """6K Landscape (6144x3160) korrekt erkannt."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "width": 6144,
                    "height": 3160,
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "r_frame_rate": "24/1"
                }],
                "format": {"duration": "10.0"}
            })
        )
        
        metadata = probe_metadata("/test/6k_landscape.mp4")
        
        assert metadata is not None
        assert metadata.width == 6144
        assert metadata.height == 3160
        assert metadata.long_side == 6144
        assert metadata.orientation == "horizontal"
    
    @patch('src.montage_ai.video_metadata.subprocess.run')
    def test_8k_portrait_detection(self, mock_run):
        """8K Portrait (4320x7680) korrekt erkannt."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "width": 4320,
                    "height": 7680,
                    "codec_name": "hevc",  # 8K benötigt HEVC
                    "pix_fmt": "yuv420p10le",  # 10-bit
                    "r_frame_rate": "30/1"
                }],
                "format": {"duration": "10.0"}
            })
        )
        
        metadata = probe_metadata("/test/8k_portrait.mp4")
        
        assert metadata is not None
        assert metadata.width == 4320
        assert metadata.height == 7680
        assert metadata.long_side == 7680
        assert metadata.orientation == "vertical"
        assert metadata.codec == "hevc"  # H.265 erforderlich
    
    def test_6k_output_profile(self):
        """Output-Profile für 6K sollte 6K beibehalten."""
        # ⚠️ Aktuell FAIL: Snappt zu 4K
        # TODO: Fix in determine_output_profile()
        pass
    
    def test_8k_requires_hevc(self):
        """8K Output benötigt HEVC, nicht H.264."""
        config = FFmpegConfig(codec="libx264", width=7680, height=4320)
        
        # ❌ Sollte Fehler werfen
        with pytest.raises(ValueError, match="8K resolution.*requires HEVC"):
            config.get_level_for_resolution(7680, 4320, 30.0)


class TestExoticCodecs:
    """Tests für RAW & Professional Codecs."""
    
    @patch('src.montage_ai.video_metadata.subprocess.run')
    def test_prores_raw_detection(self, mock_run):
        """ProRes RAW korrekt erkannt mit Warnung."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "width": 4096,
                    "height": 2160,
                    "codec_name": "prores_raw",
                    "pix_fmt": "rgb48le",  # 16-bit RGB
                    "r_frame_rate": "24/1"
                }],
                "format": {"duration": "10.0"}
            })
        )
        
        with pytest.warns(UserWarning, match="RAW codec detected"):
            metadata = probe_metadata("/test/prores_raw.mov")
        
        assert metadata.codec == "prores_raw"
        assert metadata.pix_fmt == "rgb48le"
    
    @patch('src.montage_ai.video_metadata.subprocess.run')
    def test_redcode_unsupported(self, mock_run):
        """RED RAW Codec warnt über fehlende Unterstützung."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "width": 6144,
                    "height": 3160,
                    "codec_name": "redcode",  # RED proprietary
                    "pix_fmt": "rgb48le",
                    "r_frame_rate": "24/1"
                }],
                "format": {"duration": "10.0"}
            })
        )
        
        with pytest.warns(UserWarning, match="RED RAW.*REDline SDK"):
            metadata = probe_metadata("/test/red_komodo.r3d")


class TestExtremeFPS:
    """Tests für High Frame Rate (HFR) Videos."""
    
    @patch('src.montage_ai.video_metadata.subprocess.run')
    def test_120fps_detection(self, mock_run):
        """120fps korrekt erkannt."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "width": 3840,
                    "height": 2160,
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "r_frame_rate": "120/1"
                }],
                "format": {"duration": "10.0"}
            })
        )
        
        metadata = probe_metadata("/test/4k_120fps.mp4")
        
        assert metadata.fps == 120.0
        # ⚠️ TODO: Test batch-size Anpassung für HFR


class TestExtremeAspectRatios:
    """Tests für ungewöhnliche Aspect Ratios."""
    
    @patch('src.montage_ai.video_metadata.subprocess.run')
    def test_anamorphic_2_39_1(self, mock_run):
        """Anamorphic 2.39:1 (Cinemascope)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "width": 6144,
                    "height": 2571,  # 2.39:1
                    "codec_name": "prores",
                    "pix_fmt": "yuv422p10le",
                    "r_frame_rate": "24/1"
                }],
                "format": {"duration": "10.0"}
            })
        )
        
        metadata = probe_metadata("/test/anamorphic.mov")
        
        assert metadata.aspect_ratio == pytest.approx(2.39, rel=0.01)
        # ⚠️ TODO: _snap_aspect_ratio() sollte 2.39:1 Preset haben
    
    @patch('src.montage_ai.video_metadata.subprocess.run')
    def test_ultrawide_32_9(self, mock_run):
        """Ultra-Wide 32:9 (5120x1440)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "width": 5120,
                    "height": 1440,  # 32:9
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "r_frame_rate": "60/1"
                }],
                "format": {"duration": "10.0"}
            })
        )
        
        metadata = probe_metadata("/test/ultrawide.mp4")
        
        assert metadata.aspect_ratio == pytest.approx(3.56, rel=0.01)
        assert metadata.orientation == "horizontal"


class TestMemoryConstraints:
    """Tests für Memory-Limits bei High-Res."""
    
    def test_6k_reduces_batch_size(self):
        """6K-Input reduziert Batch-Size automatisch."""
        from src.montage_ai.config import get_settings
        
        settings = get_settings()
        batch_size = settings.processing.get_adaptive_batch_size_for_resolution(
            width=6144, height=3160, low_memory=False
        )
        
        assert batch_size == 1  # Einzelne Verarbeitung
    
    def test_8k_raises_error(self):
        """8K-Input wirft Fehler ohne Proxy."""
        from src.montage_ai.config import get_settings
        
        settings = get_settings()
        
        with pytest.raises(ValueError, match="exceeds 8K limit.*proxies"):
            settings.processing.get_adaptive_batch_size_for_resolution(
                width=7680, height=4320, low_memory=False
            )
```

---

## 5. Deployment-Checkliste für High-Res

### 5.1 System Requirements

**Für 6K Workflows:**
- ✅ **CPU**: 8+ Kerne empfohlen
- ✅ **RAM**: 32GB minimum (64GB empfohlen)
- ⚠️ **GPU**: NVENC/HEVC Encoder (Level 5.2 Support)
- ✅ **Storage**: SSD/NVMe (6K = ~30GB/min uncompressed)

**Für 8K Workflows:**
- ❌ **CPU**: 16+ Kerne erforderlich
- ❌ **RAM**: 64GB minimum (128GB empfohlen)
- ❌ **GPU**: Professionelle GPU (NVIDIA RTX A6000, A100)
- ❌ **Storage**: NVMe RAID (8K = ~120GB/min uncompressed)

---

### 5.2 Environment Variables

```bash
# Für 6K Material
export LOW_MEMORY_MODE=false
export FFMPEG_THREADS=0  # Auto (nutzt alle Kerne)
export BATCH_SIZE=1  # Einzelne Verarbeitung

# Für HEVC/H.265 Output
export OUTPUT_CODEC=libx265
export FFMPEG_PRESET=slow  # Bessere Kompression für 6K

# GPU Encoding (empfohlen)
export FFMPEG_HWACCEL=nvenc  # NVIDIA
export OUTPUT_CODEC=hevc_nvenc  # HEVC GPU
```

---

### 5.3 Proxy Workflow (empfohlen für 6K+)

```bash
# Schritt 1: Proxies generieren (1080p H.264)
python -m montage_ai.proxy_generator \
    --input /data/input/*.mp4 \
    --output /data/proxies \
    --format h264 \
    --scale 1920:-1 \
    --preset fast

# Schritt 2: Montage mit Proxies
./montage-ai.sh run --input /data/proxies

# Schritt 3: Timeline-Export für Conform
python -m montage_ai.timeline_exporter \
    --input /data/output/final.mp4 \
    --original-media /data/input \
    --output /data/timelines/project.fcpxml

# Schritt 4: Final Conform in Final Cut Pro/Resolve
# → Nutzt Original 6K-Files statt Proxies
```

---

## 6. Zusammenfassung & Empfehlungen

### 6.1 Aktuelle Grenzen

| Feature | 1080p | 4K | 6K | 8K |
|---------|-------|-----|-----|-----|
| Metadata Detection | ✅ | ✅ | ✅ | ✅ |
| Scene Detection | ✅ | ✅ | ⚠️ Langsam | ❌ OOM |
| Beat Sync | ✅ | ✅ | ✅ | ⚠️ Langsam |
| Auto-Reframe | ✅ | ✅ | ⚠️ Ungetestet | ❌ |
| H.264 Output | ✅ | ✅ | ❌ Level 4.1 | ❌ |
| HEVC Output | ✅ | ✅ | ✅ Level 5.2 | ⚠️ Level 6.2 |
| RAW Support | N/A | ⚠️ ProRes | ⚠️ ProRes | ❌ |

---

### 6.2 Empfohlene Implementierungen (Priorität)

**🔴 Kritisch (Phase 4):**
1. **Adaptive Batch-Sizing** (`get_adaptive_batch_size_for_resolution`)
2. **Level Auto-Detection** (`get_level_for_resolution`)
3. **6K Output-Presets** (6144x3160, Level 5.2)
4. **RAW Codec Warnings** (ProRes RAW, BRAW, RED)

**🟡 Wichtig (Phase 5):**
5. **Scene Detection Downscaling** (`max_resolution` Parameter)
6. **10-bit/HDR Support** (`yuv420p10le`, `p010le`)
7. **VFR Handling** (`-vsync vfr`)
8. **Anamorphic Presets** (2.39:1, 2.35:1)

**🟢 Nice-to-Have (Future):**
9. **8K Support** (mit obligatorischem Proxy-Workflow)
10. **CinemaDNG/BRAW Debayering** (via FFmpeg plugins)
11. **AI Upscaling für Proxies** (Real-ESRGAN, Topaz Video AI)

---

### 6.3 Quick-Wins (Sofort umsetzbar)

```python
# 1. Warnung bei 6K+
if metadata.long_side > 5000:
    logger.warning("⚠️ 6K+ detected. Consider proxy workflow.")

# 2. HEVC für 6K+
if metadata.long_side > 5000 and config.codec == "libx264":
    logger.warning("⚠️ 6K requires HEVC. Switching to libx265...")
    config.codec = "libx265"
    config.level = "5.2"

# 3. Batch-Size = 1 für 6K
if metadata.width * metadata.height > 19_660_800:  # 6K
    batch_size = 1
```

---

**Dokumentation erstellt:** 2025-01-07
**Status:** 🟡 In Review (Tests fehlen)
**Nächster Schritt:** Implementierung der Kritischen Fixes (Phase 4)
