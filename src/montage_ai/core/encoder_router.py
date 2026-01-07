"""
Encoder Router - Intelligent GPU Resource Selection

Routes encoding jobs to the best available encoder based on:
1. Resource availability (CGPU → Local GPU → CPU)
2. Job characteristics (file size, duration, quality requirements)
3. Platform detection (Jetson → NVMPI, AMD → VAAPI, etc.)

Priority Order:
1. CGPU (Tesla T4 via cgpu) - Best quality, cloud
2. Local NVIDIA (NVENC/NVMPI) - Fast, local
3. Local AMD/Intel (VAAPI) - Good, local
4. CPU (libx264/libx265) - Universal fallback

Usage:
    from montage_ai.core.encoder_router import EncoderRouter

    router = EncoderRouter()
    config = router.get_best_encoder(
        file_size_mb=500,
        duration_sec=120,
        quality_profile="standard"
    )

    if config.use_cgpu:
        # Use CGPU encoding
        from montage_ai.cgpu_jobs import VideoEncodingJob
        job = VideoEncodingJob(input_path, output_path, **config.params)
        result = job.execute()
    else:
        # Use local FFmpeg with config.hw_config
        ...
"""

import os
import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from ..logger import logger
from ..cgpu_utils import is_cgpu_available
from .hardware import (
    HWConfig,
    get_best_hwaccel,
    get_hwaccel_by_type,
    _has_nvidia,
    _has_vaapi,
    _check_ffmpeg_encoder,
    _is_jetson,
    _is_intel_gpu,
)


class EncoderTier(Enum):
    """Encoder quality/speed tiers."""
    CGPU = "cgpu"          # Cloud GPU (Tesla T4)
    LOCAL_NVIDIA = "nvidia"  # Local NVIDIA (NVENC/NVMPI)
    LOCAL_AMD = "amd"       # Local AMD (VAAPI)
    LOCAL_INTEL = "intel"   # Local Intel (QSV/VAAPI)
    CPU = "cpu"             # Software encoding


@dataclass
class EncoderConfig:
    """Configuration for selected encoder."""
    tier: EncoderTier
    use_cgpu: bool
    hw_config: Optional[HWConfig]
    reason: str
    estimated_speed: float  # Multiplier vs realtime (e.g., 5.0 = 5x faster)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def encoder_name(self) -> str:
        """Human-readable encoder name."""
        if self.use_cgpu:
            return "CGPU (Tesla T4 NVENC)"
        if self.hw_config:
            return f"{self.hw_config.type.upper()} ({self.hw_config.encoder})"
        return "CPU (libx264)"


def _is_amd_gpu() -> bool:
    """Detect if AMD GPU is present."""
    amd_indicators = [
        "/sys/class/drm/card0/device/vendor",  # Check vendor ID
    ]

    for path in amd_indicators:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    vendor = f.read().strip()
                    # AMD vendor ID is 0x1002
                    if "0x1002" in vendor or "1002" in vendor:
                        return True
            except (IOError, PermissionError):
                pass

    # Check lspci output
    try:
        import subprocess
        result = subprocess.run(
            ["lspci", "-nn"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "AMD" in result.stdout and ("VGA" in result.stdout or "Display" in result.stdout):
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


class EncoderRouter:
    """
    Intelligent encoder selection based on available resources.

    Checks resources in priority order:
    1. CGPU (cloud) - if beneficial for the job
    2. Local NVIDIA - Jetson (NVMPI) or desktop (NVENC)
    3. Local AMD - VAAPI
    4. Local Intel - QSV or VAAPI
    5. CPU - always available
    """

    def __init__(self, test_encoders: bool = True):
        """
        Initialize router and detect available resources.

        Args:
            test_encoders: If True, test actual encoding capability (slower but accurate).
                          If False, only check encoder availability (faster but may be wrong).
        """
        self._cgpu_available: Optional[bool] = None
        self._local_gpu_type: Optional[str] = None
        self._local_gpu_tested: bool = False
        self._local_gpu_works: bool = False
        self._test_encoders = test_encoders
        self._platform_info = self._detect_platform()

    def _detect_platform(self) -> Dict[str, Any]:
        """Detect platform characteristics."""
        return {
            "is_jetson": _is_jetson(),
            "is_amd": _is_amd_gpu(),
            "is_intel": _is_intel_gpu(),
            "has_nvidia": _has_nvidia(),
            "has_vaapi": _has_vaapi(),
            "arch": platform.machine(),
            "system": platform.system(),
        }

    def _check_cgpu(self) -> bool:
        """
        Check if CGPU is available (cached).

        Sets CGPU_GPU_ENABLED=true to enable GPU checks.
        """
        if self._cgpu_available is None:
            # Ensure CGPU_GPU_ENABLED is set for the check
            os.environ.setdefault("CGPU_GPU_ENABLED", "true")
            os.environ.setdefault("CGPU_ENABLED", "true")
            self._cgpu_available = is_cgpu_available(require_gpu=True)
        return self._cgpu_available

    def _should_use_cgpu(
        self,
        file_size_mb: float,
        duration_sec: float,
        quality_profile: str,
        local_gpu_available: bool,
    ) -> bool:
        """
        Decide if CGPU encoding is beneficial.

        Uses CGPU when:
        1. Master quality requested (best encoder)
        2. No local GPU available AND file is large enough
        3. Explicitly requested via environment
        """
        # Central override via settings (env-aware)
        from ..config import get_settings
        force_cgpu = bool(get_settings().gpu.force_cgpu_encoding)
        if force_cgpu:
            return True

        # Master quality always uses CGPU for best results
        if quality_profile == "master":
            logger.info("Using CGPU for master quality encode")
            return True

        # If local GPU works, prefer it (lower latency)
        if local_gpu_available:
            return False

        # Calculate if CGPU is faster than CPU
        # Upload: ~10 MB/s, NVENC: ~10x realtime, Download: ~10 MB/s
        # CPU: ~0.3x realtime

        upload_time = file_size_mb / 10
        download_time = file_size_mb / 10
        nvenc_time = duration_sec / 10
        cpu_time = duration_sec / 0.3

        cgpu_total = upload_time + nvenc_time + download_time
        cpu_total = cpu_time

        # Use CGPU if 20% faster (accounting for overhead)
        if cgpu_total * 1.2 < cpu_total:
            logger.info(f"Using CGPU: estimated {cgpu_total:.0f}s vs CPU {cpu_total:.0f}s")
            return True

        return False

    def _test_encoder_works(self, hw_config: HWConfig) -> bool:
        """
        Test if an encoder actually works by doing a minimal encode.

        This catches cases where the encoder is available but runtime
        fails (e.g., QSV outside Docker, missing drivers).
        """
        import subprocess
        import tempfile

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "test.mp4")

                cmd = ["ffmpeg", "-y"]
                cmd.extend(hw_config.decoder_args)
                cmd.extend(["-f", "lavfi", "-i", "color=black:s=64x64:d=0.1"])

                if hw_config.hwupload_filter:
                    cmd.extend(["-vf", hw_config.hwupload_filter])

                cmd.extend(hw_config.encoder_args)

                # Add minimal quality settings
                if hw_config.type in ("nvenc", "nvmpi"):
                    cmd.extend(["-cq", "35"])
                elif hw_config.type == "vaapi":
                    cmd.extend(["-qp", "35"])
                elif hw_config.type == "qsv":
                    cmd.extend(["-global_quality", "35"])
                else:
                    cmd.extend(["-crf", "35"])

                cmd.append(output_path)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0 and os.path.exists(output_path):
                    logger.debug(f"Encoder {hw_config.encoder} test passed")
                    return True
                else:
                    logger.warning(f"Encoder {hw_config.encoder} test failed: {result.stderr[-200:]}")
                    return False

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Encoder {hw_config.encoder} test error: {e}")
            return False

    def _get_local_gpu_config(self, preferred_codec: str = "h264") -> Optional[HWConfig]:
        """
        Get best local GPU config based on platform.

        Priority:
        - Jetson: NVMPI (not NVENC!)
        - Desktop NVIDIA: NVENC
        - AMD: VAAPI
        - Intel: QSV or VAAPI

        If test_encoders is enabled, tests each encoder before returning.
        """
        # Use cached test result if available
        if self._local_gpu_tested:
            if self._local_gpu_works and self._local_gpu_type:
                return get_hwaccel_by_type(self._local_gpu_type, preferred_codec)
            return None

        info = self._platform_info

        def _try_encoder(accel_type: str, desc: str) -> Optional[HWConfig]:
            """Try an encoder, optionally testing it."""
            config = get_hwaccel_by_type(accel_type, preferred_codec)
            if not config:
                return None

            if self._test_encoders:
                if self._test_encoder_works(config):
                    logger.info(f"{desc}: {accel_type.upper()} encoder verified")
                    self._local_gpu_type = accel_type
                    self._local_gpu_works = True
                    self._local_gpu_tested = True
                    return config
                else:
                    logger.warning(f"{desc}: {accel_type.upper()} encoder failed test, skipping")
                    return None
            else:
                logger.debug(f"{desc}, using {accel_type.upper()}")
                return config

        # Jetson: Force NVMPI (NVENC doesn't work on Jetson)
        if info["is_jetson"]:
            config = _try_encoder("nvmpi", "Jetson detected")
            if config:
                return config
            config = _try_encoder("nvenc", "Jetson fallback")
            if config:
                return config

        # Desktop NVIDIA: Use NVENC
        if info["has_nvidia"] and not info["is_jetson"]:
            config = _try_encoder("nvenc", "NVIDIA desktop detected")
            if config:
                return config

        # AMD: Use VAAPI with radeonsi driver
        if info["is_amd"] and info["has_vaapi"]:
            os.environ.setdefault("LIBVA_DRIVER_NAME", "radeonsi")
            config = _try_encoder("vaapi", "AMD GPU detected")
            if config:
                return config

        # Intel: Try QSV first, then VAAPI
        if info["is_intel"]:
            config = _try_encoder("qsv", "Intel GPU detected")
            if config:
                return config

            os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")
            config = _try_encoder("vaapi", "Intel VAAPI fallback")
            if config:
                return config

        # Generic VAAPI fallback
        if info["has_vaapi"]:
            config = _try_encoder("vaapi", "Generic VAAPI")
            if config:
                return config

        # Mark as tested (no working GPU found)
        self._local_gpu_tested = True
        self._local_gpu_works = False

        return None

    def get_best_encoder(
        self,
        file_size_mb: float = 0,
        duration_sec: float = 0,
        quality_profile: str = "standard",
        preferred_codec: str = "h264",
        allow_cgpu: bool = True,
    ) -> EncoderConfig:
        """
        Select the best encoder for a job.

        Args:
            file_size_mb: Input file size in MB
            duration_sec: Video duration in seconds
            quality_profile: Quality profile (preview/standard/high/master)
            preferred_codec: Preferred codec (h264/hevc)
            allow_cgpu: Whether to consider CGPU

        Returns:
            EncoderConfig with selected encoder details
        """
        # Check local GPU first
        local_config = self._get_local_gpu_config(preferred_codec)
        local_gpu_available = local_config is not None

        # Check if CGPU should be used
        if allow_cgpu and self._check_cgpu():
            should_cgpu = self._should_use_cgpu(
                file_size_mb=file_size_mb,
                duration_sec=duration_sec,
                quality_profile=quality_profile,
                local_gpu_available=local_gpu_available,
            )

            if should_cgpu:
                # Pull centralized settings for CRF/preset mapping
                from ..config import get_settings as _get_settings
                _s = _get_settings()
                return EncoderConfig(
                    tier=EncoderTier.CGPU,
                    use_cgpu=True,
                    hw_config=None,
                    reason="CGPU selected (cloud Tesla T4)",
                    estimated_speed=10.0,
                    params={
                        "codec": preferred_codec,
                        # Derive quality from centralized settings
                        "quality": (_s.encoding.crf if quality_profile != "preview" else _s.preview.crf),
                        "preset": ("slow" if quality_profile == "master" else _s.encoding.preset),
                    }
                )

        # Use local GPU if available
        if local_config:
            tier = EncoderTier.LOCAL_NVIDIA
            if local_config.type == "vaapi":
                tier = EncoderTier.LOCAL_AMD if self._platform_info["is_amd"] else EncoderTier.LOCAL_INTEL
            elif local_config.type == "qsv":
                tier = EncoderTier.LOCAL_INTEL

            # Estimate speed based on encoder type
            speed_map = {
                "nvenc": 8.0,
                "nvmpi": 4.0,
                "vaapi": 5.0,
                "qsv": 6.0,
                "videotoolbox": 5.0,
            }
            speed = speed_map.get(local_config.type, 3.0)

            return EncoderConfig(
                tier=tier,
                use_cgpu=False,
                hw_config=local_config,
                reason=f"Local GPU: {local_config.type.upper()}",
                estimated_speed=speed,
            )

        # CPU fallback
        cpu_config = get_hwaccel_by_type("cpu", preferred_codec)
        return EncoderConfig(
            tier=EncoderTier.CPU,
            use_cgpu=False,
            hw_config=cpu_config,
            reason="CPU fallback (no GPU available)",
            estimated_speed=0.3,
        )

    def encode_video(
        self,
        input_path: str,
        output_path: str,
        quality_profile: str = "standard",
        codec: str = "h264",
        quality: int = 18,
        preset: str = "medium",
        filters: Optional[str] = None,
    ) -> Optional[str]:
        """
        Encode video using the best available encoder.

        Automatically routes to CGPU or local based on resources.

        Args:
            input_path: Input video path
            output_path: Output video path
            quality_profile: Quality profile
            codec: Video codec
            quality: Quality level (CRF/CQ)
            preset: Speed preset
            filters: Optional FFmpeg filters

        Returns:
            Output path if successful, None if failed
        """
        import os
        from pathlib import Path

        input_file = Path(input_path)
        if not input_file.exists():
            logger.error(f"Input file not found: {input_path}")
            return None

        # Get file info
        file_size_mb = input_file.stat().st_size / (1024 * 1024)

        # Estimate duration (rough, without probing)
        # Assume ~2 MB/s for compressed video
        duration_sec = file_size_mb / 2

        # Select encoder
        config = self.get_best_encoder(
            file_size_mb=file_size_mb,
            duration_sec=duration_sec,
            quality_profile=quality_profile,
            preferred_codec=codec,
        )

        logger.info(f"Encoding with {config.encoder_name} ({config.reason})")

        if config.use_cgpu:
            # Use CGPU encoding
            from ..cgpu_jobs import VideoEncodingJob

            job = VideoEncodingJob(
                input_path=input_path,
                output_path=output_path,
                codec=codec,
                quality=quality,
                preset=preset,
                filters=filters,
            )
            result = job.execute()

            if result.success:
                return result.output_path
            else:
                logger.warning(f"CGPU encoding failed: {result.error}")
                # Fall through to local fallback

        # Local encoding (GPU or CPU)
        return self._encode_local(
            input_path=input_path,
            output_path=output_path,
            hw_config=config.hw_config,
            quality=quality,
            preset=preset,
            filters=filters,
        )

    def _encode_local(
        self,
        input_path: str,
        output_path: str,
        hw_config: Optional[HWConfig],
        quality: int,
        preset: str,
        filters: Optional[str],
    ) -> Optional[str]:
        """Encode video using local FFmpeg."""
        import subprocess
        from ..ffmpeg_utils import build_ffmpeg_cmd

        cmd_args = []

        # Decoder args (hardware decoding if available)
        if hw_config and hw_config.decoder_args:
            cmd_args.extend(hw_config.decoder_args)

        # Input
        cmd_args.extend(["-i", input_path])

        # Filters
        if filters:
            if hw_config and hw_config.hwupload_filter:
                # Need hwupload for VAAPI
                cmd_args.extend(["-vf", f"{hw_config.hwupload_filter},{filters}"])
            else:
                cmd_args.extend(["-vf", filters])
        elif hw_config and hw_config.hwupload_filter:
            cmd_args.extend(["-vf", hw_config.hwupload_filter])

        # Encoder
        if hw_config:
            cmd_args.extend(hw_config.encoder_args)

            # Quality settings vary by encoder
            if hw_config.type in ("nvenc", "nvmpi"):
                cmd_args.extend(["-cq", str(quality), "-preset", preset])
            elif hw_config.type == "vaapi":
                cmd_args.extend(["-qp", str(quality)])
            elif hw_config.type == "qsv":
                cmd_args.extend(["-global_quality", str(quality), "-preset", preset])
            elif hw_config.type == "videotoolbox":
                cmd_args.extend(["-q:v", str(quality)])
            else:
                cmd_args.extend(["-crf", str(quality), "-preset", preset])
        else:
            cmd_args.extend(["-c:v", "libx264", "-crf", str(quality), "-preset", preset])

        # Audio
        cmd_args.extend(["-c:a", "aac", "-b:a", "192k"])

        # Output
        cmd_args.extend(["-movflags", "+faststart", output_path])

        cmd = build_ffmpeg_cmd(cmd_args)

        try:
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )

            if result.returncode == 0:
                return output_path
            else:
                logger.error(f"FFmpeg failed: {result.stderr[-500:]}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg encoding timed out")
            return None


# Module-level singleton for convenience
_router: Optional[EncoderRouter] = None


def get_encoder_router() -> EncoderRouter:
    """Get or create the global encoder router."""
    global _router
    if _router is None:
        _router = EncoderRouter()
    return _router


def smart_encode(
    input_path: str,
    output_path: str,
    quality_profile: str = "standard",
    **kwargs
) -> Optional[str]:
    """
    Convenience function for smart encoding.

    Uses the global EncoderRouter to select the best encoder.
    """
    router = get_encoder_router()
    return router.encode_video(
        input_path=input_path,
        output_path=output_path,
        quality_profile=quality_profile,
        **kwargs
    )
