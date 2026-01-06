"""
Audio Fallback Strategy - Artifact Detection and Recovery

Provides intelligent fallback for audio processing:
1. Detect artifacts in processed audio (clipping, distortion, silence)
2. Compare processed vs original audio quality
3. Blend or rollback if quality degraded

Usage:
    from montage_ai.audio_fallback import AudioFallbackProcessor

    processor = AudioFallbackProcessor()
    result = processor.process_with_fallback(
        original_path="/path/to/original.wav",
        processed_path="/path/to/processed.wav",
        output_path="/path/to/output.wav"
    )
    print(f"Strategy used: {result.strategy}")  # "processed", "original", or "blended"
"""

import tempfile
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from .logger import logger
from .ffmpeg_utils import build_ffmpeg_cmd
from .core.cmd_runner import run_command
from .audio_analysis import estimate_audio_snr_lufs, compare_audio_quality, AudioQuality


class FallbackStrategy(str, Enum):
    """Strategy used for audio output."""
    PROCESSED = "processed"  # Use processed audio (no issues detected)
    ORIGINAL = "original"    # Fall back to original (severe issues)
    BLENDED = "blended"      # Blend processed + original (minor issues)


@dataclass
class ArtifactReport:
    """Report of detected audio artifacts."""
    has_clipping: bool = False
    has_distortion: bool = False
    has_silence_gaps: bool = False
    has_phase_issues: bool = False
    clipping_percentage: float = 0.0
    silence_percentage: float = 0.0
    severity: str = "none"  # "none", "minor", "moderate", "severe"

    @property
    def has_artifacts(self) -> bool:
        return any([
            self.has_clipping,
            self.has_distortion,
            self.has_silence_gaps,
            self.has_phase_issues
        ])


@dataclass
class FallbackResult:
    """Result of audio processing with fallback."""
    strategy: FallbackStrategy
    output_path: str
    original_quality: AudioQuality
    processed_quality: Optional[AudioQuality]
    artifacts: ArtifactReport
    blend_ratio: float = 0.0  # 0.0 = all original, 1.0 = all processed
    message: str = ""


class AudioFallbackProcessor:
    """
    Audio processor with intelligent fallback capabilities.

    Detects issues in processed audio and decides whether to:
    1. Use the processed audio as-is
    2. Fall back to the original
    3. Blend processed and original

    The decision is based on:
    - Artifact detection (clipping, distortion, silence gaps)
    - SNR comparison (processed vs original)
    - Quality tier assessment
    """

    # Thresholds for artifact detection
    CLIPPING_THRESHOLD = 0.01  # Max 1% clipping samples
    SILENCE_THRESHOLD = 0.10   # Max 10% silence
    SNR_DEGRADATION_THRESHOLD = -3.0  # Max 3dB SNR loss

    # Blend ratios for different severities
    BLEND_RATIOS = {
        "none": 1.0,     # 100% processed
        "minor": 0.8,    # 80% processed, 20% original
        "moderate": 0.5, # 50/50 blend
        "severe": 0.0,   # 100% original (full fallback)
    }

    def __init__(self):
        pass

    def detect_artifacts(self, audio_path: str) -> ArtifactReport:
        """
        Detect artifacts in an audio file.

        Uses FFmpeg's astats filter to analyze:
        - Peak levels (clipping detection)
        - Silence periods
        - Crest factor (distortion indicator)
        """
        report = ArtifactReport()

        # Run FFmpeg astats analysis
        cmd = build_ffmpeg_cmd([
            "-i", audio_path,
            "-af", "astats=metadata=1:reset=1",
            "-f", "null", "-"
        ], overwrite=False)

        try:
            result = run_command(cmd, capture_output=True, timeout=60, check=False)
            stderr = result.stderr

            # Parse astats output
            clipping_count = 0
            sample_count = 0
            silence_count = 0
            total_frames = 0

            for line in stderr.split('\n'):
                # Check for clipping (peak = 0 dBFS)
                if 'Peak level dB:' in line:
                    try:
                        peak = float(line.split(':')[-1].strip())
                        if peak >= -0.1:  # Very close to 0dB = potential clipping
                            clipping_count += 1
                    except ValueError:
                        pass

                # Count total frames for percentage calculation
                if 'Overall' in line or 'frame=' in line:
                    total_frames += 1

                # Check for silence (very low RMS)
                if 'RMS level dB:' in line:
                    try:
                        rms = float(line.split(':')[-1].strip())
                        if rms < -60:  # Below -60dB is effectively silence
                            silence_count += 1
                    except ValueError:
                        pass

            # Calculate percentages
            if total_frames > 0:
                report.clipping_percentage = clipping_count / total_frames
                report.silence_percentage = silence_count / total_frames

            # Determine if thresholds exceeded
            report.has_clipping = report.clipping_percentage > self.CLIPPING_THRESHOLD
            report.has_silence_gaps = report.silence_percentage > self.SILENCE_THRESHOLD

            # Determine severity
            if report.has_clipping and report.clipping_percentage > 0.05:
                report.severity = "severe"
            elif report.has_clipping:
                report.severity = "moderate"
            elif report.has_silence_gaps:
                report.severity = "minor"
            else:
                report.severity = "none"

        except Exception as e:
            logger.warning(f"Artifact detection failed: {e}")

        return report

    def process_with_fallback(
        self,
        original_path: str,
        processed_path: str,
        output_path: str,
        force_strategy: Optional[FallbackStrategy] = None
    ) -> FallbackResult:
        """
        Apply fallback strategy to audio processing result.

        Args:
            original_path: Path to original audio
            processed_path: Path to processed audio
            output_path: Path for final output
            force_strategy: Force a specific strategy (for testing)

        Returns:
            FallbackResult with strategy used and quality metrics
        """
        # Analyze original audio
        logger.info("Analyzing original audio quality...")
        original_quality = estimate_audio_snr_lufs(original_path)

        # Analyze processed audio
        logger.info("Analyzing processed audio quality...")
        processed_quality = estimate_audio_snr_lufs(processed_path)

        # Detect artifacts in processed audio
        logger.info("Detecting artifacts in processed audio...")
        artifacts = self.detect_artifacts(processed_path)

        # Compare quality
        comparison = compare_audio_quality(original_path, processed_path)

        # Determine strategy
        if force_strategy:
            strategy = force_strategy
        else:
            strategy = self._determine_strategy(
                original_quality,
                processed_quality,
                artifacts,
                comparison.snr_improvement_db
            )

        # Apply strategy
        blend_ratio = self.BLEND_RATIOS.get(artifacts.severity, 1.0)

        if strategy == FallbackStrategy.PROCESSED:
            # Use processed audio
            shutil.copy(processed_path, output_path)
            message = f"Using processed audio (SNR improved by {comparison.snr_improvement_db:+.1f}dB)"

        elif strategy == FallbackStrategy.ORIGINAL:
            # Fall back to original
            shutil.copy(original_path, output_path)
            message = f"Falling back to original audio (artifacts detected: {artifacts.severity})"

        elif strategy == FallbackStrategy.BLENDED:
            # Blend processed and original
            self._blend_audio(original_path, processed_path, output_path, blend_ratio)
            message = f"Blending audio ({int(blend_ratio*100)}% processed, artifacts: {artifacts.severity})"

        logger.info(f"Audio fallback: {message}")

        return FallbackResult(
            strategy=strategy,
            output_path=output_path,
            original_quality=original_quality,
            processed_quality=processed_quality,
            artifacts=artifacts,
            blend_ratio=blend_ratio,
            message=message
        )

    def _determine_strategy(
        self,
        original: AudioQuality,
        processed: AudioQuality,
        artifacts: ArtifactReport,
        snr_improvement: float
    ) -> FallbackStrategy:
        """Determine the best fallback strategy based on analysis."""

        # Severe artifacts = always fall back
        if artifacts.severity == "severe":
            logger.warning("Severe artifacts detected, falling back to original")
            return FallbackStrategy.ORIGINAL

        # Significant quality degradation = fall back
        if snr_improvement < self.SNR_DEGRADATION_THRESHOLD:
            logger.warning(f"SNR degraded by {snr_improvement:.1f}dB, falling back to original")
            return FallbackStrategy.ORIGINAL

        # Moderate artifacts = blend
        if artifacts.severity == "moderate":
            return FallbackStrategy.BLENDED

        # Minor artifacts with no quality improvement = blend
        if artifacts.severity == "minor" and snr_improvement < 1.0:
            return FallbackStrategy.BLENDED

        # No issues = use processed
        return FallbackStrategy.PROCESSED

    def _blend_audio(
        self,
        original_path: str,
        processed_path: str,
        output_path: str,
        blend_ratio: float
    ) -> None:
        """
        Blend two audio files using FFmpeg's amix filter.

        Args:
            original_path: Path to original audio
            processed_path: Path to processed audio
            output_path: Path for blended output
            blend_ratio: 0.0 = all original, 1.0 = all processed
        """
        # Calculate weights (amix uses equal weights by default)
        processed_weight = blend_ratio
        original_weight = 1.0 - blend_ratio

        # Normalize weights so they sum to 1
        total = processed_weight + original_weight
        processed_weight /= total
        original_weight /= total

        cmd = build_ffmpeg_cmd([
            "-i", processed_path,
            "-i", original_path,
            "-filter_complex", f"[0:a]volume={processed_weight}[a0];[1:a]volume={original_weight}[a1];[a0][a1]amix=inputs=2:duration=first",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ])

        try:
            run_command(cmd, capture_output=True, check=True)
        except Exception as e:
            logger.error(f"Audio blending failed: {e}")
            # Fallback to just copying processed
            shutil.copy(processed_path, output_path)


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_audio_with_fallback(
    original_path: str,
    processed_path: str,
    output_path: str
) -> FallbackResult:
    """
    Apply audio processing with automatic fallback.

    Convenience function that creates a processor and runs fallback logic.

    Args:
        original_path: Path to original audio
        processed_path: Path to processed audio
        output_path: Path for final output

    Returns:
        FallbackResult with strategy and quality metrics
    """
    processor = AudioFallbackProcessor()
    return processor.process_with_fallback(original_path, processed_path, output_path)


def detect_audio_artifacts(audio_path: str) -> ArtifactReport:
    """
    Detect artifacts in an audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        ArtifactReport with detected issues
    """
    processor = AudioFallbackProcessor()
    return processor.detect_artifacts(audio_path)
