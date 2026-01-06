"""
Audio Analysis Module for Montage AI

Provides beat detection, energy analysis, and dynamic cut length calculation.
Uses librosa for audio processing.

Usage:
    from montage_ai.audio_analysis import analyze_audio, get_beat_times, calculate_dynamic_cut_length

    beat_info = get_beat_times("/path/to/music.mp3")
    energy = analyze_music_energy("/path/to/music.mp3")
"""

import os
import random
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import json
from pathlib import Path
import numpy as np

from .config import get_settings
from .logger import logger
from .core.cmd_runner import run_command, CommandError
from .ffmpeg_utils import build_ffmpeg_cmd
from .video_metadata import probe_duration

_settings = get_settings()

# Try to import librosa (may fail with numba/Python 3.12 compatibility issues)
LIBROSA_AVAILABLE = False
librosa = None
try:
    import librosa as _librosa
    # Test that librosa's numba-decorated functions actually work
    # This catches the 'get_call_template' error at import time
    _librosa.get_duration(y=np.zeros(22050, dtype=np.float32), sr=22050)
    librosa = _librosa
    LIBROSA_AVAILABLE = True
except Exception as e:
    # Known issue: librosa/numba incompatibility with Python 3.12
    # Error: 'function' object has no attribute 'get_call_template'
    logger.warning(f"librosa unavailable ({type(e).__name__}: {str(e)[:50]}), using FFmpeg fallback")
    LIBROSA_AVAILABLE = False

from .config import get_settings

# Import cgpu jobs for offloading
try:
    from .cgpu_jobs import BeatAnalysisJob
    from .cgpu_utils import is_cgpu_available
    CGPU_AVAILABLE = True
except ImportError:
    CGPU_AVAILABLE = False

_settings = get_settings()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BeatInfo:
    """Beat detection results."""
    tempo: float
    beat_times: np.ndarray
    duration: float
    sample_rate: int

    @property
    def beat_count(self) -> int:
        """Number of detected beats."""
        return len(self.beat_times)

    @property
    def avg_beat_interval(self) -> float:
        """Average time between beats in seconds."""
        return 60.0 / self.tempo if self.tempo > 0 else 0.0

    @property
    def tempo_category(self) -> str:
        """Categorize tempo as slow/medium/fast."""
        if self.tempo > 140:
            return "fast"
        elif self.tempo < 80:
            return "slow"
        return "medium"


@dataclass
class MusicSection:
    """A section of music with defined energy level."""
    start_time: float
    end_time: float
    energy_level: str  # "low", "medium", "high"
    avg_energy: float
    label: str = "undefined"  # "intro", "build", "drop", "outro", "verse", "chorus"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class EnergyProfile:
    """Energy envelope analysis results."""
    times: np.ndarray
    rms: np.ndarray  # Normalized 0-1
    sample_rate: int
    hop_length: int

    @property
    def avg_energy(self) -> float:
        """Average energy level."""
        return float(np.mean(self.rms))

    @property
    def max_energy(self) -> float:
        """Maximum energy level."""
        return float(np.max(self.rms))

    @property
    def min_energy(self) -> float:
        """Minimum energy level."""
        return float(np.min(self.rms))

    @property
    def high_energy_pct(self) -> float:
        """Percentage of track with energy > 70%."""
        return float(np.sum(self.rms > 0.7) / len(self.rms) * 100)

    def energy_at_time(self, time_sec: float) -> float:
        """Get energy level at a specific time."""
        if len(self.times) == 0:
            return 0.5
        idx = np.searchsorted(self.times, time_sec)
        idx = min(idx, len(self.rms) - 1)
        return float(self.rms[idx])


def fit_story_arc_to_sections(sections: List[MusicSection], total_duration: float) -> List[MusicSection]:
    """
    Label music sections with story arc phases (Intro, Build, Drop, Outro).
    
    Logic:
    - First low energy section -> "intro"
    - Section leading to high energy -> "build"
    - High energy section -> "drop"
    - Last low energy section -> "outro"
    """
    if not sections:
        return []

    # Copy list to avoid mutating original
    labeled_sections = []
    
    for i, section in enumerate(sections):
        # Default label based on energy
        label = "verse" if section.energy_level in ["low", "medium"] else "chorus"
        
        # Position-based overrides
        relative_start = section.start_time / total_duration
        relative_end = section.end_time / total_duration
        
        # Intro: First section(s), usually low energy, first 15%
        if i == 0 or (relative_end < 0.15 and section.energy_level == "low"):
            label = "intro"
            
        # Outro: Last section, usually low energy, last 10%
        elif i == len(sections) - 1 or (relative_start > 0.9 and section.energy_level == "low"):
            label = "outro"
            
        # Drop: High energy section
        elif section.energy_level == "high":
            # If preceded by a build-up, it fits the "Drop" definition perfectly
            label = "drop"
            
        # Build: Medium/High energy section BEFORE a Drop
        elif section.energy_level in ["medium", "high"]:
            # Look ahead for a drop
            is_build = False
            for forward_section in sections[i+1:]:
                if forward_section.energy_level == "high":
                    # Found a drop ahead, acts as build
                    is_build = True
                    break
                if forward_section.energy_level == "low":
                    # Energy drops back down, so not a build to a drop
                    break
            
            if is_build:
                label = "build"
        
        # Create new section with label
        labeled_sections.append(MusicSection(
            start_time=section.start_time,
            end_time=section.end_time,
            energy_level=section.energy_level,
            avg_energy=section.avg_energy,
            label=label
        ))
        
    return labeled_sections


def detect_music_sections(profile: EnergyProfile, min_section_duration: float = 5.0) -> List[MusicSection]:
    """
    Detect music sections (Intro, Verse, Chorus, etc.) based on energy levels.
    
    Args:
        profile: EnergyProfile object
        min_section_duration: Minimum duration for a section to be valid (seconds)
        
    Returns:
        List of MusicSection objects
    """
    if len(profile.rms) == 0:
        return []

    # 1. Smooth the energy profile to remove transient spikes
    window_size = int(2.0 * profile.sample_rate / profile.hop_length)  # 2 second window
    if window_size < 1:
        window_size = 1
    
    smoothed_energy = np.convolve(profile.rms, np.ones(window_size)/window_size, mode='same')
    
    # 2. Determine thresholds based on dynamic range
    # We use percentiles to be robust against overall loudness differences
    low_thresh = np.percentile(smoothed_energy, 33)
    high_thresh = np.percentile(smoothed_energy, 66)
    
    # 3. Classify each frame
    # 0=Low, 1=Medium, 2=High
    classes = np.zeros_like(smoothed_energy, dtype=int)
    classes[smoothed_energy >= low_thresh] = 1
    classes[smoothed_energy >= high_thresh] = 2
    
    # 4. Merge consecutive frames into sections
    sections = []
    current_class = classes[0]
    start_idx = 0
    
    for i in range(1, len(classes)):
        if classes[i] != current_class:
            # End of section
            end_idx = i
            start_time = profile.times[start_idx]
            end_time = profile.times[end_idx]
            
            # Map class to string
            level_map = {0: "low", 1: "medium", 2: "high"}
            
            sections.append(MusicSection(
                start_time=float(start_time),
                end_time=float(end_time),
                energy_level=level_map[current_class],
                avg_energy=float(np.mean(profile.rms[start_idx:end_idx]))
            ))
            
            current_class = classes[i]
            start_idx = i
            
    # Add final section
    end_idx = len(classes) - 1
    if start_idx < end_idx:
        level_map = {0: "low", 1: "medium", 2: "high"}
        sections.append(MusicSection(
            start_time=float(profile.times[start_idx]),
            end_time=float(profile.times[end_idx]),
            energy_level=level_map[current_class],
            avg_energy=float(np.mean(profile.rms[start_idx:end_idx]))
        ))
        
    # 5. Merge short sections into neighbors
    # This is a simple iterative merge
    merged = True
    while merged:
        merged = False
        if len(sections) <= 1:
            break
            
        new_sections = []
        i = 0
        while i < len(sections):
            current = sections[i]
            
            # If short section, try to merge with neighbor
            if current.duration < min_section_duration:
                merged = True
                # Merge with previous if possible (and similar energy preference?)
                # For now, just merge with the longer neighbor or previous
                if len(new_sections) > 0:
                    prev = new_sections[-1]
                    # Extend previous
                    prev.end_time = current.end_time
                    # Re-calculate avg energy (weighted)
                    total_dur = prev.duration + current.duration
                    prev.avg_energy = (prev.avg_energy * prev.duration + current.avg_energy * current.duration) / total_dur
                    # Keep previous energy level label
                elif i + 1 < len(sections):
                    # Merge with next
                    next_sec = sections[i+1]
                    next_sec.start_time = current.start_time
                    # Update next's stats
                    total_dur = next_sec.duration + current.duration
                    next_sec.avg_energy = (next_sec.avg_energy * next_sec.duration + current.avg_energy * current.duration) / total_dur
                    # Skip current, next iteration will handle the now-extended next_sec
                else:
                    # Orphaned short section at end, just keep it
                    new_sections.append(current)
            else:
                new_sections.append(current)
            i += 1
        sections = new_sections

    # 6. Apply Story Arc Labels
    duration = sections[-1].end_time if sections else 0.0
    labeled_sections = fit_story_arc_to_sections(sections, duration)
    return labeled_sections


# =============================================================================
# Core Functions
# =============================================================================

def _run_cloud_analysis(audio_path: str) -> Optional[dict]:
    """Run audio analysis on Cloud GPU if enabled."""
    if not (CGPU_AVAILABLE and _settings.llm.cgpu_enabled and is_cgpu_available()):
        if _settings.features.strict_cloud_compute:
            raise RuntimeError("Strict cloud compute enabled: cgpu audio analysis not available or disabled.")
        return None

    analysis_path = Path(audio_path).with_suffix('.analysis.json')

    # Use cached result if available
    if analysis_path.exists():
        try:
            with open(analysis_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass  # Re-run if corrupt

    logger.info("Offloading audio analysis to Cloud GPU...")
    try:
        job = BeatAnalysisJob(input_path=audio_path)
        result = job.execute()

        if result.success and result.output_path:
            with open(result.output_path, 'r') as f:
                data = json.load(f)
            logger.info("Cloud analysis complete.")
            return data
        else:
            if _settings.features.strict_cloud_compute:
                raise RuntimeError(f"Strict cloud compute enabled: Cloud audio analysis failed: {result.error}")
            logger.warning(f"Cloud analysis failed: {result.error}. Falling back to local.")
    except Exception as e:
        if _settings.features.strict_cloud_compute:
            raise RuntimeError(f"Strict cloud compute enabled: Cloud audio analysis error: {e}")
        logger.warning(f"Cloud analysis error: {e}. Falling back to local.")

    return None


# =============================================================================
# Audio Quality Validation (SNR Estimation)
# =============================================================================

@dataclass
class AudioQuality:
    """Audio quality assessment results."""
    snr_db: float
    mean_volume_db: float
    max_volume_db: float
    is_usable: bool
    quality_tier: str  # "excellent", "good", "acceptable", "poor", "unusable"

    @property
    def quality_level(self) -> str:
        """Alias for quality_tier (backward compatibility)."""
        return self.quality_tier

    @property
    def warning(self) -> Optional[str]:
        if self.quality_tier == "unusable":
            return f"Audio quality too low (SNR: {self.snr_db:.1f}dB). Consider re-recording."
        elif self.quality_tier == "poor":
            return f"Low audio quality (SNR: {self.snr_db:.1f}dB). Results may be unreliable."
        return None


def estimate_audio_snr(audio_path: str) -> AudioQuality:
    """
    Estimate Signal-to-Noise Ratio (SNR) of audio file.

    Uses FFmpeg to detect noise floor during silent sections
    and compare to peak signal level.

    Returns:
        AudioQuality with SNR estimate and usability assessment
    """
    # Step 1: Get volume statistics
    cmd_vol = build_ffmpeg_cmd(
        ["-i", audio_path, "-af", "volumedetect", "-f", "null", "-"],
        overwrite=False
    )

    mean_vol = -20.0
    max_vol = -10.0

    try:
        result = run_command(
            cmd_vol,
            capture_output=True,
            timeout=_settings.processing.analysis_timeout,
            check=False
        )
        for line in result.stderr.split('\n'):
            if 'mean_volume:' in line:
                mean_vol = float(line.split('mean_volume:')[1].split('dB')[0].strip())
            elif 'max_volume:' in line:
                max_vol = float(line.split('max_volume:')[1].split('dB')[0].strip())
    except Exception:
        pass

    # Step 2: Estimate noise floor from silence
    # Use silencedetect to find quiet sections, then measure their RMS
    cmd_silence = build_ffmpeg_cmd(
        [
            "-i", audio_path,
            "-af", "silencedetect=n=-50dB:d=0.1,astats=metadata=1",
            "-f", "null", "-"
        ],
        overwrite=False
    )

    noise_floor_db = -60.0  # Default assumption

    try:
        result = run_command(
            cmd_silence,
            capture_output=True,
            timeout=_settings.processing.analysis_timeout,
            check=False
        )
        # Look for RMS level during silent sections
        # If we find very low RMS values, that's our noise floor
        rms_values = []
        for line in result.stderr.split('\n'):
            if 'RMS level dB' in line:
                try:
                    rms = float(line.split('RMS level dB:')[1].strip().split()[0])
                    if rms < -30:  # Only consider quiet sections
                        rms_values.append(rms)
                except (ValueError, IndexError):
                    pass

        if rms_values:
            noise_floor_db = max(rms_values)  # Highest "quiet" level is noise floor
    except Exception:
        pass

    # Step 3: Calculate SNR
    # SNR = Peak Signal Level - Noise Floor
    snr_db = max_vol - noise_floor_db

    # Clamp to reasonable range
    snr_db = max(0, min(60, snr_db))

    # Step 4: Quality classification
    if snr_db >= 40:
        tier = "excellent"
        usable = True
    elif snr_db >= 25:
        tier = "good"
        usable = True
    elif snr_db >= 15:
        tier = "acceptable"
        usable = True
    elif snr_db >= 8:
        tier = "poor"
        usable = True
    else:
        tier = "unusable"
        usable = False

    return AudioQuality(
        snr_db=snr_db,
        mean_volume_db=mean_vol,
        max_volume_db=max_vol,
        is_usable=usable,
        quality_tier=tier
    )


def estimate_audio_snr_lufs(audio_path: str) -> AudioQuality:
    """
    Estimate SNR using LUFS (EBU R128) for more accurate measurement.

    LUFS provides perceptually accurate loudness measurements that are
    better correlated with human perception than simple dB measurements.

    Returns:
        AudioQuality with LUFS-based SNR estimate
    """
    # Get integrated loudness and loudness range
    cmd = build_ffmpeg_cmd(
        ["-i", audio_path, "-af", "ebur128=framelog=verbose:peak=true", "-f", "null", "-"],
        overwrite=False
    )

    try:
        result = run_command(cmd, capture_output=True, timeout=60, check=False)
        stderr = result.stderr

        # Parse summary stats
        integrated_lufs = -24.0  # Default
        loudness_range = 12.0  # Default
        true_peak = -1.0  # Default

        for line in stderr.split('\n'):
            if 'I:' in line and 'LUFS' in line:
                try:
                    i_part = line.split('I:')[1].split('LUFS')[0].strip()
                    integrated_lufs = float(i_part)
                except (ValueError, IndexError):
                    pass
            elif 'LRA:' in line and 'LU' in line:
                try:
                    lra_part = line.split('LRA:')[1].split('LU')[0].strip()
                    loudness_range = float(lra_part)
                except (ValueError, IndexError):
                    pass
            elif 'True peak:' in line or 'Peak:' in line:
                try:
                    peak_part = line.split(':')[-1].split('dB')[0].strip()
                    true_peak = float(peak_part)
                except (ValueError, IndexError):
                    pass

        # Estimate SNR from loudness range and integrated loudness
        # Higher LRA = more dynamic range = potentially cleaner audio
        # Lower integrated loudness with high LRA = good SNR
        snr_estimate = 30 + (loudness_range * 1.5) + (integrated_lufs + 24) * 0.5
        snr_estimate = max(0, min(60, snr_estimate))

        # Quality classification
        if snr_estimate >= 40:
            tier = "excellent"
        elif snr_estimate >= 25:
            tier = "good"
        elif snr_estimate >= 15:
            tier = "acceptable"
        elif snr_estimate >= 8:
            tier = "poor"
        else:
            tier = "unusable"

        return AudioQuality(
            snr_db=snr_estimate,
            mean_volume_db=integrated_lufs,
            max_volume_db=true_peak,
            is_usable=snr_estimate >= 8,
            quality_tier=tier
        )

    except Exception as e:
        logger.warning(f"LUFS-based SNR estimation failed: {e}, falling back to volume-based")
        return estimate_audio_snr(audio_path)


@dataclass
class AudioComparisonResult:
    """Result of comparing audio quality before and after processing."""
    before: AudioQuality
    after: AudioQuality
    snr_improvement_db: float
    quality_improved: bool
    summary: str


def compare_audio_quality(original_path: str, processed_path: str) -> AudioComparisonResult:
    """
    Compare audio quality before and after processing.

    Useful for validating Audio Polish (voice isolation, noise reduction).

    Returns:
        AudioComparisonResult with before/after metrics and improvement summary
    """
    before = estimate_audio_snr_lufs(original_path)
    after = estimate_audio_snr_lufs(processed_path)

    snr_improvement = after.snr_db - before.snr_db
    quality_improved = snr_improvement > 0

    # Generate summary
    if snr_improvement > 6:
        summary = f"Significant improvement: +{snr_improvement:.1f}dB SNR"
    elif snr_improvement > 3:
        summary = f"Moderate improvement: +{snr_improvement:.1f}dB SNR"
    elif snr_improvement > 0:
        summary = f"Slight improvement: +{snr_improvement:.1f}dB SNR"
    elif snr_improvement > -3:
        summary = f"No significant change: {snr_improvement:+.1f}dB SNR"
    else:
        summary = f"Quality degraded: {snr_improvement:.1f}dB SNR (consider rollback)"

    return AudioComparisonResult(
        before=before,
        after=after,
        snr_improvement_db=snr_improvement,
        quality_improved=quality_improved,
        summary=summary
    )


# =============================================================================
# FFmpeg Fallback (bare-metal, no Python dependencies)
# =============================================================================

def _ffmpeg_detect_onsets(audio_path: str, duration: float) -> List[float]:
    """
    Detect audio onsets (transients) using FFmpeg's silencedetect filter.

    This finds moments where audio jumps from silence/quiet to loud,
    which typically correspond to beat onsets.

    Returns:
        List of onset times in seconds
    """
    # Use silencedetect to find quiet gaps, then infer onsets
    # Silence threshold: -35dB, minimum duration: 0.05s (50ms)
    cmd = build_ffmpeg_cmd(
        [
            "-i", audio_path,
            "-af", f"silencedetect=n={_settings.audio.silence_threshold}:d={_settings.audio.min_silence_duration}",
            "-f", "null", "-"
        ],
        overwrite=False
    )

    try:
        result = run_command(
            cmd,
            capture_output=True,
            timeout=_settings.processing.analysis_timeout,
            check=False
        )
        stderr = result.stderr

        # Parse silence_end events (these are onset points)
        onsets = []
        for line in stderr.split('\n'):
            if 'silence_end:' in line:
                try:
                    # Format: [silencedetect @ 0x...] silence_end: 1.234 | silence_duration: 0.567
                    time_str = line.split('silence_end:')[1].split('|')[0].strip()
                    onset_time = float(time_str)
                    onsets.append(onset_time)
                except (ValueError, IndexError):
                    pass

        return onsets
    except Exception:
        return []


def _ffmpeg_analyze_loudness(audio_path: str, duration: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze audio loudness envelope using FFmpeg's ebur128 filter.

    Returns momentary loudness values at ~100ms intervals for peak detection.

    Returns:
        (times_array, loudness_array) - loudness in LUFS
    """
    # Use ebur128 with metadata output for momentary loudness
    cmd = build_ffmpeg_cmd(
        [
            "-i", audio_path,
            "-af", "ebur128=metadata=1:peak=true",
            "-f", "null", "-"
        ],
        overwrite=False
    )

    try:
        result = run_command(
            cmd,
            capture_output=True,
            timeout=_settings.processing.analysis_timeout,
            check=False
        )
        stderr = result.stderr

        # Parse momentary loudness (M:) values
        # Format: [Parsed_ebur128_0 @ ...] M: -23.4 S: -24.5 I: -25.6 LRA: 12.3
        loudness_values = []
        for line in stderr.split('\n'):
            if ' M:' in line and 'S:' in line:
                try:
                    m_part = line.split(' M:')[1].split(' S:')[0].strip()
                    loudness = float(m_part)
                    loudness_values.append(loudness)
                except (ValueError, IndexError):
                    pass

        if loudness_values:
            # ebur128 outputs at 10Hz (100ms intervals)
            times = np.linspace(0, duration, len(loudness_values))
            return times, np.array(loudness_values)

        return np.array([]), np.array([])

    except Exception:
        return np.array([]), np.array([])


def _detect_peaks_in_loudness(times: np.ndarray, loudness: np.ndarray,
                               threshold_percentile: float = 70) -> List[float]:
    """
    Detect beat-like peaks in loudness curve.

    Uses local maxima detection with dynamic thresholding.
    """
    if len(loudness) < 10:
        return []

    # Calculate threshold as percentile of loudness
    threshold = np.percentile(loudness, threshold_percentile)

    # Find local maxima above threshold
    peaks = []
    window = 3  # Look 3 samples on each side

    for i in range(window, len(loudness) - window):
        # Check if this is a local maximum
        if loudness[i] >= threshold:
            is_peak = True
            for j in range(-window, window + 1):
                if j != 0 and loudness[i + j] > loudness[i]:
                    is_peak = False
                    break
            if is_peak:
                # Avoid peaks too close together (minimum 0.15s apart)
                if not peaks or (times[i] - peaks[-1]) > 0.15:
                    peaks.append(times[i])

    return peaks


def _estimate_tempo_from_peaks(peak_times: List[float]) -> float:
    """
    Estimate tempo from inter-onset intervals.

    Uses histogram of intervals to find most common beat spacing.
    """
    if len(peak_times) < 4:
        return 120.0  # Default

    # Calculate intervals between peaks
    intervals = np.diff(peak_times)

    # Filter reasonable beat intervals (0.25s to 2s = 30-240 BPM)
    valid_intervals = intervals[(intervals >= 0.25) & (intervals <= 2.0)]

    if len(valid_intervals) < 3:
        return 120.0

    # Use median interval (robust to outliers)
    median_interval = np.median(valid_intervals)

    # Convert to BPM
    tempo = 60.0 / median_interval

    # Clamp to reasonable range
    tempo = max(60.0, min(200.0, tempo))

    return tempo


def _ffmpeg_estimate_tempo(audio_path: str, duration: float) -> Tuple[float, np.ndarray]:
    """
    Estimate tempo and beat times using FFmpeg's audio analysis.

    Uses multiple methods for robust beat detection:
    1. ebur128 loudness metering for energy peaks
    2. silencedetect for transient onsets
    3. Heuristic fallback based on average loudness

    Returns:
        (tempo_bpm, beat_times_array)
    """
    print(f"   🔍 Analyzing audio with FFmpeg onset detection...")

    # Method 1: Try onset detection via silencedetect
    onsets = _ffmpeg_detect_onsets(audio_path, duration)

    # Method 2: Try loudness analysis
    times, loudness = _ffmpeg_analyze_loudness(audio_path, duration)

    # Method 3: Combine results
    all_peaks = []

    if len(onsets) >= 5:
        all_peaks.extend(onsets)
        print(f"   📊 Detected {len(onsets)} transient onsets")

    if len(loudness) > 0:
        loudness_peaks = _detect_peaks_in_loudness(times, loudness)
        if len(loudness_peaks) >= 5:
            all_peaks.extend(loudness_peaks)
            print(f"   📊 Detected {len(loudness_peaks)} loudness peaks")

    # Estimate tempo from peaks
    if len(all_peaks) >= 8:
        # Sort and deduplicate (peaks within 0.1s are merged)
        all_peaks = sorted(set(all_peaks))
        merged_peaks = [all_peaks[0]]
        for p in all_peaks[1:]:
            if p - merged_peaks[-1] > 0.1:
                merged_peaks.append(p)

        tempo = _estimate_tempo_from_peaks(merged_peaks)

        # Generate regular beat grid aligned to detected peaks
        beat_interval = 60.0 / tempo

        # Find best phase alignment by testing different start offsets
        best_offset = 0.0
        best_score = 0
        for test_offset in np.linspace(0, beat_interval, 10):
            test_beats = np.arange(test_offset, duration, beat_interval)
            # Score: how many detected peaks are close to grid beats
            score = sum(1 for p in merged_peaks
                       for b in test_beats
                       if abs(p - b) < beat_interval * 0.25)
            if score > best_score:
                best_score = score
                best_offset = test_offset

        beat_times = np.arange(best_offset, duration, beat_interval)

        print(f"   📊 FFmpeg analysis: {tempo:.1f} BPM ({len(beat_times)} beats)")
        return tempo, beat_times

    # Fallback: Use volumedetect for basic heuristic
    print(f"   ⚠️ Insufficient peaks detected, using volumedetect fallback...")

    cmd = build_ffmpeg_cmd(
        [
            "-i", audio_path,
            "-af", "volumedetect",
            "-f", "null", "-"
        ],
        overwrite=False
    )

    try:
        result = run_command(
            cmd,
            capture_output=True,
            timeout=_settings.processing.analysis_timeout,
            check=False
        )
        stderr = result.stderr

        # Extract mean_volume from volumedetect
        mean_vol = -20.0
        for line in stderr.split('\n'):
            if 'mean_volume:' in line:
                try:
                    mean_vol = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                except (ValueError, IndexError):
                    pass

        # Heuristic: louder tracks tend to be faster
        if mean_vol > -10:
            tempo = 140.0
        elif mean_vol > -15:
            tempo = 125.0
        elif mean_vol > -20:
            tempo = 110.0
        else:
            tempo = 100.0

        beat_interval = 60.0 / tempo
        beat_times = np.arange(0, duration, beat_interval)

        print(f"   📊 FFmpeg heuristic: {tempo:.0f} BPM (mean vol: {mean_vol:.1f}dB)")
        return tempo, beat_times

    except Exception as e:
        print(f"   ⚠️ FFmpeg analysis failed: {e}")
        beat_times = np.arange(0, duration, 0.5)
        return 120.0, beat_times


def _ffmpeg_analyze_energy(audio_path: str, duration: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze audio energy using FFmpeg to extract raw samples + numpy RMS.

    This is a bare-metal approach that extracts audio samples and computes
    RMS energy in chunks, providing accurate energy profiles without librosa.

    Returns:
        (times_array, rms_normalized_array)
    """
    import tempfile
    import struct

    # Window size for RMS calculation (0.1s = 100ms)
    window_sec = 0.1
    sample_rate = 22050  # Downsample for efficiency

    # Extract raw audio samples as 16-bit PCM
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = build_ffmpeg_cmd([
            "-i", audio_path,
            "-ar", str(sample_rate),  # Resample
            "-ac", "1",               # Mono
            "-f", "s16le",            # 16-bit signed little-endian
            "-acodec", "pcm_s16le",
            tmp_path
        ])

        result = run_command(
            cmd,
            capture_output=True,
            timeout=_settings.processing.analysis_timeout,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg extraction failed: {result.stderr[:200]}")

        # Read raw samples
        with open(tmp_path, 'rb') as f:
            raw_data = f.read()

        # Convert to numpy array
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0  # Normalize to -1..1

        # Calculate RMS in windows
        window_samples = int(window_sec * sample_rate)
        num_windows = len(samples) // window_samples

        if num_windows < 2:
            # Very short audio, return uniform energy
            return np.array([0, duration]), np.array([0.5, 0.5])

        rms_values = []
        for i in range(num_windows):
            start = i * window_samples
            end = start + window_samples
            window = samples[start:end]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)

        rms_array = np.array(rms_values)

        # Normalize to 0-1 scale
        rms_min = np.min(rms_array)
        rms_max = np.max(rms_array)
        if rms_max > rms_min:
            rms_normalized = (rms_array - rms_min) / (rms_max - rms_min)
        else:
            rms_normalized = np.ones_like(rms_array) * 0.5

        times = np.linspace(0, duration, len(rms_normalized))

        print(f"   📊 Energy profile: {len(rms_normalized)} samples (RMS computed)")
        return times, rms_normalized

    except Exception as e:
        print(f"   ⚠️ FFmpeg energy extraction failed: {e}")
        # Fallback: synthetic curve based on volumedetect
        cmd_vol = build_ffmpeg_cmd(
            [
                "-i", audio_path,
                "-af", "volumedetect",
                "-f", "null", "-"
            ],
            overwrite=False
        )
        try:
            result_vol = run_command(
                cmd_vol,
                capture_output=True,
                timeout=_settings.processing.analysis_timeout,
                check=False
            )

            mean_vol = -20.0
            max_vol = -10.0
            for line in result_vol.stderr.split('\n'):
                if 'mean_volume:' in line:
                    mean_vol = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                elif 'max_volume:' in line:
                    max_vol = float(line.split('max_volume:')[1].split('dB')[0].strip())

            # Generate energy curve based on detected dynamic range
            num_samples = max(50, int(duration * 10))
            times = np.linspace(0, duration, num_samples)

            # Base energy from mean volume
            base = max(0.3, (mean_vol + 40) / 40)
            dynamic_range = min(0.3, (max_vol - mean_vol) / 40)

            # Create realistic energy curve with dynamics
            energy = base + dynamic_range * (
                0.3 * np.sin(2 * np.pi * times / duration) +
                0.2 * np.sin(4 * np.pi * times / duration) +
                0.1 * np.sin(8 * np.pi * times / duration)
            )
            rms = np.clip(energy, 0, 1)
            return times, rms

        except Exception:
            times = np.linspace(0, duration, 100)
            rms = np.ones(100) * 0.5
            return times, rms

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def analyze_music_energy(audio_path: str, verbose: Optional[bool] = None) -> EnergyProfile:
    """
    Analyze the energy envelope of an audio file.

    Args:
        audio_path: Path to audio file
        verbose: Override verbose setting (uses config if None)

    Returns:
        EnergyProfile with normalized RMS energy curve
    """
    if verbose is None:
        verbose = _settings.features.verbose

    print(f"🎵 Analyzing energy levels of {os.path.basename(audio_path)}...")

    # Try Cloud GPU first
    cloud_data = _run_cloud_analysis(audio_path)
    if cloud_data:
        energy_data = cloud_data['energy']
        rms = np.array(energy_data['rms'])
        times = np.array(energy_data['times'])

        # Normalize energy 0-1 (if not already)
        rms_min = np.min(rms)
        rms_max = np.max(rms)
        rms_normalized = (rms - rms_min) / (rms_max - rms_min + 1e-6)

        profile = EnergyProfile(
            times=times,
            rms=rms_normalized,
            sample_rate=cloud_data['sample_rate'],
            hop_length=512  # Assumed default
        )

        if verbose:
            print(f"   📊 Energy Stats: avg={profile.avg_energy:.2f}, max={profile.max_energy:.2f}, min={profile.min_energy:.2f}")
            print(f"   📊 High Energy (>70%): {profile.high_energy_pct:.1f}% of track")

        return profile

    # Use librosa if available, otherwise FFmpeg fallback
    if LIBROSA_AVAILABLE:
        y, sr = librosa.load(audio_path)
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        # Normalize energy 0-1
        rms_min = np.min(rms)
        rms_max = np.max(rms)
        rms_normalized = (rms - rms_min) / (rms_max - rms_min + 1e-6)

        profile = EnergyProfile(
            times=times,
            rms=rms_normalized,
            sample_rate=sr,
            hop_length=hop_length
        )
    else:
        # FFmpeg fallback (bare-metal, no Python audio deps)
        duration = probe_duration(audio_path)
        times, rms_normalized = _ffmpeg_analyze_energy(audio_path, duration)

        profile = EnergyProfile(
            times=times,
            rms=rms_normalized,
            sample_rate=44100,  # Assumed
            hop_length=512
        )

    if verbose:
        print(f"   📊 Energy Stats: avg={profile.avg_energy:.2f}, max={profile.max_energy:.2f}, min={profile.min_energy:.2f}")
        print(f"   📊 High Energy (>70%): {profile.high_energy_pct:.1f}% of track")

    return profile


def get_beat_times(audio_path: str, verbose: Optional[bool] = None) -> BeatInfo:
    """
    Detect beats and tempo in an audio file using librosa.

    Args:
        audio_path: Path to audio file
        verbose: Override verbose setting (uses config if None)

    Returns:
        BeatInfo with tempo, beat times, duration, and sample rate
    """
    if verbose is None:
        verbose = _settings.features.verbose

    print(f"🎵 Analyzing beat structure of {os.path.basename(audio_path)}...")

    # Try Cloud GPU first
    cloud_data = _run_cloud_analysis(audio_path)
    if cloud_data:
        tempo = cloud_data['tempo']
        beat_times = np.array(cloud_data['beat_times'])
        duration = cloud_data['duration']
        sr = cloud_data['sample_rate']
        
        print(f"   Tempo: {tempo:.1f} BPM, Detected {len(beat_times)} beats.")
        
        if verbose:
            print(f"   📊 Track Duration: {duration:.1f}s ({duration/60:.1f} min)")
            print(f"   📊 Sample Rate: {sr} Hz")
            print(f"   📊 Beat Interval: {60/tempo:.2f}s avg")

        return BeatInfo(
            tempo=tempo,
            beat_times=beat_times,
            duration=duration,
            sample_rate=sr
        )

    # Use librosa if available, otherwise FFmpeg fallback
    if LIBROSA_AVAILABLE:
        y, sr = librosa.load(audio_path)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Handle tempo being an array (newer librosa versions)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo.item())
        else:
            tempo = float(tempo)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        sr = int(sr)
    else:
        # FFmpeg fallback (bare-metal, no Python audio deps)
        duration = probe_duration(audio_path)
        tempo, beat_times = _ffmpeg_estimate_tempo(audio_path, duration)
        sr = 44100  # Assumed

    print(f"   Tempo: {tempo:.1f} BPM, Detected {len(beat_times)} beats.")

    if verbose:
        print(f"   📊 Track Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"   📊 Sample Rate: {sr} Hz")
        print(f"   📊 Beat Interval: {60/tempo:.2f}s avg")

        # Detect if tempo is fast/slow
        if tempo > 140:
            print(f"   🚀 Fast Tempo (>140 BPM) - will use longer beat groups to avoid seizure cuts")
        elif tempo < 80:
            print(f"   🐢 Slow Tempo (<80 BPM) - will use shorter beat groups for variety")
        else:
            print(f"   ⚖️ Medium Tempo - balanced pacing")

    return BeatInfo(
        tempo=tempo,
        beat_times=beat_times,
        duration=duration,
        sample_rate=sr
    )


def calculate_dynamic_cut_length(
    current_energy: float,
    tempo: float,
    current_time: float,
    total_duration: float,
    pattern_pool: List[List[float]],
    chaos_factor: float = 0.15
) -> List[float]:
    """
    Advanced pacing algorithm with position-aware intelligence.

    Based on research from:
    - Film Editing Pro 2024: Track-position pacing theory
    - Premiere Gal: Fibonacci rhythm patterns
    - Industry standard: Intro/Build/Climax/Outro structure

    Args:
        current_energy: Audio RMS energy (0-1)
        tempo: BPM of music
        current_time: Current position in track (seconds)
        total_duration: Total track duration (seconds)
        pattern_pool: List of available cut patterns
        chaos_factor: Probability of injecting random pattern (default 15%)

    Returns:
        List of beat counts for next cuts (Fibonacci or custom pattern)
    """
    # Calculate track position (0-1)
    progress = current_time / total_duration if total_duration > 0 else 0

    # PHASE 1: INTRO (0-20%) - Establish atmosphere, longer cuts
    if progress < 0.2:
        if current_energy < 0.3:
            # Calm intro: Very long takes to set mood
            base_pattern = [8, 8, 8, 4]
        else:
            # Energetic intro: Steady rhythm
            base_pattern = [4, 4, 4, 4]

    # PHASE 2: BUILD-UP (20-40%) - Increasing tension and variation
    elif progress < 0.4:
        if current_energy > 0.6:
            # Energy rising: Fibonacci acceleration
            base_pattern = [8, 5, 3, 2, 1, 1]  # Fibonacci descent
        else:
            # Gentle build: Classic pattern
            base_pattern = [4, 4, 2, 2]

    # PHASE 3: CLIMAX (40-75%) - Peak energy, maximum variation
    elif progress < 0.75:
        if current_energy > 0.8:
            # Hyper-energy peak: Rapid cuts (TikTok style)
            # "The Stutter": Rapid fire 1s followed by a breath
            base_pattern = [1, 1, 1, 0.5, 0.5, 2, 1, 1]
        elif current_energy > 0.6:
            # High energy: Fibonacci magic & Syncopation
            # "The Golden Spiral": 1, 1, 2, 3, 5
            base_pattern = [1, 1, 2, 3, 5]
        else:
            # Medium energy: Varied but controlled
            # "The Heartbeat": Short-Short-Long
            base_pattern = [1.5, 1.5, 5]  # Syncopated 3+5=8

    # PHASE 4: OUTRO/RESOLUTION (75-100%) - Wind down, return to calm
    else:
        if current_energy > 0.7:
            # High-energy ending: Sustain excitement then resolve
            base_pattern = [2, 2, 4, 8]
        else:
            # Calm ending: Long reflective cuts
            # "The Fade": Progressively longer
            base_pattern = [4, 8, 12, 16]

    # CHAOS FACTOR: Occasionally inject a random pattern from the pool
    # This simulates "creative intuition" breaking the rules
    if pattern_pool and random.random() < chaos_factor:
        base_pattern = random.choice(pattern_pool)

    # TEMPO MODULATION: Adjust for BPM
    # Fast tempos need longer beat counts to avoid seizure-inducing cuts
    if tempo > 140:
        base_pattern = [max(2, b) for b in base_pattern]  # Minimum 2 beats
    elif tempo < 80:
        base_pattern = [max(1, b / 2) for b in base_pattern]  # Halve for slow songs

    return base_pattern


def analyze_audio(audio_path: str, verbose: Optional[bool] = None) -> Tuple[BeatInfo, EnergyProfile]:
    """
    Convenience function to analyze both beats and energy in one call.

    Args:
        audio_path: Path to audio file
        verbose: Override verbose setting

    Returns:
        Tuple of (BeatInfo, EnergyProfile)
    """
    beat_info = get_beat_times(audio_path, verbose=verbose)
    energy_profile = analyze_music_energy(audio_path, verbose=verbose)
    return beat_info, energy_profile


def remove_filler_words(transcript: dict, fillers: Optional[List[str]] = None) -> List[int]:
    """
    Identify indices of filler words in a transcript.
    
    Args:
        transcript: The transcript data structure (from Whisper JSON).
        fillers: List of words to remove (defaults to common English fillers).
        
    Returns:
        List of indices to mark as deleted.
    """
    if fillers is None:
        fillers = ["um", "uh", "like", "you know", "sort of", "kind of", "hmm", "er"]
    
    indices_to_remove = []
    
    # Handle different transcript formats (segments vs words)
    words = []
    if 'segments' in transcript:
        for segment in transcript['segments']:
            if 'words' in segment:
                words.extend(segment['words'])
    elif 'words' in transcript:
        words = transcript['words']
        
    for i, word_obj in enumerate(words):
        # Clean word: remove punctuation, lowercase
        raw_word = word_obj.get('word', '')
        text = ''.join(c for c in raw_word if c.isalnum()).lower()
        
        if text in fillers:
            indices_to_remove.append(i)
            
    return indices_to_remove


# =============================================================================
# Dialogue Detection Integration
# =============================================================================

# Re-export dialogue detection for convenience
try:
    from .dialogue_ducking import (
        DialogueSegment,
        DialogueDetector,
        DuckingConfig,
        detect_dialogue_segments,
        generate_duck_keyframes,
        apply_ducking_to_audio,
    )
    DIALOGUE_DETECTION_AVAILABLE = True
except ImportError:
    DIALOGUE_DETECTION_AVAILABLE = False


def analyze_dialogue_for_ducking(
    voice_path: str,
    music_duration: float,
    duck_level_db: float = -12.0,
    method: str = "auto"
) -> Optional[dict]:
    """
    Analyze dialogue/speech for auto-ducking.

    This convenience function wraps DialogueDetector for simple usage.

    Args:
        voice_path: Path to voice/dialogue audio
        music_duration: Duration of music track (for keyframe generation)
        duck_level_db: Volume reduction in dB during speech
        method: Detection method (auto, silero, webrtc, ffmpeg)

    Returns:
        Dict with 'segments', 'keyframes', and 'ffmpeg_filter' keys,
        or None if dialogue detection is not available.
    """
    if not DIALOGUE_DETECTION_AVAILABLE:
        logger.warning("Dialogue detection not available")
        return None

    config = DuckingConfig(duck_level_db=duck_level_db)
    detector = DialogueDetector(config)

    result = detector.analyze(voice_path, music_duration, method)

    return {
        "segments": [s.to_dict() for s in result.segments],
        "keyframes": [k.to_dict() for k in result.keyframes],
        "ffmpeg_filter": detector.generate_ffmpeg_filter(result),
        "total_speech_duration": result.total_speech_duration,
        "speech_percentage": result.speech_percentage,
        "method": result.method,
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "BeatInfo",
    "EnergyProfile",
    "MusicSection",
    "AudioQuality",
    # Main functions
    "analyze_audio",
    "analyze_music_energy",
    "get_beat_times",
    "calculate_dynamic_cut_length",
    "detect_music_sections",
    "fit_story_arc_to_sections",
    "remove_filler_words",
    # Audio quality
    "estimate_audio_snr",
    "estimate_audio_snr_lufs",
    "compare_audio_quality",
    # Dialogue detection (if available)
    "analyze_dialogue_for_ducking",
    "DIALOGUE_DETECTION_AVAILABLE",
]
