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
from dataclasses import dataclass
from typing import List, Tuple, Optional

import json
from pathlib import Path
import numpy as np
import librosa

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


# =============================================================================
# Core Functions
# =============================================================================

def _run_cloud_analysis(audio_path: str) -> Optional[dict]:
    """Run audio analysis on Cloud GPU if enabled."""
    if not (CGPU_AVAILABLE and _settings.llm.cgpu_enabled and is_cgpu_available()):
        return None

    analysis_path = Path(audio_path).with_suffix('.analysis.json')
    
    # Use cached result if available
    if analysis_path.exists():
        try:
            with open(analysis_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass  # Re-run if corrupt

    print(f"   ☁️ Offloading audio analysis to Cloud GPU...")
    try:
        job = BeatAnalysisJob(input_path=audio_path)
        result = job.execute()
        
        if result.success and result.output_path:
            with open(result.output_path, 'r') as f:
                data = json.load(f)
            print(f"   ✅ Cloud analysis complete.")
            return data
        else:
            print(f"   ⚠️ Cloud analysis failed: {result.error}. Falling back to local.")
    except Exception as e:
        print(f"   ⚠️ Cloud analysis error: {e}. Falling back to local.")
    
    return None


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
            hop_length=512 # Assumed default
        )
        
        if verbose:
            print(f"   📊 Energy Stats: avg={profile.avg_energy:.2f}, max={profile.max_energy:.2f}, min={profile.min_energy:.2f}")
            print(f"   📊 High Energy (>70%): {profile.high_energy_pct:.1f}% of track")
            
        return profile

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

    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Handle tempo being an array (newer librosa versions)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo.item())
    else:
        tempo = float(tempo)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

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


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "BeatInfo",
    "EnergyProfile",
    # Main functions
    "analyze_audio",
    "analyze_music_energy",
    "get_beat_times",
    "calculate_dynamic_cut_length",
]
