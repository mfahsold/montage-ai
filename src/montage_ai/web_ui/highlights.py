"""
Highlight Detection Module - Extract key moments from video.

Refactored from app.py to improve code quality and testability.
Multi-signal detection combining audio energy, beats, and speech.

Usage:
    from .highlights import detect_highlights

    highlights = detect_highlights(
        video_path="/path/to/video.mp4",
        max_clips=5,
        min_duration=5.0,
        max_duration=60.0
    )
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from ..logger import logger


@dataclass
class Highlight:
    """A detected highlight moment in video."""
    time: float
    start: float
    end: float
    duration: float
    score: float
    type: str
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "start": self.start,
            "end": self.end,
            "duration": round(self.duration, 2),
            "score": round(self.score, 3),
            "type": self.type,
            "label": self.label
        }


def detect_energy_highlights(
    energy: np.ndarray,
    hop_time: float,
    min_duration: float,
    max_duration: float
) -> List[Highlight]:
    """
    Detect high-energy moments in audio.

    Args:
        energy: Energy curve array from AudioAnalyzer
        hop_time: Time per sample in seconds
        min_duration: Minimum highlight duration
        max_duration: Maximum highlight duration

    Returns:
        List of energy-based highlights
    """
    highlights = []

    if len(energy) == 0:
        return highlights

    threshold_high = np.percentile(energy, 85)
    threshold_low = np.percentile(energy, 70)
    max_energy = np.max(energy) + 0.001

    in_highlight = False
    start_time = 0.0
    peak_energy = 0.0

    for i, e in enumerate(energy):
        time = i * hop_time

        if e > threshold_high and not in_highlight:
            in_highlight = True
            start_time = max(0, time - 0.5)
            peak_energy = e
        elif in_highlight:
            peak_energy = max(peak_energy, e)
            if e <= threshold_low:
                in_highlight = False
                duration = time - start_time
                if min_duration <= duration <= max_duration:
                    score = min(1.0, peak_energy / max_energy)
                    highlights.append(Highlight(
                        time=start_time,
                        start=start_time,
                        end=time,
                        duration=duration,
                        score=score,
                        type="Energy",
                        label=f"🔥 High Energy ({int(score*100)}%)"
                    ))

    return highlights


def detect_beat_drops(
    energy: np.ndarray,
    beats: List[float],
    sr: int,
    hop_length: int,
    min_duration: float,
    existing_highlights: List[Highlight]
) -> List[Highlight]:
    """
    Detect beat drop moments (sudden energy increases).

    Args:
        energy: Energy curve array
        beats: Beat times in seconds
        sr: Sample rate
        hop_length: Hop length for energy calculation
        min_duration: Minimum highlight duration
        existing_highlights: Already detected highlights (to avoid duplicates)

    Returns:
        List of beat drop highlights
    """
    highlights = []

    if len(energy) <= 20 or len(beats) == 0:
        return highlights

    energy_diff = np.diff(energy)
    beat_indices = [int(b * sr / hop_length) for b in beats]
    threshold = np.percentile(energy_diff, 90)

    for beat_idx, beat_time in zip(beat_indices[:20], beats[:20]):
        if 0 < beat_idx < len(energy_diff):
            if energy_diff[beat_idx] > threshold:
                # Check not too close to existing highlights
                if not any(abs(h.time - beat_time) < 3 for h in existing_highlights):
                    highlights.append(Highlight(
                        time=beat_time,
                        start=max(0, beat_time - 1),
                        end=beat_time + min_duration,
                        duration=min_duration + 1,
                        score=0.85,
                        type="Drop",
                        label="💥 Beat Drop"
                    ))

    return highlights


def detect_speech_hooks(
    video_path: str,
    existing_highlights: List[Highlight],
    max_clips: int
) -> List[Highlight]:
    """
    Detect speech hooks (punchy, high-density speech segments).

    Args:
        video_path: Path to video file
        existing_highlights: Already detected highlights
        max_clips: Maximum number of clips to detect

    Returns:
        List of speech hook highlights
    """
    highlights = []

    if len(existing_highlights) >= max_clips:
        return highlights

    try:
        from ..transcriber import transcribe_audio
        transcript = transcribe_audio(video_path, model='tiny', word_timestamps=True)

        segments = transcript.get('segments', [])
        for seg in segments[:5]:
            start = seg.get('start', 0)
            end = seg.get('end', start + 3)
            duration = end - start

            words = seg.get('words', [])
            word_density = len(words) / (duration + 0.001) if duration > 0 else 0

            # >2 words/sec = energetic delivery
            if word_density > 2.0 and duration >= 2:
                if not any(abs(h.time - start) < 3 for h in existing_highlights):
                    score = min(0.9, 0.5 + word_density * 0.15)
                    highlights.append(Highlight(
                        time=start,
                        start=start,
                        end=end,
                        duration=duration,
                        score=score,
                        type="Speech",
                        label="🎤 Hook"
                    ))
    except Exception as e:
        logger.debug(f"Speech hook detection skipped: {e}")

    return highlights


def detect_beat_aligned_fallback(
    beats: List[float],
    video_duration: float,
    existing_highlights: List[Highlight],
    max_clips: int,
    min_duration: float
) -> List[Highlight]:
    """
    Generate evenly distributed beat-aligned moments as fallback.

    Args:
        beats: Beat times in seconds
        video_duration: Total video duration
        existing_highlights: Already detected highlights
        max_clips: Maximum number of clips
        min_duration: Minimum highlight duration

    Returns:
        List of beat-aligned highlights
    """
    highlights = []

    if len(existing_highlights) >= max_clips or len(beats) <= 4:
        return highlights

    interval = video_duration / (max_clips + 1)
    needed = max_clips - len(existing_highlights)

    for i in range(1, needed + 1):
        target_time = i * interval
        nearest_beat = min(beats, key=lambda b: abs(b - target_time)) if beats else target_time

        if not any(abs(h.time - nearest_beat) < 5 for h in existing_highlights):
            highlights.append(Highlight(
                time=nearest_beat,
                start=nearest_beat,
                end=nearest_beat + min_duration,
                duration=min_duration,
                score=0.6,
                type="Beat",
                label="🎵 Beat"
            ))

    return highlights


def detect_highlights(
    video_path: str,
    max_clips: int = 5,
    min_duration: float = 5.0,
    max_duration: float = 60.0,
    include_speech: bool = True
) -> List[Dict[str, Any]]:
    """
    Detect highlight moments for clip extraction.

    Multi-signal highlight detection:
    1. Audio Energy: High-energy regions (music drops, loud moments)
    2. Beat Alignment: Key beats in music
    3. Speech Phrases: Important speech segments (hooks)
    4. Fallback: Evenly distributed beat-aligned moments

    Args:
        video_path: Path to video file
        max_clips: Maximum number of highlights to return
        min_duration: Minimum highlight duration in seconds
        max_duration: Maximum highlight duration in seconds
        include_speech: Whether to include speech hook detection

    Returns:
        List of highlight dictionaries sorted by time
    """
    from ..audio_analysis import AudioAnalyzer

    analyzer = AudioAnalyzer(video_path)
    analyzer.analyze()

    beats = getattr(analyzer, 'beat_times', []) or []
    energy = getattr(analyzer, 'energy_curve', []) or []
    sr = getattr(analyzer, 'sr', 22050)
    hop_length = getattr(analyzer, 'hop_length', 512)
    hop_time = hop_length / sr
    video_duration = getattr(analyzer, 'duration', 60)

    highlights: List[Highlight] = []

    # 1. Energy-based highlights
    if len(energy) > 0:
        energy_np = np.array(energy)
        highlights.extend(detect_energy_highlights(
            energy_np, hop_time, min_duration, max_duration
        ))

    # 2. Beat-drop detection
    if len(energy) > 20 and len(beats) > 0:
        highlights.extend(detect_beat_drops(
            np.array(energy), beats, sr, hop_length, min_duration, highlights
        ))

    # 3. Speech hook detection
    if include_speech:
        highlights.extend(detect_speech_hooks(video_path, highlights, max_clips))

    # 4. Fallback: beat-aligned moments
    highlights.extend(detect_beat_aligned_fallback(
        beats, video_duration, highlights, max_clips, min_duration
    ))

    # Sort by time
    highlights.sort(key=lambda x: x.time)

    # Take best clips by score if too many
    if len(highlights) > max_clips:
        highlights.sort(key=lambda x: x.score, reverse=True)
        highlights = highlights[:max_clips]
        highlights.sort(key=lambda x: x.time)

    return [h.to_dict() for h in highlights]


__all__ = [
    'Highlight',
    'detect_highlights',
    'detect_energy_highlights',
    'detect_beat_drops',
    'detect_speech_hooks',
    'detect_beat_aligned_fallback',
]
