"""
Engagement Score - Predict clip virality potential.

Multi-signal engagement prediction combining:
1. Audio Energy: High-energy moments engage viewers
2. Beat Alignment: Beat-synced cuts feel professional
3. Visual Variety: Scene changes maintain interest
4. Hook Quality: First 3 seconds determine retention
5. Pacing Score: Optimal cut frequency for platform

Usage:
    from montage_ai.engagement_score import calculate_engagement_score, EngagementReport

    report = calculate_engagement_score("/path/to/video.mp4")
    print(f"Overall: {report.overall_score}/100")
    print(f"Hook: {report.hook_score}/100")
    print(f"Recommendations: {report.recommendations}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import math

from .logger import logger
from .utils import clamp


@dataclass
class EngagementReport:
    """
    Comprehensive engagement score report.

    Scores are 0-100, higher is better.
    """
    # Overall score (weighted average)
    overall_score: float = 0.0

    # Component scores
    hook_score: float = 0.0        # First 3 seconds quality
    energy_score: float = 0.0     # Audio energy profile
    pacing_score: float = 0.0     # Cut frequency optimization
    variety_score: float = 0.0    # Visual variety
    audio_score: float = 0.0      # Music/speech quality

    # Metadata
    duration: float = 0.0
    scene_count: int = 0
    beat_count: int = 0
    avg_scene_duration: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Platform-specific scores
    platform_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 1),
            "hook_score": round(self.hook_score, 1),
            "energy_score": round(self.energy_score, 1),
            "pacing_score": round(self.pacing_score, 1),
            "variety_score": round(self.variety_score, 1),
            "audio_score": round(self.audio_score, 1),
            "duration": round(self.duration, 1),
            "scene_count": self.scene_count,
            "beat_count": self.beat_count,
            "avg_scene_duration": round(self.avg_scene_duration, 2),
            "recommendations": self.recommendations,
            "platform_scores": {k: round(v, 1) for k, v in self.platform_scores.items()},
        }

    @property
    def grade(self) -> str:
        """Letter grade for overall score."""
        if self.overall_score >= 90:
            return "A+"
        elif self.overall_score >= 80:
            return "A"
        elif self.overall_score >= 70:
            return "B"
        elif self.overall_score >= 60:
            return "C"
        elif self.overall_score >= 50:
            return "D"
        else:
            return "F"


# =============================================================================
# Platform-Specific Optimal Parameters
# =============================================================================

PLATFORM_PARAMS = {
    "tiktok": {
        "optimal_duration": (15, 60),      # Seconds
        "optimal_scene_duration": (0.5, 2.5),  # Seconds per scene
        "hook_window": 3.0,                 # First N seconds critical
        "energy_weight": 0.35,
        "pacing_weight": 0.3,
        "hook_weight": 0.35,
    },
    "youtube_shorts": {
        "optimal_duration": (30, 60),
        "optimal_scene_duration": (1.0, 3.0),
        "hook_window": 5.0,
        "energy_weight": 0.3,
        "pacing_weight": 0.3,
        "hook_weight": 0.4,
    },
    "instagram_reels": {
        "optimal_duration": (15, 90),
        "optimal_scene_duration": (0.8, 2.5),
        "hook_window": 3.0,
        "energy_weight": 0.3,
        "pacing_weight": 0.35,
        "hook_weight": 0.35,
    },
    "youtube_long": {
        "optimal_duration": (180, 600),
        "optimal_scene_duration": (3.0, 10.0),
        "hook_window": 10.0,
        "energy_weight": 0.25,
        "pacing_weight": 0.35,
        "hook_weight": 0.4,
    },
}


# =============================================================================
# Score Calculation Functions
# =============================================================================

def _calculate_hook_score(
    energy_curve: List[float],
    beats: List[float],
    duration: float,
    hook_window: float = 3.0,
) -> Tuple[float, List[str]]:
    """
    Calculate hook quality (first N seconds).

    High scores indicate:
    - Strong energy in opening
    - Beat alignment in hook
    - Quick engagement
    """
    recommendations = []

    if duration < hook_window:
        hook_window = duration

    if not energy_curve:
        return 50.0, ["Add audio for better hook analysis"]

    # Calculate hook energy (first hook_window seconds)
    import numpy as np
    energy = np.array(energy_curve)
    total_samples = len(energy)
    hook_samples = int(total_samples * (hook_window / duration))

    if hook_samples == 0:
        return 50.0, recommendations

    hook_energy = energy[:hook_samples]
    full_energy = energy

    # Hook should have above-average energy
    hook_avg = np.mean(hook_energy)
    full_avg = np.mean(full_energy)

    if full_avg > 0:
        hook_ratio = hook_avg / full_avg
    else:
        hook_ratio = 1.0

    # Score based on hook energy ratio
    # 1.2x = perfect (hook is 20% more energetic than average)
    # 1.0x = okay
    # 0.8x = weak hook
    if hook_ratio >= 1.2:
        base_score = 90
    elif hook_ratio >= 1.0:
        base_score = 70 + (hook_ratio - 1.0) * 100
    else:
        base_score = max(30, hook_ratio * 70)

    # Check for beat in first 2 seconds
    early_beats = [b for b in beats if b < 2.0]
    if early_beats:
        base_score += 10
    else:
        recommendations.append("Add a beat drop in first 2 seconds")

    # Check for energy peak in hook
    if hook_samples > 10:
        peak_idx = np.argmax(hook_energy)
        peak_position = peak_idx / hook_samples
        if peak_position < 0.5:  # Peak in first half of hook
            base_score += 5
        else:
            recommendations.append("Move energy peak earlier in the hook")

    if base_score < 70:
        recommendations.append("Strengthen your opening - hooks make or break engagement")

    return clamp(base_score, 0, 100), recommendations


def _calculate_energy_score(
    energy_curve: List[float],
    duration: float,
) -> Tuple[float, List[str]]:
    """
    Calculate energy profile score.

    High scores indicate:
    - Dynamic energy (not flat)
    - Peaks at appropriate times
    - Good overall energy level
    """
    recommendations = []

    if not energy_curve or len(energy_curve) < 10:
        return 50.0, ["Audio analysis incomplete"]

    import numpy as np
    energy = np.array(energy_curve)

    # 1. Dynamic range (variance)
    variance = np.var(energy)
    max_energy = np.max(energy)
    if max_energy > 0:
        normalized_variance = variance / (max_energy ** 2)
    else:
        normalized_variance = 0

    # Good variance is 0.05-0.2 (not too flat, not too chaotic)
    if 0.05 <= normalized_variance <= 0.2:
        variance_score = 100
    elif normalized_variance < 0.05:
        variance_score = max(40, normalized_variance * 800)
        recommendations.append("Energy is too flat - add dynamic moments")
    else:
        variance_score = max(50, 100 - (normalized_variance - 0.2) * 200)
        recommendations.append("Energy is too chaotic - smooth out transitions")

    # 2. Peak distribution (should have clear peaks, not constant)
    threshold = np.percentile(energy, 80)
    peaks = energy > threshold
    peak_count = np.sum(np.diff(peaks.astype(int)) == 1)

    # Optimal: 1 peak per 10-20 seconds
    optimal_peaks = duration / 15
    peak_ratio = peak_count / max(1, optimal_peaks)

    if 0.5 <= peak_ratio <= 2.0:
        peak_score = 100
    else:
        peak_score = max(40, 100 - abs(peak_ratio - 1.0) * 40)
        if peak_ratio < 0.5:
            recommendations.append("Add more energy peaks/highlights")
        else:
            recommendations.append("Too many peaks - let some moments breathe")

    # 3. Overall energy level
    avg_energy = np.mean(energy)
    if avg_energy > 0.6:
        level_score = 90
    elif avg_energy > 0.4:
        level_score = 70
    elif avg_energy > 0.2:
        level_score = 50
    else:
        level_score = 30
        recommendations.append("Overall energy is low - consider more energetic music")

    # Weighted average
    score = variance_score * 0.4 + peak_score * 0.3 + level_score * 0.3

    return clamp(score, 0, 100), recommendations


def _calculate_pacing_score(
    scene_count: int,
    duration: float,
    platform: str = "tiktok",
) -> Tuple[float, List[str]]:
    """
    Calculate pacing optimization score.

    High scores indicate:
    - Optimal cut frequency for platform
    - Good scene duration distribution
    """
    recommendations = []

    if duration <= 0:
        return 50.0, ["Duration unknown"]

    params = PLATFORM_PARAMS.get(platform, PLATFORM_PARAMS["tiktok"])
    optimal_scene_duration = params["optimal_scene_duration"]

    avg_scene_duration = duration / max(1, scene_count)

    # Score based on how close to optimal range
    min_optimal, max_optimal = optimal_scene_duration

    if min_optimal <= avg_scene_duration <= max_optimal:
        scene_score = 100
    elif avg_scene_duration < min_optimal:
        # Too fast
        ratio = avg_scene_duration / min_optimal
        scene_score = max(40, ratio * 100)
        recommendations.append(f"Cuts are too fast for {platform} - slow down pacing")
    else:
        # Too slow
        ratio = max_optimal / avg_scene_duration
        scene_score = max(40, ratio * 100)
        recommendations.append(f"Cuts are too slow for {platform} - increase pace")

    # Duration optimization
    min_dur, max_dur = params["optimal_duration"]
    if min_dur <= duration <= max_dur:
        duration_score = 100
    elif duration < min_dur:
        duration_score = max(50, (duration / min_dur) * 100)
        recommendations.append(f"Video is short for {platform} - aim for {min_dur}-{max_dur}s")
    else:
        duration_score = max(50, (max_dur / duration) * 100)
        recommendations.append(f"Video is long for {platform} - consider trimming")

    score = scene_score * 0.6 + duration_score * 0.4

    return clamp(score, 0, 100), recommendations


def _calculate_variety_score(
    scene_count: int,
    duration: float,
    visual_similarity: Optional[float] = None,
) -> Tuple[float, List[str]]:
    """
    Calculate visual variety score.

    High scores indicate:
    - Diverse shots
    - Sufficient scene changes
    - Not repetitive
    """
    recommendations = []

    if duration <= 0:
        return 50.0, ["Duration unknown"]

    # Scene frequency
    scenes_per_minute = (scene_count / duration) * 60

    # Optimal: 10-30 scenes per minute for short-form
    if 10 <= scenes_per_minute <= 30:
        freq_score = 100
    elif scenes_per_minute < 10:
        freq_score = max(40, scenes_per_minute * 10)
        recommendations.append("Add more visual variety - more scene changes")
    else:
        freq_score = max(50, 100 - (scenes_per_minute - 30) * 2)
        recommendations.append("Too many scene changes - may feel chaotic")

    # Visual similarity (if available)
    if visual_similarity is not None:
        # Lower similarity = more variety = better
        similarity_score = max(0, 100 - visual_similarity * 100)
        if similarity_score < 50:
            recommendations.append("Shots look too similar - add more diverse footage")
        score = freq_score * 0.6 + similarity_score * 0.4
    else:
        score = freq_score

    return clamp(score, 0, 100), recommendations


def _calculate_audio_score(
    has_music: bool,
    has_speech: bool,
    beat_count: int,
    duration: float,
) -> Tuple[float, List[str]]:
    """
    Calculate audio quality score.

    High scores indicate:
    - Music present
    - Good beat structure
    - Clear audio
    """
    recommendations = []
    score = 50.0  # Base score

    if has_music:
        score += 25
        # Check beat density
        if duration > 0:
            beats_per_minute = (beat_count / duration) * 60
            if 60 <= beats_per_minute <= 180:
                score += 15
            elif beats_per_minute < 60:
                score += 5
                recommendations.append("Music tempo is slow - consider upbeat track")
            else:
                score += 10
    else:
        recommendations.append("Add background music for higher engagement")

    if has_speech:
        score += 10
    else:
        recommendations.append("Consider adding voiceover for context")

    return clamp(score, 0, 100), recommendations


# =============================================================================
# Main API
# =============================================================================

def calculate_engagement_score(
    video_path: str,
    platform: str = "tiktok",
    detailed: bool = True,
) -> EngagementReport:
    """
    Calculate comprehensive engagement score for a video.

    Args:
        video_path: Path to video file
        platform: Target platform (tiktok, youtube_shorts, instagram_reels, youtube_long)
        detailed: Whether to include detailed recommendations

    Returns:
        EngagementReport with scores and recommendations
    """
    from .audio_analysis import AudioAnalyzer
    from .video_metadata import probe_metadata
    from .core.scene_provider import get_scene_provider

    path = Path(video_path)
    if not path.exists():
        logger.warning(f"Video not found: {video_path}")
        return EngagementReport(recommendations=["Video file not found"])

    # Get video metadata
    meta = probe_metadata(str(path))
    duration = meta.duration if meta else 0.0

    # Analyze audio
    try:
        analyzer = AudioAnalyzer(str(path))
        analyzer.analyze()
        energy_curve = list(analyzer.energy_curve) if hasattr(analyzer, 'energy_curve') else []
        beats = list(analyzer.beat_times) if hasattr(analyzer, 'beat_times') else []
        has_music = len(beats) > 5
        has_speech = False  # Would need transcription to detect
    except Exception as e:
        logger.warning(f"Audio analysis failed: {e}")
        energy_curve = []
        beats = []
        has_music = False
        has_speech = False

    # Detect scenes
    try:
        provider = get_scene_provider()
        scenes = provider.detect_scenes(str(path))
        scene_count = len(scenes)
    except Exception as e:
        logger.warning(f"Scene detection failed: {e}")
        scene_count = max(1, int(duration / 3))  # Estimate

    # Calculate component scores
    params = PLATFORM_PARAMS.get(platform, PLATFORM_PARAMS["tiktok"])
    all_recommendations = []

    hook_score, hook_recs = _calculate_hook_score(
        energy_curve, beats, duration, params["hook_window"]
    )
    all_recommendations.extend(hook_recs)

    energy_score, energy_recs = _calculate_energy_score(energy_curve, duration)
    all_recommendations.extend(energy_recs)

    pacing_score, pacing_recs = _calculate_pacing_score(scene_count, duration, platform)
    all_recommendations.extend(pacing_recs)

    variety_score, variety_recs = _calculate_variety_score(scene_count, duration)
    all_recommendations.extend(variety_recs)

    audio_score, audio_recs = _calculate_audio_score(has_music, has_speech, len(beats), duration)
    all_recommendations.extend(audio_recs)

    # Calculate overall score (weighted by platform)
    overall_score = (
        hook_score * params["hook_weight"] +
        energy_score * params["energy_weight"] +
        pacing_score * params["pacing_weight"] +
        variety_score * 0.1 +
        audio_score * 0.1
    ) / (params["hook_weight"] + params["energy_weight"] + params["pacing_weight"] + 0.2)

    # Calculate platform-specific scores
    platform_scores = {}
    for p_name in PLATFORM_PARAMS:
        p_params = PLATFORM_PARAMS[p_name]
        p_pacing, _ = _calculate_pacing_score(scene_count, duration, p_name)
        p_overall = (
            hook_score * p_params["hook_weight"] +
            energy_score * p_params["energy_weight"] +
            p_pacing * p_params["pacing_weight"]
        ) / (p_params["hook_weight"] + p_params["energy_weight"] + p_params["pacing_weight"])
        platform_scores[p_name] = p_overall

    # Deduplicate recommendations
    if detailed:
        seen = set()
        unique_recs = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)
        # Limit to top 5 most important
        recommendations = unique_recs[:5]
    else:
        recommendations = []

    return EngagementReport(
        overall_score=overall_score,
        hook_score=hook_score,
        energy_score=energy_score,
        pacing_score=pacing_score,
        variety_score=variety_score,
        audio_score=audio_score,
        duration=duration,
        scene_count=scene_count,
        beat_count=len(beats),
        avg_scene_duration=duration / max(1, scene_count),
        recommendations=recommendations,
        platform_scores=platform_scores,
    )


def get_engagement_summary(video_path: str, platform: str = "tiktok") -> Dict[str, Any]:
    """
    Get a simple engagement summary (for API responses).

    Args:
        video_path: Path to video file
        platform: Target platform

    Returns:
        Dict with score, grade, and top recommendations
    """
    report = calculate_engagement_score(video_path, platform, detailed=True)
    return {
        "score": round(report.overall_score),
        "grade": report.grade,
        "hook": round(report.hook_score),
        "energy": round(report.energy_score),
        "pacing": round(report.pacing_score),
        "recommendations": report.recommendations[:3],
        "best_platform": max(report.platform_scores.items(), key=lambda x: x[1])[0],
    }


__all__ = [
    'EngagementReport',
    'calculate_engagement_score',
    'get_engagement_summary',
    'PLATFORM_PARAMS',
]
