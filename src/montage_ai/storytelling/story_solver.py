"""Greedy solver that maps clips to a story arc over beats."""

from dataclasses import dataclass
from typing import List, Dict, Any

from .story_arc import StoryArc
from .tension_provider import TensionProvider


@dataclass
class StorySolver:
    """
    Greedy solver that maps clips to a story arc over beats.

    This solver iterates through the provided beats and selects the best available clip
    that matches the target tension for that point in the story arc.

    Algorithm:
    1. Iterate through each beat time.
    2. Calculate the 'progress' (0.0 to 1.0) of the story.
    3. Query the `StoryArc` for the target tension at that progress.
    4. Score all available clips based on how close their tension is to the target.
    5. Apply penalties for reusing clips (if allowed).
    6. Select the clip with the highest score.

    Attributes:
        arc (StoryArc): The narrative arc defining tension over time.
        tension_provider (TensionProvider): Source of tension data for clips.
        allow_reuse (bool): Whether clips can be used multiple times.
        reuse_penalty (float): Score penalty for reusing a clip (default: 0.15).
    """
    arc: StoryArc
    tension_provider: TensionProvider
    allow_reuse: bool = True
    reuse_penalty: float = 0.15

    def solve(self, clips: List[str], duration: float, beats: List[float]) -> List[Dict[str, Any]]:
        """
        Return a list of timeline events mapping beats to clips.

        Args:
            clips: List of absolute file paths to candidate video clips.
            duration: Total duration of the montage in seconds.
            beats: List of timestamps (seconds) where cuts should occur.

        Returns:
            List of dicts, each containing:
                - 'time': The timestamp of the cut.
                - 'clip': The selected clip path.
                - 'target_tension': The ideal tension at this time.
                - 'clip_tension': The actual tension of the selected clip.
                - 'score': The matching score (higher is better).

        Raises:
            ValueError: If duration is not positive.
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")

        timeline: List[Dict[str, Any]] = []
        used = set()

        for beat_time in beats:
            if beat_time < 0 or beat_time > duration:
                continue
            progress = beat_time / duration
            target = self.arc.get_target_tension(progress)

            best_clip = None
            best_score = -1e9
            best_tension = None

            for clip in clips:
                reuse = clip in used
                if reuse and not self.allow_reuse:
                    continue

                clip_tension = self.tension_provider.get_tension(clip)
                error = abs(clip_tension - target)
                score = 1.0 - error
                if reuse:
                    score -= self.reuse_penalty

                if score > best_score:
                    best_score = score
                    best_clip = clip
                    best_tension = clip_tension

            if best_clip:
                timeline.append(
                    {
                        "time": beat_time,
                        "clip": best_clip,
                        "target_tension": target,
                        "clip_tension": best_tension,
                        "score": best_score,
                    }
                )
                used.add(best_clip)

        return timeline
