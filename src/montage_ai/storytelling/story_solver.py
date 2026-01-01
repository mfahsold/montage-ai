"""Greedy solver that maps clips to a story arc over beats."""

from dataclasses import dataclass
from typing import List, Dict, Any

from .story_arc import StoryArc
from .tension_provider import TensionProvider


@dataclass
class StorySolver:
    arc: StoryArc
    tension_provider: TensionProvider
    allow_reuse: bool = True
    reuse_penalty: float = 0.15

    def solve(self, clips: List[str], duration: float, beats: List[float]) -> List[Dict[str, Any]]:
        """Return a list of timeline events mapping beats to clips."""
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
