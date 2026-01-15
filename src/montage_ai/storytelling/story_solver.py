"""Greedy solver that maps clips to a story arc over beats."""

from dataclasses import dataclass
from typing import List, Dict, Any

from .story_arc import StoryArc
from .tension_provider import TensionProvider


@dataclass
class StorySolver:
    """
    Greedy solver that maps clips to a story arc over beats.
    ...
    """
    arc: StoryArc
    tension_provider: TensionProvider
    allow_reuse: bool = True
    reuse_penalty: float = 0.15
    fatigue_sensitivity: float = 0.4  # Default
    momentum_weight: float = 0.1      # Default

    def solve(self, clips: List[str], duration: float, beats: List[float]) -> List[Dict[str, Any]]:
        """
        Return a list of timeline events mapping beats to clips.
        Uses a Look-ahead Search to optimize global narrative flow.
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")

        # Fallback to greedy if few clips or beats
        if len(clips) < 5 or len(beats) < 3:
            return self._solve_greedy(clips, duration, beats)

        return self._solve_path_search(clips, duration, beats)

    def _solve_greedy(self, clips: List[str], duration: float, beats: List[float]) -> List[Dict[str, Any]]:
        """Original greedy implementation."""
        timeline: List[Dict[str, Any]] = []
        used = set()
        for beat_time in beats:
            if beat_time < 0 or beat_time > duration: continue
            progress = beat_time / duration
            target = self.arc.get_target_tension(progress)
            best_clip, best_score, best_tension = self._find_best_clip(clips, target, used)
            if best_clip:
                timeline.append({"time": beat_time, "clip": best_clip, "target_tension": target, "clip_tension": best_tension, "score": best_score})
                used.add(best_clip)
        return timeline

    def _solve_path_search(self, clips: List[str], duration: float, beats: List[float]) -> List[Dict[str, Any]]:
        """
        SOTA: Path Search (Simplified Beam Search) for Storytelling.
        Optimizes for global tension curve fit, variety, and momentum.
        """
        beam_size = 3
        # State: (score, used_clips_set, timeline_list, last_clip_tension)
        beams = [(0.0, set(), [], 0.5)]

        for i, beat_time in enumerate(beats):
            if beat_time < 0 or beat_time > duration: continue
            progress = beat_time / duration
            target = self.arc.get_target_tension(progress)
            
            # Look ahead for target tension to determine direction (momentum)
            next_progress = min(1.0, progress + 0.05)
            next_target = self.arc.get_target_tension(next_progress)
            target_direction = 1 if next_target > target else -1 if next_target < target else 0

            new_beams = []
            for total_score, used, current_timeline, last_tension in beams:
                # Find candidates
                candidates = []
                for clip in clips:
                    is_reuse = clip in used
                    if is_reuse and not self.allow_reuse: continue
                    
                    clip_tension = self.tension_provider.get_tension(clip)
                    error = abs(clip_tension - target)
                    match_score = 1.0 - error
                    
                    # SOTA: Momentum Bonus
                    # Reward clips that move tension in the desired direction of the arc
                    actual_direction = 1 if clip_tension > last_tension else -1 if clip_tension < last_tension else 0
                    if target_direction != 0 and actual_direction == target_direction:
                        match_score += self.momentum_weight
                    
                    # 2025: Contrast & Fatigue Heuristics
                    tension_diff = abs(clip_tension - last_tension)
                    if 0.1 < tension_diff < 0.4:
                        match_score += 0.05
                    
                    # High Tension Fatigue
                    high_tension_count = sum(1 for e in current_timeline[-3:] if e.get("clip_tension", 0) > 0.8)
                    if high_tension_count >= 2 and clip_tension > 0.8:
                        match_score -= (0.5 * self.fatigue_sensitivity)
                    
                    if is_reuse: match_score -= self.reuse_penalty
                    
                    candidates.append((match_score, clip, clip_tension))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                for score, clip, tension in candidates[:beam_size]:
                    new_used = used | {clip}
                    new_timeline = current_timeline + [{
                        "time": beat_time,
                        "clip": clip,
                        "target_tension": target,
                        "clip_tension": tension,
                        "score": score
                    }]
                    new_beams.append((total_score + score, new_used, new_timeline, tension))
            
            # Keep top beams
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

        return beams[0][2] if beams else []

    def _find_best_clip(self, clips, target, used):
        best_clip = None
        best_score = -1e9
        best_tension = None
        for clip in clips:
            reuse = clip in used
            if reuse and not self.allow_reuse: continue
            clip_tension = self.tension_provider.get_tension(clip)
            error = abs(clip_tension - target)
            score = 1.0 - error
            if reuse: score -= self.reuse_penalty
            if score > best_score:
                best_score = score
                best_clip = clip
                best_tension = clip_tension
        return best_clip, best_score, best_tension
