"""Story arc definition and tension curve evaluation."""

from dataclasses import dataclass
from typing import List, Tuple


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


@dataclass
class StoryArc:
    """Defines a target tension curve over normalized time (0..1)."""

    curve_points: List[Tuple[float, float]]

    def __post_init__(self) -> None:
        if not self.curve_points:
            self.curve_points = [(0.0, 0.3), (0.5, 0.6), (1.0, 0.3)]
        # Sort by progress and clamp values into [0, 1]
        normalized = []
        for t, tension in self.curve_points:
            normalized.append((_clamp(float(t)), _clamp(float(tension))))
        normalized.sort(key=lambda pair: pair[0])
        self.curve_points = normalized

    def get_target_tension(self, progress: float) -> float:
        """Linearly interpolate tension for a given progress (0..1)."""
        if not self.curve_points:
            return 0.5

        p = _clamp(float(progress))
        first_t, first_val = self.curve_points[0]
        if p <= first_t:
            return first_val
        last_t, last_val = self.curve_points[-1]
        if p >= last_t:
            return last_val

        for (t0, v0), (t1, v1) in zip(self.curve_points, self.curve_points[1:]):
            if t0 <= p <= t1:
                if t1 == t0:
                    return v1
                ratio = (p - t0) / (t1 - t0)
                return v0 + ratio * (v1 - v0)

        return last_val

    @classmethod
    def from_preset(cls, name: str) -> "StoryArc":
        """Load a standard arc by name."""
        presets = {
            "hero_journey": [
                (0.0, 0.2),
                (0.35, 0.45),
                (0.7, 0.9),
                (0.9, 0.6),
                (1.0, 0.25),
            ],
            "mtv_energy": [
                (0.0, 0.85),
                (0.5, 0.95),
                (1.0, 0.85),
            ],
            "slow_burn": [
                (0.0, 0.15),
                (0.5, 0.25),
                (0.85, 0.95),
                (1.0, 0.4),
            ],
            "documentary": [
                (0.0, 0.25),
                (0.4, 0.4),
                (0.7, 0.6),
                (1.0, 0.35),
            ],
        }
        key = (name or "").lower().strip()
        if key in presets:
            return cls(curve_points=presets[key])
        return cls(curve_points=presets["hero_journey"])
