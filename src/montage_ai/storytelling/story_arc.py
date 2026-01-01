"""Story arc definition and tension curve evaluation."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _coerce_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
            "three_act": [
                (0.0, 0.25),
                (0.35, 0.5),
                (0.7, 0.9),
                (1.0, 0.3),
            ],
            "fichtean_curve": [
                (0.0, 0.3),
                (0.2, 0.55),
                (0.4, 0.45),
                (0.6, 0.75),
                (0.8, 0.6),
                (1.0, 0.85),
            ],
            "linear_build": [
                (0.0, 0.2),
                (1.0, 0.85),
            ],
            "constant": [
                (0.0, 0.5),
                (1.0, 0.5),
            ],
        }
        key = (name or "").lower().strip().replace(" ", "_").replace("-", "_")
        aliases = {
            "mtv": "mtv_energy",
            "hero": "hero_journey",
            "journey": "hero_journey",
            "linear": "linear_build",
            "flat": "constant",
        }
        key = aliases.get(key, key)
        if key in presets:
            return cls(curve_points=presets[key])
        return cls(curve_points=presets["hero_journey"])

    @classmethod
    def from_spec(cls, spec: Optional[Dict[str, Any]]) -> "StoryArc":
        """Build a StoryArc from Creative Director spec."""
        if not spec:
            return cls.from_preset("hero_journey")

        arc_type = (spec.get("type") or "").lower().strip().replace(" ", "_").replace("-", "_")
        tension_target = _coerce_float(spec.get("tension_target"))
        if tension_target is not None:
            tension_target = _clamp(tension_target)

        climax_position = _coerce_float(spec.get("climax_position"))
        if climax_position is None:
            climax_position = 0.75
        climax_position = _clamp(climax_position, 0.6, 0.9)

        if arc_type == "constant":
            value = tension_target if tension_target is not None else 0.5
            return cls(curve_points=[(0.0, value), (1.0, value)])

        if arc_type == "linear_build":
            peak = tension_target if tension_target is not None else 0.85
            start = max(0.15, peak * 0.35)
            return cls(curve_points=[(0.0, start), (climax_position, peak), (1.0, peak)])

        if arc_type == "three_act":
            peak = tension_target if tension_target is not None else 0.9
            low = max(0.2, peak * 0.25)
            mid = max(low, peak * 0.6)
            pre = max(mid, peak * 0.75)
            end = max(low, peak * 0.35)
            return cls(
                curve_points=[
                    (0.0, low),
                    (climax_position * 0.4, mid),
                    (climax_position * 0.75, pre),
                    (climax_position, peak),
                    (1.0, end),
                ]
            )

        if arc_type == "fichtean_curve":
            peak = tension_target if tension_target is not None else 0.9
            base = max(0.3, peak * 0.4)
            mid1 = max(base, peak * 0.6)
            mid2 = max(base, peak * 0.7)
            mid3 = max(base, peak * 0.8)
            end = max(base, peak * 0.5)
            return cls(
                curve_points=[
                    (0.0, base),
                    (climax_position * 0.3, mid1),
                    (climax_position * 0.55, mid2),
                    (climax_position * 0.8, mid3),
                    (climax_position, peak),
                    (1.0, end),
                ]
            )

        if arc_type == "hero_journey":
            peak = tension_target if tension_target is not None else 0.9
            low = max(0.15, peak * 0.25)
            mid = max(low, peak * 0.5)
            fall = max(low, peak * 0.65)
            end = max(low, peak * 0.3)
            return cls(
                curve_points=[
                    (0.0, low),
                    (climax_position * 0.5, mid),
                    (climax_position, peak),
                    (min(1.0, climax_position + 0.15), fall),
                    (1.0, end),
                ]
            )

        return cls.from_preset(arc_type or "hero_journey")
