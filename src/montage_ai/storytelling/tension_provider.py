"""Lightweight tension lookup from precomputed metadata."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..utils import clamp


class MissingAnalysisError(RuntimeError):
    """Raised when tension metadata is missing for a clip."""


@dataclass
class TensionProvider:
    """
    Reads precomputed clip metadata and returns a tension score.

    Tension is a normalized value (0.0 to 1.0) representing the intensity of a clip.
    It is typically derived from:
    - Visual Motion (Optical Flow magnitude)
    - Audio Energy (RMS amplitude)
    - Semantic Content (e.g., "action", "calm" labels)

    This provider expects a directory of JSON files named `{clip_hash}_analysis.json`.

    Attributes:
        metadata_dir (Path): Directory containing the JSON analysis files.
        allow_dummy (bool): If True, generates a deterministic random tension for missing files.
                            Useful for testing or when full analysis is pending.
        _cache (Dict): Internal cache to avoid re-reading JSON files.
    """

    metadata_dir: Path
    allow_dummy: bool = False
    _cache: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.metadata_dir = Path(self.metadata_dir)
        if self._cache is None:
            self._cache = {}

    def get_tension(self, clip_path: str) -> float:
        """
        Retrieve tension score for a clip from metadata.

        Args:
            clip_path: Absolute path to the video clip.

        Returns:
            float: Tension score between 0.0 and 1.0.

        Raises:
            MissingAnalysisError: If metadata is missing and allow_dummy is False.
        """
        clip_id = self._get_clip_id(clip_path)
        if clip_id in self._cache:
            return self._cache[clip_id]

        meta_file = self.metadata_dir / f"{clip_id}_analysis.json"
        if not meta_file.exists():
            if self.allow_dummy:
                tension = self._dummy_tension(clip_id)
                self._cache[clip_id] = tension
                return tension
            raise MissingAnalysisError(f"Analysis missing for {clip_id}. Run cgpu job first.")

        with meta_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        tension = self._extract_tension(data)
        self._cache[clip_id] = tension
        return tension

    def _extract_tension(self, data: Dict[str, object]) -> float:
        """Compute tension from metadata with safe defaults."""
        if isinstance(data, dict) and "tension" in data:
            return clamp(float(data["tension"]))

        visual = data.get("visual", {}) if isinstance(data, dict) else {}
        motion = float(visual.get("motion_score", 0.0))
        edge = float(visual.get("edge_density", 0.0))
        tension = (motion * 0.6) + (edge * 0.4)
        return clamp(tension)

    def _dummy_tension(self, clip_id: str) -> float:
        """Deterministic fallback tension for dry-run testing."""
        digest = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()
        # Map first 8 hex chars to [0, 1]
        value = int(digest[:8], 16) / float(0xFFFFFFFF)
        return clamp(value)

    @staticmethod
    def _get_clip_id(clip_path: str) -> str:
        path = Path(clip_path)
        try:
            # Use filename + size for stable ID across mounts/environments
            stat = path.stat()
            identifier = f"{path.name}_{stat.st_size}"
        except OSError:
            # Fallback to just filename if stat fails
            identifier = path.name

        digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:8]
        return f"{path.stem}_{digest}"
