"""
MoE Base Expert Interface

All experts inherit from BaseExpert and implement analyze() and propose() methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..contracts import EditingState, EditDelta, ParameterType


@dataclass
class ExpertConfig:
    """Configuration for an expert."""

    enabled: bool = True
    weight: float = 1.0  # Influence on final decision
    confidence_threshold: float = 0.5
    max_suggestions: int = 10


class BaseExpert(ABC):
    """
    Abstract base class for all MoE editing experts.

    Experts analyze the current editing state and propose deltas.
    The Control Plane validates, resolves conflicts, and composes final edits.
    """

    def __init__(self, expert_id: str, config: Optional[ExpertConfig] = None):
        self.expert_id = expert_id
        self.config = config if config is not None else ExpertConfig()
        self._analysis_cache: Dict[str, Any] = {}

    @abstractmethod
    def analyze(
        self, state: EditingState, media_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze current state and media to extract relevant features.

        Args:
            state: Current editing state
            media_context: Video/audio analysis results (beats, scenes, etc.)

        Returns:
            Analysis results (cached for propose step)
        """
        pass

    @abstractmethod
    def propose(self, state: EditingState, analysis: Dict[str, Any]) -> List[EditDelta]:
        """
        Propose editing deltas based on analysis.

        Args:
            state: Current editing state
            analysis: Results from analyze()

        Returns:
            List of proposed EditDeltas (may be empty)
        """
        pass

    def execute(
        self, state: EditingState, media_context: Dict[str, Any]
    ) -> List[EditDelta]:
        """
        Full execution: analyze + propose.

        This is the main entry point called by Control Plane.
        """
        if not self.config.enabled:
            return []

        analysis = self.analyze(state, media_context)
        self._analysis_cache = analysis

        deltas = self.propose(state, analysis)

        # Filter by confidence threshold
        filtered = [
            d for d in deltas if d.confidence >= self.config.confidence_threshold
        ]

        # Limit number of suggestions
        return filtered[: self.config.max_suggestions]

    def get_analysis(self) -> Dict[str, Any]:
        """Get cached analysis from last execution."""
        return self._analysis_cache.copy()
