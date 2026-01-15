"""
Regisseur Memory: AI Experience-driven Video Editing

Stores and retrieves successful editing patterns to improve future jobs.
Part of the Agentic Video Editing system (SOTA 2026).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .logger import logger
from .config import get_settings

class RegisseurMemory:
    """
    Stores 'Directorial Experiences' to guide the Creative Director.
    """
    def __init__(self, memory_path: Optional[Path] = None):
        self.settings = get_settings()
        self.memory_path = memory_path or self.settings.paths.metadata_cache_dir / "regisseur_memory.json"
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load Regisseur Memory: {e}")
        return {"patterns": [], "style_feedback": {}}

    def save_experience(self, style: str, audio_tags: List[str], stats: Dict[str, Any], satisfaction: float):
        """Record a successful edit."""
        experience = {
            "timestamp": datetime.now().isoformat(),
            "style": style,
            "audio_tags": audio_tags,
            "stats": stats,
            "satisfaction": satisfaction
        }
        self._data["patterns"].append(experience)
        
        # Keep only last 100 experiences
        if len(self._data["patterns"]) > 100:
            self._data["patterns"] = self._data["patterns"][-100:]
            
        self._persist()

    def _persist(self):
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Regisseur Memory: {e}")

    def get_advice(self, current_style: str, audio_tags: List[str]) -> Optional[str]:
        """Return a string advising the AI Director based on past hits."""
        relevant = [p for p in self._data["patterns"] if p["style"] == current_style]
        if not relevant:
            return None
            
        # Simplistic recommendation: Average cut duration of successful edits
        avg_cut = sum(p["stats"].get("avg_cut_length", 3.0) for p in relevant) / len(relevant)
        return f"Pro Tip: In previous '{current_style}' edits, an average cut duration of {avg_cut:.1f}s achieved high satisfaction."

memory = None

def get_regisseur_memory() -> RegisseurMemory:
    global memory
    if memory is None:
        memory = RegisseurMemory()
    return memory
