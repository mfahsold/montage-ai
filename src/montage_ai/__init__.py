"""
Montage AI - AI-Powered Automatic Video Montage Creation

A professional-grade video editing engine with:
- Natural language control via LLM
- Beat-synced editing with advanced pacing algorithms
- Professional footage management (clip consumption, story arc)
- Cinematic style templates (Hitchcock, MTV, Documentary, etc.)
- Timeline export to DaVinci Resolve, Premiere Pro, FCP

Usage:
    from montage_ai import MontageEditor
    
    editor = MontageEditor()
    editor.create_montage(
        video_dir="./videos",
        music_path="./music.mp3",
        creative_prompt="Edit like a Hitchcock thriller"
    )

Version: 0.3.0
License: MIT
"""

__version__ = "0.3.0"
__author__ = "Matthias Fahsold"

from .editor import MontageEditor
from .creative_director import CreativeDirector, interpret_natural_language
from .style_templates import get_style_template, list_available_styles
from .footage_manager import (
    FootagePoolManager,
    FootageClip,
    StoryArcController,
    integrate_footage_manager,
    select_next_clip
)
from .monitoring import Monitor, init_monitor, get_monitor

__all__ = [
    # Main editor
    "MontageEditor",
    
    # Creative direction
    "CreativeDirector",
    "interpret_natural_language",
    "get_style_template",
    "list_available_styles",
    
    # Footage management
    "FootagePoolManager",
    "FootageClip",
    "StoryArcController",
    "integrate_footage_manager",
    "select_next_clip",
    
    # Monitoring
    "Monitor",
    "init_monitor",
    "get_monitor",
]
