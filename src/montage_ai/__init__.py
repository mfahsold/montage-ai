"""
Montage AI - AI-Powered Video Montage Creation

Core:
    from montage_ai import create_montage
    create_montage(variant_id=1)

Styles:
    from montage_ai.style_templates import (
        get_style_template,
        list_available_styles,
        load_style_templates,
        reload_style_templates,
    )

Experimental:
    from montage_ai.timeline_exporter import TimelineExporter  # WIP
    from montage_ai.footage_analyzer import DeepFootageAnalyzer  # WIP
"""

__version__ = "0.3.0"

from .editor import create_montage
from .style_templates import (
    get_style_template,
    list_available_styles,
    load_style_templates,
    reload_style_templates,
)

__all__ = [
    "create_montage",
    "get_style_template",
    "list_available_styles",
    "load_style_templates",
    "reload_style_templates",
]
