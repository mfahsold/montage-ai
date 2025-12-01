"""
Style Templates - Predefined editing styles

Usage:
    from montage_ai.style_templates import get_style_template, list_available_styles
    
    styles = list_available_styles()  # ['hitchcock', 'mtv', 'action', ...]
    template = get_style_template('hitchcock')
"""

STYLE_TEMPLATES = {
    "hitchcock": {
        "name": "Hitchcock Suspense",
        "description": "Long takes building tension, rapid cuts at climax",
        "params": {
            "style": {"name": "hitchcock", "mood": "suspenseful"},
            "pacing": {"speed": "dynamic", "variation": "high", "intro_duration_beats": 16, "climax_intensity": 0.9},
            "cinematography": {"prefer_wide_shots": False, "prefer_high_action": True, "match_cuts_enabled": True, "invisible_cuts_enabled": True, "shot_variation_priority": "high"},
            "transitions": {"type": "hard_cuts", "crossfade_duration_sec": 0.3},
            "effects": {"color_grading": "high_contrast", "stabilization": False, "sharpness_boost": True}
        }
    },
    "mtv": {
        "name": "MTV Fast-Paced",
        "description": "Rapid 1-2 beat cuts, high energy music video style",
        "params": {
            "style": {"name": "mtv", "mood": "energetic"},
            "pacing": {"speed": "very_fast", "variation": "high", "intro_duration_beats": 2, "climax_intensity": 1.0},
            "cinematography": {"prefer_wide_shots": False, "prefer_high_action": True, "match_cuts_enabled": False, "invisible_cuts_enabled": True, "shot_variation_priority": "high"},
            "transitions": {"type": "hard_cuts", "crossfade_duration_sec": 0.0},
            "effects": {"color_grading": "vibrant", "stabilization": False, "sharpness_boost": True},
            "energy_mapping": {"sync_to_beats": True, "energy_amplification": 1.5}
        }
    },
    "action": {
        "name": "Action Blockbuster",
        "description": "Michael Bay style - explosive energy, rapid cuts",
        "params": {
            "style": {"name": "action", "mood": "dramatic"},
            "pacing": {"speed": "fast", "variation": "fibonacci", "intro_duration_beats": 4, "climax_intensity": 0.95},
            "cinematography": {"prefer_wide_shots": False, "prefer_high_action": True, "match_cuts_enabled": True, "invisible_cuts_enabled": True, "shot_variation_priority": "high"},
            "transitions": {"type": "hard_cuts", "crossfade_duration_sec": 0.0},
            "effects": {"color_grading": "vibrant", "stabilization": False, "sharpness_boost": True},
            "energy_mapping": {"sync_to_beats": True, "energy_amplification": 1.8}
        }
    },
    "documentary": {
        "name": "Documentary Realism",
        "description": "Natural pacing, observational, minimal effects",
        "params": {
            "style": {"name": "documentary", "mood": "calm"},
            "pacing": {"speed": "slow", "variation": "moderate", "intro_duration_beats": 16, "climax_intensity": 0.4},
            "cinematography": {"prefer_wide_shots": True, "prefer_high_action": False, "match_cuts_enabled": True, "invisible_cuts_enabled": True, "shot_variation_priority": "medium"},
            "transitions": {"type": "crossfade", "crossfade_duration_sec": 1.0},
            "effects": {"color_grading": "neutral", "stabilization": True, "sharpness_boost": False},
            "energy_mapping": {"sync_to_beats": False, "energy_amplification": 0.7}
        }
    },
    "minimalist": {
        "name": "Minimalist Art Film",
        "description": "Very long takes, contemplative, sparse cuts",
        "params": {
            "style": {"name": "minimalist", "mood": "calm"},
            "pacing": {"speed": "very_slow", "variation": "minimal", "intro_duration_beats": 32, "climax_intensity": 0.3},
            "cinematography": {"prefer_wide_shots": True, "prefer_high_action": False, "match_cuts_enabled": True, "invisible_cuts_enabled": True, "shot_variation_priority": "low"},
            "transitions": {"type": "crossfade", "crossfade_duration_sec": 2.0},
            "effects": {"color_grading": "desaturated", "stabilization": True, "sharpness_boost": False},
            "energy_mapping": {"sync_to_beats": False, "energy_amplification": 0.5},
            "constraints": {"min_clip_duration_sec": 4.0, "max_clip_duration_sec": 60.0}
        }
    },
    "wes_anderson": {
        "name": "Wes Anderson",
        "description": "Symmetrical, whimsical, measured pacing",
        "params": {
            "style": {"name": "wes_anderson", "mood": "playful"},
            "pacing": {"speed": "slow", "variation": "minimal", "intro_duration_beats": 8, "climax_intensity": 0.5},
            "cinematography": {"prefer_wide_shots": True, "prefer_high_action": False, "match_cuts_enabled": True, "invisible_cuts_enabled": False, "shot_variation_priority": "low"},
            "transitions": {"type": "crossfade", "crossfade_duration_sec": 0.8},
            "effects": {"color_grading": "warm", "stabilization": True, "sharpness_boost": False}
        }
    }
}


def get_style_template(style_name: str) -> dict:
    """Get a style template by name."""
    if style_name not in STYLE_TEMPLATES:
        available = ", ".join(STYLE_TEMPLATES.keys())
        raise KeyError(f"Unknown style '{style_name}'. Available: {available}")
    return STYLE_TEMPLATES[style_name]


def list_available_styles() -> list:
    """List all available style names."""
    return list(STYLE_TEMPLATES.keys())


def get_style_description(style_name: str) -> str:
    """Get the description of a style."""
    return STYLE_TEMPLATES[style_name]["description"]
