"""
Cinematic Style Templates for Natural Language Video Director

Based on 2024/2025 research on text-driven video editing and
film theory from legendary directors.

Each template defines editing parameters that capture the essence
of a particular directorial style or genre.
"""

STYLE_TEMPLATES = {
    "hitchcock": {
        "name": "Hitchcock Suspense",
        "description": "Master of suspense - long takes building tension, then rapid cuts at climax",
        "system_prompt": """You are Alfred Hitchcock's editor. Create suspenseful pacing:
- Long establishing shots (8-16 beats) to build unease
- Gradual acceleration as tension rises
- Rapid cuts (1-2 beats) at climax moments
- Prefer high-action scenes during peaks
- Use match cuts for psychological continuity""",
        "params": {
            "style": {
                "name": "hitchcock",
                "mood": "suspenseful"
            },
            "pacing": {
                "speed": "dynamic",
                "variation": "high",
                "intro_duration_beats": 16,
                "climax_intensity": 0.9
            },
            "cinematography": {
                "prefer_wide_shots": False,
                "prefer_high_action": True,
                "match_cuts_enabled": True,
                "invisible_cuts_enabled": True,
                "shot_variation_priority": "high"
            },
            "transitions": {
                "type": "hard_cuts",
                "crossfade_duration_sec": 0.3
            },
            "effects": {
                "color_grading": "high_contrast",
                "stabilization": False,
                "sharpness_boost": True
            }
        }
    },

    "wes_anderson": {
        "name": "Wes Anderson Symmetry",
        "description": "Whimsical, perfectly composed, pastel-colored precision",
        "system_prompt": """You are Wes Anderson's editor. Create meticulous symmetry:
- Steady, measured pacing (4-8 beats per cut)
- Minimal variation - mathematical precision
- Prefer wide/medium shots over close-ups
- Long takes to appreciate composition
- Gentle crossfades for dreamy transitions""",
        "params": {
            "style": {
                "name": "wes_anderson",
                "mood": "playful"
            },
            "pacing": {
                "speed": "slow",
                "variation": "minimal",
                "intro_duration_beats": 8,
                "climax_intensity": 0.5
            },
            "cinematography": {
                "prefer_wide_shots": True,
                "prefer_high_action": False,
                "match_cuts_enabled": True,
                "invisible_cuts_enabled": False,
                "shot_variation_priority": "low"
            },
            "transitions": {
                "type": "crossfade",
                "crossfade_duration_sec": 0.8
            },
            "effects": {
                "color_grading": "warm",
                "stabilization": True,
                "sharpness_boost": False
            }
        }
    },

    "mtv": {
        "name": "MTV Fast-Paced",
        "description": "2000s music video energy - rapid cuts, high saturation, constant motion",
        "system_prompt": """You are an MTV music video editor (2000s peak era). Create explosive energy:
- Very fast cuts (1-2 beats constant)
- Maximum variation and chaos
- All high-action, no static shots
- Hard cuts only - no fades
- Vibrant color grading
- Invisible cuts during motion blur for seamless flow""",
        "params": {
            "style": {
                "name": "mtv",
                "mood": "energetic"
            },
            "pacing": {
                "speed": "very_fast",
                "variation": "high",
                "intro_duration_beats": 2,
                "climax_intensity": 1.0
            },
            "cinematography": {
                "prefer_wide_shots": False,
                "prefer_high_action": True,
                "match_cuts_enabled": False,
                "invisible_cuts_enabled": True,
                "shot_variation_priority": "high"
            },
            "transitions": {
                "type": "hard_cuts",
                "crossfade_duration_sec": 0.0
            },
            "effects": {
                "color_grading": "vibrant",
                "stabilization": False,
                "sharpness_boost": True
            },
            "energy_mapping": {
                "sync_to_beats": True,
                "energy_amplification": 1.5
            }
        }
    },

    "documentary": {
        "name": "Documentary Realism",
        "description": "Observational, natural pacing, minimal effects",
        "system_prompt": """You are a documentary editor (vérité style). Create authentic flow:
- Long takes (8-16 beats) to let moments breathe
- Natural pacing that follows content, not music
- Prefer wide shots for context
- Gentle crossfades for time transitions
- Minimal color grading - realistic tones
- Stabilization to reduce handheld shake""",
        "params": {
            "style": {
                "name": "documentary",
                "mood": "calm"
            },
            "pacing": {
                "speed": "slow",
                "variation": "moderate",
                "intro_duration_beats": 16,
                "climax_intensity": 0.4
            },
            "cinematography": {
                "prefer_wide_shots": True,
                "prefer_high_action": False,
                "match_cuts_enabled": True,
                "invisible_cuts_enabled": True,
                "shot_variation_priority": "medium"
            },
            "transitions": {
                "type": "crossfade",
                "crossfade_duration_sec": 1.0
            },
            "effects": {
                "color_grading": "neutral",
                "stabilization": True,
                "sharpness_boost": False
            },
            "energy_mapping": {
                "sync_to_beats": False,
                "energy_amplification": 0.7
            }
        }
    },

    "minimalist": {
        "name": "Minimalist / Art Film",
        "description": "Sparse cuts, contemplative pacing, focus on composition",
        "system_prompt": """You are an art film editor (Tarkovsky, Malick style). Create meditative space:
- Very long takes (16-32 beats)
- Minimal cuts - let the image speak
- Prefer static wide shots
- Slow crossfades when transitioning
- Desaturated or natural color
- Maximum stabilization for stillness""",
        "params": {
            "style": {
                "name": "minimalist",
                "mood": "calm"
            },
            "pacing": {
                "speed": "very_slow",
                "variation": "minimal",
                "intro_duration_beats": 32,
                "climax_intensity": 0.3
            },
            "cinematography": {
                "prefer_wide_shots": True,
                "prefer_high_action": False,
                "match_cuts_enabled": True,
                "invisible_cuts_enabled": True,
                "shot_variation_priority": "low"
            },
            "transitions": {
                "type": "crossfade",
                "crossfade_duration_sec": 2.0
            },
            "effects": {
                "color_grading": "desaturated",
                "stabilization": True,
                "sharpness_boost": False
            },
            "energy_mapping": {
                "sync_to_beats": False,
                "energy_amplification": 0.5
            },
            "constraints": {
                "min_clip_duration_sec": 4.0,
                "max_clip_duration_sec": 60.0
            }
        }
    },

    "action": {
        "name": "Action / Blockbuster",
        "description": "Michael Bay style - explosive energy, rapid cuts, high saturation",
        "system_prompt": """You are a Hollywood action editor (Michael Bay, Russo Brothers). Create adrenaline:
- Fast cuts (1-3 beats) throughout
- Fibonacci variation during build-up
- Hyper-fast (1 beat) at climax
- All high-action scenes
- Maximum shot variation
- Hard cuts only - no softness
- High contrast, vibrant colors""",
        "params": {
            "style": {
                "name": "action",
                "mood": "dramatic"
            },
            "pacing": {
                "speed": "fast",
                "variation": "fibonacci",
                "intro_duration_beats": 4,
                "climax_intensity": 0.95
            },
            "cinematography": {
                "prefer_wide_shots": False,
                "prefer_high_action": True,
                "match_cuts_enabled": True,
                "invisible_cuts_enabled": True,
                "shot_variation_priority": "high"
            },
            "transitions": {
                "type": "hard_cuts",
                "crossfade_duration_sec": 0.0
            },
            "effects": {
                "color_grading": "vibrant",
                "stabilization": False,
                "sharpness_boost": True
            },
            "energy_mapping": {
                "sync_to_beats": True,
                "energy_amplification": 1.8
            }
        }
    }
}


def get_style_template(style_name: str) -> dict:
    """
    Get a predefined style template by name.

    Args:
        style_name: One of the template names (e.g., 'hitchcock', 'mtv')

    Returns:
        Dictionary with system_prompt and editing params

    Raises:
        KeyError if style_name not found
    """
    if style_name not in STYLE_TEMPLATES:
        available = ", ".join(STYLE_TEMPLATES.keys())
        raise KeyError(
            f"Unknown style '{style_name}'. Available styles: {available}"
        )

    return STYLE_TEMPLATES[style_name]


def list_available_styles() -> list:
    """Get list of all available style template names."""
    return list(STYLE_TEMPLATES.keys())


def get_style_description(style_name: str) -> str:
    """Get human-readable description of a style."""
    return STYLE_TEMPLATES[style_name]["description"]
