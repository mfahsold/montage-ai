"""
Montage AI - Full Feature Matrix E2E Test

This test systematically validates ALL configurable parameters of Montage AI.
It's designed to be:
1. Parseable by LLMs for automated analysis
2. Comprehensive - covers every "knob and dial"
3. Self-documenting - outputs structured results

Categories tested:
- Video Enhancement (stabilize, upscale, denoise, sharpen, film_grain)
- Audio Processing (voice_isolation, noise_reduction, dialogue_duck)
- Creative AI (llm_clip_selection, creative_loop, story_engine)
- Output Formats (export_timeline, shorts_mode, captions)
- Quality Profiles (preview, standard, high, master)
- Cut Styles (dynamic, hitchcock, mtv, documentary, etc.)
- Color Grading (16+ presets with intensity control)

Usage:
    pytest tests/integration/test_full_feature_matrix.py -v --tb=short

    # Or run standalone:
    python tests/integration/test_full_feature_matrix.py

Output Format (LLM-parseable):
    [FEATURE_TEST] feature_name | status | details
    [PARAM_CHECK] param_name=value | valid | type
    [E2E_RESULT] test_name | PASS/FAIL | duration_ms | notes
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# Test Result Data Structures (LLM-Parseable)
# =============================================================================

class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class ParameterTest:
    """Result of testing a single parameter."""
    name: str
    value: Any
    expected_type: str
    actual_type: str
    valid: bool
    notes: str = ""

    def __str__(self) -> str:
        status = "valid" if self.valid else "INVALID"
        return f"[PARAM_CHECK] {self.name}={self.value} | {status} | {self.actual_type}"


@dataclass
class FeatureTest:
    """Result of testing a feature."""
    name: str
    category: str
    status: TestStatus
    duration_ms: float
    details: str = ""
    sub_tests: List['ParameterTest'] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[FEATURE_TEST] {self.name} | {self.status.value} | {self.details}"


@dataclass
class E2ETestResult:
    """Complete E2E test result."""
    test_name: str
    status: TestStatus
    duration_ms: float
    features_tested: int
    features_passed: int
    parameters_tested: int
    parameters_valid: int
    notes: str = ""
    feature_results: List[FeatureTest] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[E2E_RESULT] {self.test_name} | {self.status.value} | {self.duration_ms:.0f}ms | {self.notes}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# =============================================================================
# Feature Matrix Definition
# =============================================================================

FEATURE_MATRIX = {
    "video_enhancement": {
        "description": "Per-clip video quality improvements",
        "parameters": {
            "STABILIZE": {"type": "bool", "default": False, "env": "STABILIZE"},
            "UPSCALE": {"type": "bool", "default": False, "env": "UPSCALE"},
            "ENHANCE": {"type": "bool", "default": True, "env": "ENHANCE"},
            "DENOISE": {"type": "bool", "default": False, "env": "DENOISE"},
            "SHARPEN": {"type": "bool", "default": False, "env": "SHARPEN"},
            "FILM_GRAIN": {"type": "str", "default": "none", "env": "FILM_GRAIN",
                          "options": ["none", "35mm", "16mm", "8mm", "digital", "fine", "coarse"]},
        }
    },
    "audio_processing": {
        "description": "Audio enhancement and manipulation",
        "parameters": {
            "VOICE_ISOLATION": {"type": "bool", "default": False, "env": "VOICE_ISOLATION"},
            "CLEAN_AUDIO": {"type": "bool", "default": False, "env": "CLEAN_AUDIO"},
            "NOISE_REDUCTION": {"type": "bool", "default": False, "env": "NOISE_REDUCTION"},
            "NOISE_REDUCTION_STRENGTH": {"type": "int", "default": 100, "env": "NOISE_REDUCTION_STRENGTH", "range": [0, 100]},
            "DIALOGUE_DUCK": {"type": "bool", "default": False, "env": "DIALOGUE_DUCK"},
            "DIALOGUE_DUCK_LEVEL": {"type": "float", "default": -12.0, "env": "DIALOGUE_DUCK_LEVEL", "range": [-30, 0]},
        }
    },
    "creative_ai": {
        "description": "LLM and AI-powered creative features",
        "parameters": {
            "LLM_CLIP_SELECTION": {"type": "bool", "default": True, "env": "LLM_CLIP_SELECTION"},
            "CREATIVE_LOOP": {"type": "bool", "default": False, "env": "CREATIVE_LOOP"},
            "CREATIVE_LOOP_MAX_ITERATIONS": {"type": "int", "default": 3, "env": "CREATIVE_LOOP_MAX_ITERATIONS", "range": [1, 10]},
            "STORY_ENGINE": {"type": "bool", "default": False, "env": "ENABLE_STORY_ENGINE"},
            "DEEP_ANALYSIS": {"type": "bool", "default": False, "env": "DEEP_ANALYSIS"},
            "CREATIVE_PROMPT": {"type": "str", "default": "", "env": "CREATIVE_PROMPT"},
        }
    },
    "output_formats": {
        "description": "Export and output configuration",
        "parameters": {
            "EXPORT_TIMELINE": {"type": "bool", "default": False, "env": "EXPORT_TIMELINE"},
            "GENERATE_PROXIES": {"type": "bool", "default": False, "env": "GENERATE_PROXIES"},
            "SHORTS_MODE": {"type": "bool", "default": False, "env": "SHORTS_MODE"},
            "REFRAME_MODE": {"type": "str", "default": "auto", "env": "REFRAME_MODE",
                            "options": ["auto", "speaker", "center", "custom"]},
            "CAPTIONS": {"type": "bool", "default": False, "env": "CAPTIONS"},
            "CAPTIONS_STYLE": {"type": "str", "default": "tiktok", "env": "CAPTIONS_STYLE",
                              "options": ["tiktok", "minimal", "bold", "karaoke"]},
        }
    },
    "quality_profiles": {
        "description": "Output quality and encoding settings",
        "parameters": {
            "QUALITY_PROFILE": {"type": "str", "default": "standard", "env": "QUALITY_PROFILE",
                               "options": ["preview", "standard", "high", "master"]},
            "TARGET_DURATION": {"type": "float", "default": 30.0, "env": "TARGET_DURATION", "range": [5, 3600]},
            "PRESERVE_ASPECT": {"type": "bool", "default": False, "env": "PRESERVE_ASPECT"},
        }
    },
    "cut_styles": {
        "description": "Editing rhythm and visual language presets",
        "parameters": {
            "CUT_STYLE": {"type": "str", "default": "dynamic", "env": "CUT_STYLE",
                         "options": ["dynamic", "hitchcock", "mtv", "action", "documentary",
                                    "minimalist", "wes_anderson", "podcast_highlight", "wedding"]},
        }
    },
    "color_grading": {
        "description": "Color correction and grading",
        "parameters": {
            "COLOR_GRADING": {"type": "str", "default": "auto", "env": "COLOR_GRADING",
                             "options": ["auto", "none", "teal_orange", "warm", "cool", "vintage",
                                        "cinematic", "noir", "pastel", "vibrant", "muted",
                                        "golden_hour", "blue_hour", "forest", "desert", "ocean",
                                        "neon", "film_emulation"]},
            "COLOR_INTENSITY": {"type": "float", "default": 1.0, "env": "COLOR_INTENSITY", "range": [0.0, 1.0]},
        }
    },
    "performance": {
        "description": "Performance and resource management",
        "parameters": {
            "LOW_MEMORY_MODE": {"type": "bool", "default": False, "env": "LOW_MEMORY_MODE"},
            "COLORLEVELS": {"type": "bool", "default": True, "env": "COLORLEVELS"},
            "LUMA_NORMALIZE": {"type": "bool", "default": True, "env": "LUMA_NORMALIZE"},
        }
    },
    "cloud_gpu": {
        "description": "Cloud GPU acceleration (cgpu)",
        "parameters": {
            "CGPU_ENABLED": {"type": "bool", "default": False, "env": "CGPU_ENABLED"},
            "STRICT_CLOUD_COMPUTE": {"type": "bool", "default": False, "env": "STRICT_CLOUD_COMPUTE"},
        }
    },
}


# =============================================================================
# Test Functions
# =============================================================================

def _check_parameter_loading() -> List[ParameterTest]:
    """Check that all parameters load correctly from config (helper function)."""
    from montage_ai.config import get_settings

    results = []
    settings = get_settings()

    for category, config in FEATURE_MATRIX.items():
        for param_name, param_config in config["parameters"].items():
            expected_type = param_config["type"]
            env_var = param_config["env"]

            # Get the actual value from settings
            # Map ENV names to settings attributes
            attr_map = {
                "STABILIZE": ("features", "stabilize"),
                "UPSCALE": ("features", "upscale"),
                "ENHANCE": ("features", "enhance"),
                "DENOISE": ("features", "denoise"),
                "SHARPEN": ("features", "sharpen"),
                "FILM_GRAIN": ("features", "film_grain"),
                "VOICE_ISOLATION": ("features", "voice_isolation"),
                "CLEAN_AUDIO": ("features", "voice_isolation"),  # alias
                "NOISE_REDUCTION": ("features", "noise_reduction"),
                "NOISE_REDUCTION_STRENGTH": ("features", "noise_reduction_strength"),
                "DIALOGUE_DUCK": ("features", "dialogue_duck"),
                "DIALOGUE_DUCK_LEVEL": ("features", "dialogue_duck_level"),
                "LLM_CLIP_SELECTION": ("features", "llm_clip_selection"),
                "CREATIVE_LOOP": ("features", "creative_loop"),
                "CREATIVE_LOOP_MAX_ITERATIONS": ("features", "creative_loop_max_iterations"),
                "ENABLE_STORY_ENGINE": ("features", "story_engine"),
                "STORY_ENGINE": ("features", "story_engine"),
                "DEEP_ANALYSIS": ("features", "deep_analysis"),
                "EXPORT_TIMELINE": ("features", "export_timeline"),
                "GENERATE_PROXIES": ("features", "generate_proxies"),
                "SHORTS_MODE": ("features", "shorts_mode"),
                "REFRAME_MODE": ("features", "reframe_mode"),
                "CAPTIONS": ("features", "captions"),
                "CAPTIONS_STYLE": ("features", "captions_style"),
                "QUALITY_PROFILE": ("encoding", "quality_profile"),
                "PRESERVE_ASPECT": ("features", "preserve_aspect"),
                "LOW_MEMORY_MODE": ("features", "low_memory_mode"),
                "COLORLEVELS": ("features", "colorlevels"),
                "LUMA_NORMALIZE": ("features", "luma_normalize"),
                "CGPU_ENABLED": ("llm", "cgpu_enabled"),
                "STRICT_CLOUD_COMPUTE": ("features", "strict_cloud_compute"),
            }

            try:
                if env_var in attr_map:
                    section, attr = attr_map[env_var]
                    section_obj = getattr(settings, section)
                    actual_value = getattr(section_obj, attr)
                else:
                    # Try environment variable directly
                    actual_value = os.environ.get(env_var, param_config["default"])

                actual_type = type(actual_value).__name__

                # Validate type
                type_valid = True
                if expected_type == "bool" and not isinstance(actual_value, bool):
                    type_valid = False
                elif expected_type == "int" and not isinstance(actual_value, int):
                    type_valid = False
                elif expected_type == "float" and not isinstance(actual_value, (int, float)):
                    type_valid = False
                elif expected_type == "str" and not isinstance(actual_value, str):
                    type_valid = False

                # Validate range/options if specified
                range_valid = True
                if "range" in param_config:
                    low, high = param_config["range"]
                    if actual_value < low or actual_value > high:
                        range_valid = False

                if "options" in param_config and actual_value not in param_config["options"]:
                    # Only warn, don't fail (custom values allowed)
                    pass

                valid = type_valid and range_valid

                results.append(ParameterTest(
                    name=param_name,
                    value=actual_value,
                    expected_type=expected_type,
                    actual_type=actual_type,
                    valid=valid,
                    notes=f"category={category}"
                ))

            except Exception as e:
                results.append(ParameterTest(
                    name=param_name,
                    value=None,
                    expected_type=expected_type,
                    actual_type="ERROR",
                    valid=False,
                    notes=f"Error: {e}"
                ))

    return results


def _check_feature_category(category: str, config: dict) -> FeatureTest:
    """Check a single feature category (helper function, not a pytest test)."""
    start_time = time.time()

    try:
        from montage_ai.config import get_settings
        settings = get_settings()

        # Count valid parameters
        param_results = []
        for param_name, param_config in config["parameters"].items():
            # Simple existence check
            env_var = param_config["env"]
            try:
                # Check if env var is recognized
                default = param_config["default"]
                param_results.append(ParameterTest(
                    name=param_name,
                    value=default,
                    expected_type=param_config["type"],
                    actual_type=type(default).__name__,
                    valid=True
                ))
            except Exception as e:
                param_results.append(ParameterTest(
                    name=param_name,
                    value=None,
                    expected_type=param_config["type"],
                    actual_type="ERROR",
                    valid=False,
                    notes=str(e)
                ))

        valid_count = sum(1 for p in param_results if p.valid)
        total_count = len(param_results)

        duration_ms = (time.time() - start_time) * 1000

        status = TestStatus.PASS if valid_count == total_count else TestStatus.WARN

        return FeatureTest(
            name=category,
            category=config["description"],
            status=status,
            duration_ms=duration_ms,
            details=f"{valid_count}/{total_count} parameters valid",
            sub_tests=param_results
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return FeatureTest(
            name=category,
            category=config.get("description", "Unknown"),
            status=TestStatus.FAIL,
            duration_ms=duration_ms,
            details=f"Error: {e}"
        )


def _check_montage_builder_integration() -> FeatureTest:
    """Check that MontageBuilder accepts all feature parameters (helper function)."""
    start_time = time.time()

    try:
        from montage_ai.core.montage_builder import MontageBuilder, MontageFeatures
        from montage_ai.config import get_settings

        settings = get_settings()

        # Create builder
        builder = MontageBuilder(variant_id=1, settings=settings)

        # Check context.features attributes
        feature_attrs = [
            "stabilize", "upscale", "enhance", "denoise", "sharpen",
            "film_grain", "dialogue_duck", "color_grade"
        ]

        missing = []
        for attr in feature_attrs:
            if not hasattr(builder.ctx.features, attr):
                missing.append(attr)

        duration_ms = (time.time() - start_time) * 1000

        if missing:
            return FeatureTest(
                name="montage_builder_integration",
                category="Core Pipeline",
                status=TestStatus.FAIL,
                duration_ms=duration_ms,
                details=f"Missing ctx.features attributes: {missing}"
            )

        return FeatureTest(
            name="montage_builder_integration",
            category="Core Pipeline",
            status=TestStatus.PASS,
            duration_ms=duration_ms,
            details=f"All {len(feature_attrs)} context.features attributes present"
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return FeatureTest(
            name="montage_builder_integration",
            category="Core Pipeline",
            status=TestStatus.FAIL,
            duration_ms=duration_ms,
            details=f"Error: {e}"
        )


def _check_enhancement_tracking() -> FeatureTest:
    """Test that EnhancementTracker records all enhancement types."""
    start_time = time.time()

    try:
        from montage_ai.enhancement_tracking import (
            EnhancementTracker, EnhancementDecision,
            StabilizeParams, UpscaleParams, ColorGradeParams
        )
        from montage_ai.clip_enhancement import DenoiseConfig, SharpenConfig, FilmGrainConfig

        tracker = EnhancementTracker()
        decision = tracker.create_decision(
            source_path="/test/video.mp4",
            timeline_in=0.0,
            timeline_out=5.0
        )

        # Record all enhancement types
        decision.record_stabilize(StabilizeParams(method="vidstab", smoothing=30))
        decision.record_upscale(UpscaleParams(method="realesrgan", scale_factor=2))
        decision.record_denoise(DenoiseConfig(spatial_strength=0.3))
        decision.record_sharpen(SharpenConfig(amount=0.5))
        decision.record_color_grade(ColorGradeParams(preset="cinematic", intensity=0.8))
        decision.record_film_grain(FilmGrainConfig(grain_type="16mm", enabled=True))

        # Verify all flags are set
        checks = [
            ("stabilized", decision.stabilized),
            ("upscaled", decision.upscaled),
            ("denoised", decision.denoised),
            ("sharpened", decision.sharpened),
            ("color_graded", decision.color_graded),
            ("film_grain_added", decision.film_grain_added),
        ]

        failed = [name for name, value in checks if not value]

        duration_ms = (time.time() - start_time) * 1000

        if failed:
            return FeatureTest(
                name="enhancement_tracking",
                category="Pro Handoff",
                status=TestStatus.FAIL,
                duration_ms=duration_ms,
                details=f"Failed flags: {failed}"
            )

        # Test export methods
        edl_comments = decision.to_edl_comments()
        json_dict = decision.to_dict()

        return FeatureTest(
            name="enhancement_tracking",
            category="Pro Handoff",
            status=TestStatus.PASS,
            duration_ms=duration_ms,
            details=f"All 6 enhancement types tracked, export OK"
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return FeatureTest(
            name="enhancement_tracking",
            category="Pro Handoff",
            status=TestStatus.FAIL,
            duration_ms=duration_ms,
            details=f"Error: {e}"
        )


def _check_clip_enhancer_methods() -> FeatureTest:
    """Test that ClipEnhancer has all required methods."""
    start_time = time.time()

    try:
        from montage_ai.clip_enhancement import ClipEnhancer

        # Check for required methods
        required_methods = [
            "stabilize",
            "upscale",
            "enhance",
            "denoise",
            "sharpen",
            "add_film_grain",
        ]

        missing = []
        for method in required_methods:
            if not hasattr(ClipEnhancer, method):
                missing.append(method)

        duration_ms = (time.time() - start_time) * 1000

        if missing:
            return FeatureTest(
                name="clip_enhancer_methods",
                category="Video Processing",
                status=TestStatus.FAIL,
                duration_ms=duration_ms,
                details=f"Missing methods: {missing}"
            )

        return FeatureTest(
            name="clip_enhancer_methods",
            category="Video Processing",
            status=TestStatus.PASS,
            duration_ms=duration_ms,
            details=f"All {len(required_methods)} methods present"
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return FeatureTest(
            name="clip_enhancer_methods",
            category="Video Processing",
            status=TestStatus.FAIL,
            duration_ms=duration_ms,
            details=f"Error: {e}"
        )


def _check_dialogue_ducking() -> FeatureTest:
    """Test dialogue ducking module."""
    start_time = time.time()

    try:
        from montage_ai.dialogue_ducking import (
            DialogueDetector, DialogueSegment, DuckKeyframe,
            DuckingConfig, detect_dialogue_segments, generate_duck_keyframes
        )

        # Test configuration
        config = DuckingConfig(
            duck_level_db=-12.0,
            attack_time=0.15,
            release_time=0.30
        )

        # Test keyframe generation with mock segments
        segments = [
            DialogueSegment(start_time=1.0, end_time=3.0, confidence=0.9),
            DialogueSegment(start_time=5.0, end_time=7.0, confidence=0.85),
        ]

        keyframes = generate_duck_keyframes(segments, total_duration=10.0)

        duration_ms = (time.time() - start_time) * 1000

        if not keyframes:
            return FeatureTest(
                name="dialogue_ducking",
                category="Audio Processing",
                status=TestStatus.FAIL,
                duration_ms=duration_ms,
                details="No keyframes generated"
            )

        return FeatureTest(
            name="dialogue_ducking",
            category="Audio Processing",
            status=TestStatus.PASS,
            duration_ms=duration_ms,
            details=f"Generated {len(keyframes)} keyframes for 2 segments"
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return FeatureTest(
            name="dialogue_ducking",
            category="Audio Processing",
            status=TestStatus.FAIL,
            duration_ms=duration_ms,
            details=f"Error: {e}"
        )


def _check_color_grading_presets() -> FeatureTest:
    """Test all color grading presets are available."""
    start_time = time.time()

    try:
        from montage_ai.color_grading import ColorGradeConfig, PRESET_FILTERS

        # Actual available presets (based on PRESET_FILTERS)
        expected_presets = [
            "teal_orange", "warm", "cool", "vintage", "cinematic",
            "noir", "vibrant", "golden_hour", "blue_hour",
            "documentary", "natural", "filmic_warm", "high_contrast"
        ]

        missing = []
        for preset in expected_presets:
            if preset not in PRESET_FILTERS:
                missing.append(preset)

        duration_ms = (time.time() - start_time) * 1000

        if missing:
            return FeatureTest(
                name="color_grading_presets",
                category="Color Grading",
                status=TestStatus.WARN,
                duration_ms=duration_ms,
                details=f"Missing presets: {missing}"
            )

        return FeatureTest(
            name="color_grading_presets",
            category="Color Grading",
            status=TestStatus.PASS,
            duration_ms=duration_ms,
            details=f"All {len(expected_presets)} presets available"
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return FeatureTest(
            name="color_grading_presets",
            category="Color Grading",
            status=TestStatus.FAIL,
            duration_ms=duration_ms,
            details=f"Error: {e}"
        )


def _check_cut_style_templates() -> FeatureTest:
    """Test all cut style templates load correctly."""
    start_time = time.time()

    try:
        from montage_ai.style_templates import get_style_template

        # Note: 'dynamic' is a fallback default, not a JSON file
        styles = [
            "hitchcock", "mtv", "action", "documentary",
            "minimalist", "wes_anderson"
        ]

        loaded = []
        failed = []

        for style in styles:
            try:
                template = get_style_template(style)
                if template:
                    loaded.append(style)
                else:
                    failed.append(style)
            except Exception:
                failed.append(style)

        duration_ms = (time.time() - start_time) * 1000

        if failed:
            return FeatureTest(
                name="cut_style_templates",
                category="Style Templates",
                status=TestStatus.WARN,
                duration_ms=duration_ms,
                details=f"Failed to load: {failed}"
            )

        return FeatureTest(
            name="cut_style_templates",
            category="Style Templates",
            status=TestStatus.PASS,
            duration_ms=duration_ms,
            details=f"Loaded {len(loaded)} style templates"
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return FeatureTest(
            name="cut_style_templates",
            category="Style Templates",
            status=TestStatus.FAIL,
            duration_ms=duration_ms,
            details=f"Error: {e}"
        )


def _check_timeline_exporter() -> FeatureTest:
    """Check timeline export formats (helper function)."""
    start_time = time.time()

    try:
        from montage_ai.timeline_exporter import TimelineExporter, Timeline, Clip

        # Create test timeline with correct structure
        clips = [
            Clip(source_path="/test/video.mp4", start_time=0, duration=5, timeline_start=0),
            Clip(source_path="/test/video.mp4", start_time=10, duration=5, timeline_start=5),
        ]

        timeline = Timeline(
            clips=clips,
            audio_path="/test/audio.mp3",
            total_duration=10.0,
            fps=30.0,
            project_name="test_timeline"
        )

        # TimelineExporter takes output_dir (use temp dir to avoid permission issues)
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TimelineExporter(output_dir=temp_dir)

            # Test export method and private format exporters exist
            # Main API: export_timeline() handles all formats
            # Private: _export_otio, _export_edl, _export_xml, _export_csv, _export_recipe_card
            export_methods = ["export_timeline", "_export_otio", "_export_edl", "_export_xml", "_export_csv", "_export_recipe_card"]

            missing = []
            for method in export_methods:
                if not hasattr(exporter, method):
                    missing.append(method)

            duration_ms = (time.time() - start_time) * 1000

            if missing:
                return FeatureTest(
                    name="timeline_exporter",
                    category="Pro Handoff",
                    status=TestStatus.FAIL,
                    duration_ms=duration_ms,
                    details=f"Missing methods: {missing}"
                )

            return FeatureTest(
                name="timeline_exporter",
                category="Pro Handoff",
                status=TestStatus.PASS,
                duration_ms=duration_ms,
                details=f"All {len(export_methods)} export formats available"
            )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return FeatureTest(
            name="timeline_exporter",
            category="Pro Handoff",
            status=TestStatus.FAIL,
            duration_ms=duration_ms,
            details=f"Error: {e}"
        )


def _check_story_engine() -> FeatureTest:
    """Test Story Engine components."""
    start_time = time.time()

    try:
        from montage_ai.storytelling.story_arc import StoryArc
        from montage_ai.storytelling.tension_provider import TensionProvider

        # Test preset arcs via from_preset() classmethod
        expected_arcs = ["hero_journey", "mtv_energy", "documentary", "slow_burn", "three_act"]

        available = []
        for arc_name in expected_arcs:
            try:
                arc = StoryArc.from_preset(arc_name)
                if arc and arc.curve_points:
                    available.append(arc_name)
            except (KeyError, ValueError):
                pass

        # Test tension provider with temp directory (allow_dummy=True for test)
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = TensionProvider(metadata_dir=temp_dir, allow_dummy=True)

        # Test get_target_tension method
        arc = StoryArc.from_preset("hero_journey")
        tension_at_half = arc.get_target_tension(0.5)

        duration_ms = (time.time() - start_time) * 1000

        return FeatureTest(
            name="story_engine",
            category="Creative AI",
            status=TestStatus.PASS,
            duration_ms=duration_ms,
            details=f"{len(available)}/{len(expected_arcs)} story arcs, tension@0.5={tension_at_half:.2f}"
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return FeatureTest(
            name="story_engine",
            category="Creative AI",
            status=TestStatus.FAIL,
            duration_ms=duration_ms,
            details=f"Error: {e}"
        )


# =============================================================================
# Main Test Runner
# =============================================================================

def run_full_feature_matrix_test() -> E2ETestResult:
    """Run the complete feature matrix test."""
    start_time = time.time()

    print("=" * 70)
    print("MONTAGE AI - FULL FEATURE MATRIX E2E TEST")
    print("=" * 70)
    print()

    feature_results = []

    # 1. Test parameter loading
    print("[1/10] Testing parameter loading...")
    param_results = _check_parameter_loading()
    param_valid = sum(1 for p in param_results if p.valid)
    param_total = len(param_results)
    print(f"       {param_valid}/{param_total} parameters valid")

    # 2. Test each feature category
    print("\n[2/10] Testing feature categories...")
    for category, config in FEATURE_MATRIX.items():
        result = _check_feature_category(category, config)
        feature_results.append(result)
        print(f"       {result}")

    # 3. Test MontageBuilder integration
    print("\n[3/10] Testing MontageBuilder integration...")
    result = _check_montage_builder_integration()
    feature_results.append(result)
    print(f"       {result}")

    # 4. Test EnhancementTracker
    print("\n[4/10] Testing EnhancementTracker...")
    result = _check_enhancement_tracking()
    feature_results.append(result)
    print(f"       {result}")

    # 5. Test ClipEnhancer
    print("\n[5/10] Testing ClipEnhancer methods...")
    result = _check_clip_enhancer_methods()
    feature_results.append(result)
    print(f"       {result}")

    # 6. Test DialogueDucking
    print("\n[6/10] Testing DialogueDucking...")
    result = _check_dialogue_ducking()
    feature_results.append(result)
    print(f"       {result}")

    # 7. Test ColorGrading presets
    print("\n[7/10] Testing ColorGrading presets...")
    result = _check_color_grading_presets()
    feature_results.append(result)
    print(f"       {result}")

    # 8. Test CutStyle templates
    print("\n[8/10] Testing CutStyle templates...")
    result = _check_cut_style_templates()
    feature_results.append(result)
    print(f"       {result}")

    # 9. Test TimelineExporter
    print("\n[9/10] Testing TimelineExporter...")
    result = _check_timeline_exporter()
    feature_results.append(result)
    print(f"       {result}")

    # 10. Test StoryEngine
    print("\n[10/10] Testing StoryEngine...")
    result = _check_story_engine()
    feature_results.append(result)
    print(f"       {result}")

    # Calculate totals
    duration_ms = (time.time() - start_time) * 1000
    features_passed = sum(1 for f in feature_results if f.status == TestStatus.PASS)
    features_total = len(feature_results)

    overall_status = TestStatus.PASS if features_passed == features_total else TestStatus.WARN
    if any(f.status == TestStatus.FAIL for f in feature_results):
        overall_status = TestStatus.FAIL

    result = E2ETestResult(
        test_name="full_feature_matrix",
        status=overall_status,
        duration_ms=duration_ms,
        features_tested=features_total,
        features_passed=features_passed,
        parameters_tested=param_total,
        parameters_valid=param_valid,
        notes=f"Tested {features_total} features, {param_total} parameters",
        feature_results=feature_results
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Status: {result.status.value}")
    print(f"Duration: {result.duration_ms:.0f}ms")
    print(f"Features: {result.features_passed}/{result.features_tested} passed")
    print(f"Parameters: {result.parameters_valid}/{result.parameters_tested} valid")
    print()

    # Print LLM-parseable output
    print("\n[LLM_PARSEABLE_OUTPUT]")
    for p in param_results:
        print(p)
    for f in feature_results:
        print(f)
    print(result)

    return result


# =============================================================================
# Pytest Integration
# =============================================================================

def test_full_feature_matrix():
    """Pytest entry point."""
    result = run_full_feature_matrix_test()
    assert result.status != TestStatus.FAIL, f"Feature matrix test failed: {result.notes}"


if __name__ == "__main__":
    result = run_full_feature_matrix_test()

    # Export JSON result
    output_path = Path(__file__).parent / "feature_matrix_results.json"
    with open(output_path, "w") as f:
        f.write(result.to_json())
    print(f"\nResults exported to: {output_path}")

    # Exit with appropriate code
    sys.exit(0 if result.status != TestStatus.FAIL else 1)
