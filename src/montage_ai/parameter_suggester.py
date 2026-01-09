"""
Central LLM-based parameter suggestion system for AI director/editor control.

This module provides a unified interface for AI agents to intelligently tune
editing parameters based on footage analysis and creative intent. All parameter
suggesters use CreativeDirector's LLM backend (Ollama/cgpu/Gemini) for consistency.

Architecture:
  Scene Analysis → LLM Reasoning → Parameter Adjustments → Validation → Application

Design Principles:
  1. Central mechanism: All LLM calls go through CreativeDirector
  2. cgpu-robust: Handles cgpu backend with fallback to Ollama
  3. Typed parameters: Uses EditingParameters schema for validation
  4. Explainable: Returns reasoning for parameter choices
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from .creative_director import CreativeDirector
from .editing_parameters import (
    EditingParameters,
    ColorGradingParameters,
    StabilizationParameters,
    COLOR_GRADING_PRESETS,
)
from .logger import logger


@dataclass
class ParameterSuggestion:
    """
    Result of LLM-based parameter suggestion.
    
    Attributes:
        parameters: Suggested parameter values
        reasoning: LLM's explanation for the suggestions
        confidence: Confidence score (0.0-1.0)
        alternatives: Alternative parameter sets (if requested)
    """
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float = 0.8
    alternatives: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class ParameterSuggester(ABC):
    """
    Base class for LLM-powered parameter suggesters.
    
    All suggesters use CreativeDirector's LLM backend for consistency
    and cgpu robustness. Subclasses implement domain-specific prompts
    and validation logic.
    """
    
    def __init__(self, creative_director: Optional[CreativeDirector] = None):
        """
        Initialize parameter suggester.
        
        Args:
            creative_director: Optional CreativeDirector instance. If None,
                             creates a new one (will auto-detect cgpu/Ollama).
        """
        self.director = creative_director or CreativeDirector(
            persona="an expert post-production colorist and video editor"
        )
    
    @abstractmethod
    def _build_suggestion_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build LLM prompt for parameter suggestion.
        
        Args:
            context: Analysis data (scene info, quality metrics, etc.)
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def _parse_llm_response(self, response: str) -> ParameterSuggestion:
        """
        Parse LLM response into structured parameter suggestion.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Validated ParameterSuggestion
        """
        pass
    
    def suggest(self, context: Dict[str, Any]) -> ParameterSuggestion:
        """
        Generate parameter suggestions using LLM.
        
        Args:
            context: Analysis context (footage analysis, user intent, etc.)
            
        Returns:
            ParameterSuggestion with reasoning
            
        Raises:
            RuntimeError: If LLM query fails
        """
        try:
            # Build prompt with context
            prompt = self._build_suggestion_prompt(context)
            
            # Query LLM (uses CreativeDirector's backend - cgpu/Ollama/Gemini)
            logger.info(f"Querying LLM for parameter suggestions (backend: {self._get_backend_name()})")
            response = self._query_llm(prompt)
            
            # Parse and validate response
            suggestion = self._parse_llm_response(response)
            
            logger.info(f"LLM suggestion confidence: {suggestion.confidence:.2f}")
            logger.debug(f"LLM reasoning: {suggestion.reasoning}")
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Parameter suggestion failed: {e}")
            raise RuntimeError(f"Failed to generate parameter suggestions: {e}") from e
    
    def _query_llm(self, prompt: str) -> str:
        """
        Query LLM via CreativeDirector (cgpu-robust).
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            LLM response text
        """
        # CreativeDirector handles cgpu/Ollama/Gemini backend selection
        # and fallback logic automatically
        try:
            # Use director's query method directly
            # Note: We may need to add a generic query method to CreativeDirector
            # For now, use suggest_edits with custom prompt
            response = self.director._query_backend(prompt, temperature=0.7, max_tokens=1000)
            return response
        except AttributeError:
            # Fallback: DirectorOutput.parse expects specific schema
            # We'll implement _query_backend in CreativeDirector separately
            raise NotImplementedError(
                "CreativeDirector needs _query_backend method for generic queries"
            )
    
    def _get_backend_name(self) -> str:
        """Get current LLM backend name for logging."""
        if self.director.use_openai_api:
            return "OpenAI-compatible"
        elif self.director.use_cgpu:
            return "cgpu/Gemini"
        elif self.director.use_google_ai:
            return "Google AI"
        else:
            return f"Ollama ({self.director.ollama_model})"


class ColorGradingSuggester(ParameterSuggester):
    """
    LLM-powered color grading parameter suggester.
    
    Analyzes scene content, lighting conditions, and creative intent
    to suggest optimal color grading presets and parameters.
    
    Usage:
        suggester = ColorGradingSuggester()
        context = {
            "scene_description": "sunset beach scene with warm tones",
            "histogram": {"shadows": 0.2, "midtones": 0.5, "highlights": 0.3},
            "dominant_colors": ["orange", "blue"],
            "user_intent": "cinematic blockbuster feel"
        }
        suggestion = suggester.suggest(context)
        print(f"Suggested preset: {suggestion.parameters['preset']}")
        print(f"Reasoning: {suggestion.reasoning}")
    """
    
    def _build_suggestion_prompt(self, context: Dict[str, Any]) -> str:
        """Build color grading suggestion prompt."""
        scene_desc = context.get("scene_description", "unknown scene")
        user_intent = context.get("user_intent", "professional cinematic look")
        histogram = context.get("histogram", {})
        dominant_colors = context.get("dominant_colors", [])
        
        # Build histogram description
        hist_desc = ""
        if histogram:
            hist_desc = f"""
Histogram Analysis:
- Shadows: {histogram.get('shadows', 0):.1%}
- Midtones: {histogram.get('midtones', 0):.1%}
- Highlights: {histogram.get('highlights', 0):.1%}
"""
        
        # Build color description
        color_desc = ""
        if dominant_colors:
            color_desc = f"Dominant Colors: {', '.join(dominant_colors)}"
        
        prompt = f"""You are an expert post-production colorist. Analyze the following scene and suggest optimal color grading parameters.

Scene Description: {scene_desc}
User Intent: {user_intent}
{hist_desc}{color_desc}

Available Color Grading Presets:
{self._format_preset_list()}

Task: Suggest the best color grading parameters for this scene.

Respond in JSON format:
{{
    "preset": "<preset_name>",
    "intensity": <0.0-1.0>,
    "temperature": <-1.0 to 1.0>,
    "tint": <-1.0 to 1.0>,
    "saturation": <0.0-2.0>,
    "contrast": <0.0-2.0>,
    "brightness": <-1.0 to 1.0>,
    "reasoning": "<explanation of choices>",
    "confidence": <0.0-1.0>
}}

Guidelines:
- Choose presets that enhance the scene's mood
- Consider histogram balance when adjusting brightness/contrast
- Maintain natural skin tones when possible
- Intensity 0.8-1.0 for strong looks, 0.5-0.7 for subtle grades
- Temperature: negative = cooler, positive = warmer
- Confidence: higher for clear-cut choices, lower for ambiguous scenes
"""
        return prompt
    
    def _format_preset_list(self) -> str:
        """Format available presets for prompt."""
        # Group presets by category for better LLM understanding
        preset_descriptions = {
            "teal_orange": "Modern blockbuster look with teal shadows, orange highlights",
            "cinematic": "Classic cinema grade with rich colors and contrast",
            "blockbuster": "High-contrast Hollywood action film look",
            "vintage": "Retro film look with muted colors and grain",
            "noir": "Black & white high-contrast dramatic look",
            "warm": "Warm golden tones throughout",
            "cool": "Cool blue tones throughout",
            "vibrant": "Saturated vivid colors",
            "desaturated": "Muted, washed-out colors",
            "high_contrast": "Strong blacks and whites",
            "filmic_warm": "Film emulation with warm cast",
            "filmic_cool": "Film emulation with cool cast",
            "bleach_bypass": "Desaturated with retained highlights",
            "sepia": "Classic brown-toned photograph look",
            "cross_process": "Shifted color curves for surreal look",
            "technicolor": "Vibrant classic Hollywood colors",
            "moonlight": "Cool blue night scene",
            "golden_hour": "Warm sunset/sunrise tones",
            "flat_log": "Log-style flat profile for manual grading",
            "rec709": "Standard broadcast color space",
        }
        
        lines = []
        for preset in COLOR_GRADING_PRESETS:
            desc = preset_descriptions.get(preset, "")
            lines.append(f"- {preset}: {desc}")
        
        return "\n".join(lines)
    
    def _parse_llm_response(self, response: str) -> ParameterSuggestion:
        """Parse LLM response into color grading parameters."""
        try:
            # Extract JSON from response (LLM may include extra text)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Validate preset
            preset = data.get("preset")
            if preset not in COLOR_GRADING_PRESETS:
                logger.warning(f"LLM suggested invalid preset '{preset}', using 'cinematic'")
                preset = "cinematic"
            
            # Clamp parameters to valid ranges
            parameters = {
                "preset": preset,
                "intensity": max(0.0, min(1.0, float(data.get("intensity", 0.8)))),
                "temperature": max(-1.0, min(1.0, float(data.get("temperature", 0.0)))),
                "tint": max(-1.0, min(1.0, float(data.get("tint", 0.0)))),
                "saturation": max(0.0, min(2.0, float(data.get("saturation", 1.0)))),
                "contrast": max(0.0, min(2.0, float(data.get("contrast", 1.0)))),
                "brightness": max(-1.0, min(1.0, float(data.get("brightness", 0.0)))),
            }
            
            reasoning = data.get("reasoning", "No reasoning provided")
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.8))))
            
            return ParameterSuggestion(
                parameters=parameters,
                reasoning=reasoning,
                confidence=confidence
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Fallback to safe defaults
            return ParameterSuggestion(
                parameters={
                    "preset": "cinematic",
                    "intensity": 0.7,
                    "temperature": 0.0,
                    "tint": 0.0,
                    "saturation": 1.0,
                    "contrast": 1.0,
                    "brightness": 0.0,
                },
                reasoning="Fallback to cinematic preset due to parse error",
                confidence=0.5
            )


class StabilizationTuner(ParameterSuggester):
    """
    LLM-powered stabilization parameter tuner.
    
    Analyzes shake intensity and footage characteristics to suggest
    optimal stabilization parameters. Uses cgpu for analysis-driven tuning.
    
    Usage:
        tuner = StabilizationTuner()
        context = {
            "shake_score": 0.7,  # 0-1 scale
            "motion_type": "handheld",  # handheld, walking, driving, etc.
            "resolution": "1080p",
            "user_intent": "smooth cinematic motion"
        }
        suggestion = tuner.suggest(context)
    """
    
    def _build_suggestion_prompt(self, context: Dict[str, Any]) -> str:
        """Build stabilization tuning prompt."""
        shake_score = context.get("shake_score", 0.5)
        motion_type = context.get("motion_type", "unknown")
        resolution = context.get("resolution", "1080p")
        user_intent = context.get("user_intent", "smooth motion")
        
        prompt = f"""You are an expert video stabilization engineer. Analyze the following footage characteristics and suggest optimal stabilization parameters.

Footage Analysis:
- Shake Score: {shake_score:.2f} (0=stable, 1=very shaky)
- Motion Type: {motion_type}
- Resolution: {resolution}
- User Intent: {user_intent}

Stabilization Parameters (vidstab):
- smoothing: Camera motion smoothness (1-30, higher = smoother)
- shakiness: Shake detection sensitivity (1-10, higher = more aggressive)
- accuracy: Motion estimation accuracy (1-15, higher = slower but better)
- stepsize: Search step size (1-32, lower = more accurate)
- zoom: Static zoom percentage (-100 to 100, 0 = no zoom)
- optzoom: Optimal zoom (0=off, 1=static, 2=adaptive)
- crop: Border handling ("black" or "keep")

Task: Suggest optimal stabilization parameters.

Respond in JSON format:
{{
    "smoothing": <1-30>,
    "shakiness": <1-10>,
    "accuracy": <1-15>,
    "stepsize": <1-32>,
    "zoom": <-100 to 100>,
    "optzoom": <0-2>,
    "crop": "black" or "keep",
    "reasoning": "<explanation>",
    "confidence": <0.0-1.0>
}}

Guidelines:
- Higher shake_score → higher smoothing (15-25)
- Handheld → medium shakiness (5-7), Walking → higher (7-9)
- High resolution → lower stepsize (4-6) for better quality
- Use zoom cautiously (5-10% max) to avoid cropping too much
- optzoom=2 (adaptive) for variable shake intensity
- crop="keep" removes black borders
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> ParameterSuggestion:
        """Parse LLM response into stabilization parameters."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Clamp to valid ranges (same as StabilizeJob)
            parameters = {
                "smoothing": int(max(1, min(30, data.get("smoothing", 15)))),
                "shakiness": int(max(1, min(10, data.get("shakiness", 5)))),
                "accuracy": int(max(1, min(15, data.get("accuracy", 10)))),
                "stepsize": int(max(1, min(32, data.get("stepsize", 6)))),
                "zoom": max(-100.0, min(100.0, float(data.get("zoom", 0.0)))),
                "optzoom": int(max(0, min(2, data.get("optzoom", 1)))),
                "crop": data.get("crop", "keep") if data.get("crop") in ["black", "keep"] else "keep",
            }
            
            reasoning = data.get("reasoning", "No reasoning provided")
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.8))))
            
            return ParameterSuggestion(
                parameters=parameters,
                reasoning=reasoning,
                confidence=confidence
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse stabilization LLM response: {e}")
            
            # Safe defaults (balanced stabilization)
            return ParameterSuggestion(
                parameters={
                    "smoothing": 15,
                    "shakiness": 5,
                    "accuracy": 10,
                    "stepsize": 6,
                    "zoom": 0.0,
                    "optzoom": 1,
                    "crop": "keep",
                },
                reasoning="Fallback to balanced stabilization defaults",
                confidence=0.5
            )


# Convenience function for quick color grading suggestions
def suggest_color_grading(
    scene_description: str,
    user_intent: str = "cinematic",
    creative_director: Optional[CreativeDirector] = None
) -> ColorGradingParameters:
    """
    Quick color grading suggestion for a scene.
    
    Args:
        scene_description: Natural language description of scene
        user_intent: Desired creative look
        creative_director: Optional CreativeDirector instance
        
    Returns:
        ColorGradingParameters with suggested values
    """
    suggester = ColorGradingSuggester(creative_director)
    context = {
        "scene_description": scene_description,
        "user_intent": user_intent
    }
    suggestion = suggester.suggest(context)
    
    return ColorGradingParameters(**suggestion.parameters)


# Convenience function for stabilization tuning
def suggest_stabilization(
    shake_score: float,
    motion_type: str = "handheld",
    creative_director: Optional[CreativeDirector] = None
) -> StabilizationParameters:
    """
    Quick stabilization parameter suggestion.
    
    Args:
        shake_score: Shake intensity (0-1)
        motion_type: Type of camera motion
        creative_director: Optional CreativeDirector instance
        
    Returns:
        StabilizationParameters with suggested values
    """
    tuner = StabilizationTuner(creative_director)
    context = {
        "shake_score": shake_score,
        "motion_type": motion_type
    }
    suggestion = tuner.suggest(context)
    
    from .editing_parameters import StabilizationMethod, CropMode
    return StabilizationParameters(
        method=StabilizationMethod.VIDSTAB,
        smoothing=suggestion.parameters["smoothing"],
        shakiness=suggestion.parameters["shakiness"],
        accuracy=suggestion.parameters["accuracy"],
        stepsize=suggestion.parameters["stepsize"],
        zoom=int(suggestion.parameters["zoom"]),
        optzoom=suggestion.parameters["optzoom"],
        crop=CropMode(suggestion.parameters["crop"]),
    )
