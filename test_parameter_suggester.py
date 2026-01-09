"""
Test for LLM-based parameter suggestion system.

This demonstrates how AI directors can intelligently tune editing parameters
based on scene analysis and creative intent.
"""

import sys
sys.path.insert(0, '/home/codeai/montage-ai/src')

from montage_ai.parameter_suggester import (
    ColorGradingSuggester,
    StabilizationTuner,
    suggest_color_grading,
    suggest_stabilization,
)
from montage_ai.creative_director import CreativeDirector
from montage_ai.editing_parameters import EditingParameters


def test_color_grading_suggestion():
    """Test color grading parameter suggestion."""
    print("=" * 80)
    print("TEST 1: Color Grading Suggestion")
    print("=" * 80)
    
    # Initialize suggester (will auto-detect cgpu/Ollama backend)
    suggester = ColorGradingSuggester()
    
    # Test scene 1: Sunset beach
    print("\n--- Scene 1: Sunset Beach ---")
    context = {
        "scene_description": "sunset beach scene with warm orange sky and blue ocean",
        "user_intent": "cinematic blockbuster feel",
        "dominant_colors": ["orange", "blue", "yellow"],
        "histogram": {
            "shadows": 0.25,
            "midtones": 0.50,
            "highlights": 0.25
        }
    }
    
    suggestion = suggester.suggest(context)
    print(f"Suggested Preset: {suggestion.parameters['preset']}")
    print(f"Intensity: {suggestion.parameters['intensity']:.2f}")
    print(f"Temperature: {suggestion.parameters['temperature']:.2f}")
    print(f"Saturation: {suggestion.parameters['saturation']:.2f}")
    print(f"Confidence: {suggestion.confidence:.2f}")
    print(f"Reasoning: {suggestion.reasoning}")
    
    # Test scene 2: Night city
    print("\n--- Scene 2: Night City ---")
    context = {
        "scene_description": "night city street with neon lights and rain",
        "user_intent": "cyberpunk noir atmosphere",
        "dominant_colors": ["blue", "purple", "cyan"],
        "histogram": {
            "shadows": 0.60,
            "midtones": 0.30,
            "highlights": 0.10
        }
    }
    
    suggestion = suggester.suggest(context)
    print(f"Suggested Preset: {suggestion.parameters['preset']}")
    print(f"Intensity: {suggestion.parameters['intensity']:.2f}")
    print(f"Temperature: {suggestion.parameters['temperature']:.2f}")
    print(f"Confidence: {suggestion.confidence:.2f}")
    print(f"Reasoning: {suggestion.reasoning}")


def test_stabilization_tuning():
    """Test stabilization parameter tuning."""
    print("\n" + "=" * 80)
    print("TEST 2: Stabilization Parameter Tuning")
    print("=" * 80)
    
    tuner = StabilizationTuner()
    
    # Test 1: Light shake (handheld)
    print("\n--- Footage 1: Light Handheld Shake ---")
    context = {
        "shake_score": 0.3,
        "motion_type": "handheld",
        "resolution": "1080p",
        "user_intent": "smooth cinematic motion"
    }
    
    suggestion = tuner.suggest(context)
    print(f"Smoothing: {suggestion.parameters['smoothing']}")
    print(f"Shakiness: {suggestion.parameters['shakiness']}")
    print(f"Accuracy: {suggestion.parameters['accuracy']}")
    print(f"Zoom: {suggestion.parameters['zoom']:.1f}%")
    print(f"Confidence: {suggestion.confidence:.2f}")
    print(f"Reasoning: {suggestion.reasoning}")
    
    # Test 2: Heavy shake (running)
    print("\n--- Footage 2: Heavy Shake (Running) ---")
    context = {
        "shake_score": 0.8,
        "motion_type": "running",
        "resolution": "4K",
        "user_intent": "action sequence stabilization"
    }
    
    suggestion = tuner.suggest(context)
    print(f"Smoothing: {suggestion.parameters['smoothing']}")
    print(f"Shakiness: {suggestion.parameters['shakiness']}")
    print(f"Accuracy: {suggestion.parameters['accuracy']}")
    print(f"Zoom: {suggestion.parameters['zoom']:.1f}%")
    print(f"Confidence: {suggestion.confidence:.2f}")
    print(f"Reasoning: {suggestion.reasoning}")


def test_convenience_functions():
    """Test convenience functions for quick suggestions."""
    print("\n" + "=" * 80)
    print("TEST 3: Convenience Functions")
    print("=" * 80)
    
    # Quick color grading suggestion
    print("\n--- Quick Color Grading ---")
    params = suggest_color_grading(
        scene_description="vintage cafe interior with warm lighting",
        user_intent="nostalgic film look"
    )
    print(f"Preset: {params.preset}")
    print(f"Intensity: {params.intensity}")
    print(f"Temperature: {params.temperature}")
    
    # Quick stabilization suggestion
    print("\n--- Quick Stabilization ---")
    params = suggest_stabilization(
        shake_score=0.5,
        motion_type="walking"
    )
    print(f"Smoothing: {params.smoothing}")
    print(f"Shakiness: {params.shakiness}")
    print(f"Crop Mode: {params.crop.value}")


def test_unified_parameters():
    """Test unified EditingParameters schema."""
    print("\n" + "=" * 80)
    print("TEST 4: Unified EditingParameters Schema")
    print("=" * 80)
    
    # Create default parameters
    params = EditingParameters()
    
    # Show defaults
    print("\n--- Default Parameters ---")
    print(f"Stabilization Method: {params.stabilization.method.value}")
    print(f"Stabilization Smoothing: {params.stabilization.smoothing}")
    print(f"Color Grading Preset: {params.color_grading.preset}")
    print(f"Color Intensity: {params.color_grading.intensity}")
    print(f"Pacing Speed: {params.pacing.speed.value}")
    print(f"Pacing Pattern: {params.pacing.pattern.value}")
    
    # Modify parameters
    params.stabilization.smoothing = 20
    params.color_grading.preset = "teal_orange"
    params.color_grading.intensity = 0.9
    
    # Validate
    try:
        params.validate()
        print("\n✓ Parameters validated successfully")
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
    
    # Serialize to dict
    print("\n--- Serialized to Dict ---")
    param_dict = params.to_dict()
    print(f"Stabilization: smoothing={param_dict['stabilization']['smoothing']}")
    print(f"Color Grading: preset={param_dict['color_grading']['preset']}, "
          f"intensity={param_dict['color_grading']['intensity']}")
    
    # Deserialize from dict
    print("\n--- Deserialized from Dict ---")
    restored = EditingParameters.from_dict(param_dict)
    print(f"Restored smoothing: {restored.stabilization.smoothing}")
    print(f"Restored preset: {restored.color_grading.preset}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("LLM-BASED PARAMETER SUGGESTION SYSTEM TESTS")
    print("Testing cgpu-robust AI director parameter tuning")
    print("=" * 80)
    
    try:
        # Test 1: Color grading
        test_color_grading_suggestion()
        
        # Test 2: Stabilization
        test_stabilization_tuning()
        
        # Test 3: Convenience functions
        test_convenience_functions()
        
        # Test 4: Unified parameters
        test_unified_parameters()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
