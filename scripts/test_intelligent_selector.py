#!/usr/bin/env python3
"""
Test script for Intelligent Clip Selector

Usage:
    LLM_CLIP_SELECTION=true python test_intelligent_selector.py
"""

import os
import sys

# Enable LLM clip selection for testing
os.environ['LLM_CLIP_SELECTION'] = 'true'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from montage_ai.clip_selector import IntelligentClipSelector, ClipCandidate


def test_intelligent_selector():
    print("🧪 Testing Intelligent Clip Selector\n")

    # Create selector
    print("1. Initializing selector with 'hitchcock' style...")
    selector = IntelligentClipSelector(style="hitchcock", use_llm=True)

    # Create test candidates
    print("\n2. Creating test candidates...")
    candidates = [
        ClipCandidate(
            path="/data/input/clip1.mp4",
            start_time=5.0,
            duration=3.0,
            heuristic_score=85,
            metadata={'action': 'high', 'shot': 'close', 'energy': 0.8}
        ),
        ClipCandidate(
            path="/data/input/clip2.mp4",
            start_time=10.0,
            duration=2.5,
            heuristic_score=78,
            metadata={'action': 'medium', 'shot': 'wide', 'energy': 0.6}
        ),
        ClipCandidate(
            path="/data/input/clip3.mp4",
            start_time=15.0,
            duration=4.0,
            heuristic_score=72,
            metadata={'action': 'low', 'shot': 'medium', 'energy': 0.4}
        ),
    ]

    for i, c in enumerate(candidates, 1):
        print(f"   Candidate {i}: {c.metadata['shot']} shot, {c.metadata['action']} action, "
              f"energy={c.metadata['energy']}, score={c.heuristic_score}")

    # Create context
    print("\n3. Building context...")
    context = {
        'current_energy': 0.7,
        'position': 'climax',
        'previous_clips': [
            {'meta': {'action': 'medium', 'shot': 'wide', 'energy': 0.5}, 'duration': 3.0}
        ],
        'beat_position': 2
    }
    print(f"   Energy: {context['current_energy']}")
    print(f"   Position: {context['position']}")
    print(f"   Previous clips: {len(context['previous_clips'])}")

    # Select best clip
    print("\n4. Querying LLM for best clip selection...")
    try:
        best_clip, reasoning = selector.select_best_clip(candidates, context, top_n=3)

        print(f"\n✅ LLM Selection Complete!")
        print(f"   Selected: {best_clip.metadata['shot']} shot, {best_clip.metadata['action']} action")
        print(f"   Heuristic Score: {best_clip.heuristic_score}")
        print(f"   Reasoning: {reasoning}")

    except Exception as e:
        print(f"\n❌ Error during selection: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get statistics
    print("\n5. Selector Statistics:")
    stats = selector.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✅ Test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_intelligent_selector()
    sys.exit(0 if success else 1)
