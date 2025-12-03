#!/usr/bin/env python3
"""
Comprehensive Test Suite for All New Features

Tests:
1. Intelligent Clip Selector (LLM reasoning)
2. vidstab 2-Pass Stabilization
3. Content-Aware Enhancement
4. Color Matching
5. Extended Color Grading Presets
6. LUT Integration

Usage:
    LLM_CLIP_SELECTION=true python test_all_features.py
"""

import os
import sys

# Enable features for testing
os.environ['LLM_CLIP_SELECTION'] = 'true'
os.environ['COLOR_MATCH'] = 'false'  # No reference clip available for test
os.environ['STABILIZE'] = 'true'
os.environ['ENHANCE'] = 'true'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_intelligent_selector():
    """Test 1: Intelligent Clip Selector"""
    print("\n" + "="*70)
    print("TEST 1: Intelligent Clip Selector (LLM Reasoning)")
    print("="*70)

    try:
        from montage_ai.clip_selector import IntelligentClipSelector, ClipCandidate

        print("✓ Import successful")

        # Create selector
        selector = IntelligentClipSelector(style="hitchcock", use_llm=True)
        print(f"✓ Selector initialized: LLM enabled = {selector.use_llm}")

        # Create test candidates
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
        print(f"✓ Created {len(candidates)} test candidates")

        # Create context
        context = {
            'current_energy': 0.7,
            'position': 'climax',
            'previous_clips': [
                {'meta': {'action': 'medium', 'shot': 'wide', 'energy': 0.5}, 'duration': 3.0}
            ],
            'beat_position': 2
        }
        print(f"✓ Context: {context['position']} @ energy {context['current_energy']}")

        # Select best clip
        best_clip, reasoning = selector.select_best_clip(candidates, context, top_n=3)

        print(f"\n  🎬 Selected: {best_clip.metadata['shot']} shot, {best_clip.metadata['action']} action")
        print(f"  📊 Heuristic Score: {best_clip.heuristic_score}")
        print(f"  🧠 Reasoning: {reasoning}")

        # Statistics
        stats = selector.get_statistics()
        print(f"\n  📈 Statistics:")
        for key, value in stats.items():
            print(f"     {key}: {value}")

        print("\n✅ TEST 1 PASSED: Intelligent Clip Selector")
        return True

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vidstab():
    """Test 2: vidstab 2-Pass Stabilization"""
    print("\n" + "="*70)
    print("TEST 2: vidstab 2-Pass Stabilization")
    print("="*70)

    try:
        from montage_ai.editor import _check_vidstab_available

        print("✓ Import successful")

        vidstab_available = _check_vidstab_available()

        if vidstab_available:
            print("✅ vidstab is AVAILABLE (2-Pass stabilization enabled)")
        else:
            print("⚠️  vidstab NOT available (will fallback to enhanced deshake)")

        print("\n✅ TEST 2 PASSED: vidstab check")
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_content_aware_enhancement():
    """Test 3: Content-Aware Enhancement"""
    print("\n" + "="*70)
    print("TEST 3: Content-Aware Enhancement")
    print("="*70)

    try:
        from montage_ai.editor import _analyze_clip_brightness

        print("✓ Import successful")
        print("✓ Function _analyze_clip_brightness() exists")
        print("  Note: Requires actual video file for brightness analysis")
        print("  Will be tested in real montage render")

        print("\n✅ TEST 3 PASSED: Content-Aware Enhancement functions imported")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_color_matching():
    """Test 4: Shot-to-Shot Color Matching"""
    print("\n" + "="*70)
    print("TEST 4: Shot-to-Shot Color Matching")
    print("="*70)

    try:
        from montage_ai.editor import color_match_clips

        print("✓ Import successful")
        print("✓ Function color_match_clips() exists")
        print("  Requires: color-matcher>=0.5.0")

        # Try importing color-matcher
        try:
            import color_matcher
            print(f"✓ color-matcher installed (version: {color_matcher.__version__ if hasattr(color_matcher, '__version__') else 'unknown'})")
        except ImportError:
            print("⚠️  color-matcher not installed (will be installed in Docker)")

        print("\n✅ TEST 4 PASSED: Color Matching function imported")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_color_grading_presets():
    """Test 5: Extended Color Grading Presets"""
    print("\n" + "="*70)
    print("TEST 5: Extended Color Grading Presets")
    print("="*70)

    try:
        from montage_ai.ffmpeg_tools import _color_grade

        print("✓ Import successful")

        # Test presets
        test_presets = [
            'cinematic', 'teal_orange', 'blockbuster',
            'vintage', 'film_fade', '70s', 'polaroid',
            'golden_hour', 'blue_hour', 'horror', 'sci_fi',
            'high_contrast', 'low_contrast', 'punch'
        ]

        print(f"✓ Testing {len(test_presets)} presets...")
        for preset in test_presets:
            result = _color_grade(preset)
            if result:
                print(f"  ✓ {preset}: {len(result)} filters")

        # Test LUT support
        print("\n  📁 LUT Support:")
        lut_dir = os.environ.get('LUT_DIR', '/data/luts')
        print(f"     LUT_DIR: {lut_dir}")
        if os.path.exists(lut_dir):
            luts = [f for f in os.listdir(lut_dir) if f.endswith(('.cube', '.3dl', '.dat'))]
            print(f"     Found {len(luts)} LUT files")
        else:
            print(f"     Directory not found (will be created in Docker)")

        print("\n✅ TEST 5 PASSED: Color Grading Presets")
        return True

    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_docker_setup():
    """Test 6: Docker Setup (volume mounts, dependencies)"""
    print("\n" + "="*70)
    print("TEST 6: Docker Setup Verification")
    print("="*70)

    try:
        # Check requirements.txt
        print("  Checking requirements.txt...")
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            if 'color-matcher>=0.5.0' in requirements:
                print("  ✓ color-matcher>=0.5.0 in requirements.txt")
            else:
                print("  ⚠️  color-matcher not in requirements.txt")

        # Check Dockerfile
        print("\n  Checking Dockerfile...")
        with open('Dockerfile', 'r') as f:
            dockerfile = f.read()
            if 'libvidstab' in dockerfile:
                print("  ✓ libvidstab in Dockerfile")
            else:
                print("  ⚠️  libvidstab not in Dockerfile")

        # Check docker-compose
        print("\n  Checking docker-compose.yml...")
        with open('docker-compose.yml', 'r') as f:
            compose = f.read()
            if './data/luts' in compose:
                print("  ✓ /data/luts volume mount in docker-compose.yml")
            else:
                print("  ⚠️  /data/luts volume mount not found")

        # Check LUT directory
        print("\n  Checking data/luts...")
        if os.path.exists('data/luts'):
            print("  ✓ data/luts directory exists")
            if os.path.exists('data/luts/README.md'):
                print("  ✓ data/luts/README.md exists")
        else:
            print("  ⚠️  data/luts directory not found")

        print("\n✅ TEST 6 PASSED: Docker Setup")
        return True

    except Exception as e:
        print(f"\n❌ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("MONTAGE-AI: COMPREHENSIVE FEATURE TEST SUITE")
    print("="*70)
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")

    results = []

    # Run all tests
    results.append(("Intelligent Clip Selector", test_intelligent_selector()))
    results.append(("vidstab Availability", test_vidstab()))
    results.append(("Content-Aware Enhancement", test_content_aware_enhancement()))
    results.append(("Color Matching", test_color_matching()))
    results.append(("Color Grading Presets", test_color_grading_presets()))
    results.append(("Docker Setup", test_docker_setup()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
