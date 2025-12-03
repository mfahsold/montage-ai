#!/bin/bash
# Test all new features inside Docker container

set -e

echo "======================================================================="
echo "MONTAGE-AI: Docker-Based Feature Test"
echo "======================================================================="

# Test 1: Syntax verification
echo ""
echo "TEST 1: Python Syntax Verification"
echo "-----------------------------------"
python3 -m py_compile /app/src/montage_ai/clip_selector.py && echo "✅ clip_selector.py OK" || echo "❌ clip_selector.py FAILED"
python3 -m py_compile /app/src/montage_ai/editor.py && echo "✅ editor.py OK" || echo "❌ editor.py FAILED"
python3 -m py_compile /app/src/montage_ai/ffmpeg_tools.py && echo "✅ ffmpeg_tools.py OK" || echo "❌ ffmpeg_tools.py FAILED"

# Test 2: Import verification
echo ""
echo "TEST 2: Module Import Verification"
echo "-----------------------------------"
python3 -c "from montage_ai.clip_selector import IntelligentClipSelector, ClipCandidate; print('✅ Intelligent Clip Selector imports OK')" || echo "❌ Import failed"
python3 -c "from montage_ai.editor import _check_vidstab_available, color_match_clips, _analyze_clip_brightness; print('✅ Enhancement functions import OK')" || echo "❌ Import failed"
python3 -c "from montage_ai.ffmpeg_tools import _color_grade; print('✅ Color grading functions import OK')" || echo "❌ Import failed"

# Test 3: vidstab availability
echo ""
echo "TEST 3: vidstab Availability Check"
echo "-----------------------------------"
python3 -c "from montage_ai.editor import _check_vidstab_available; print('vidstab available:', _check_vidstab_available())"

# Test 4: FFmpeg vidstab support
echo ""
echo "TEST 4: FFmpeg vidstab Filter Check"
echo "-----------------------------------"
if ffmpeg -filters 2>&1 | grep -q vidstab; then
    echo "✅ FFmpeg has vidstab filters"
    ffmpeg -filters 2>&1 | grep vidstab
else
    echo "⚠️  FFmpeg vidstab filters not found (may need rebuild)"
fi

# Test 5: Color presets
echo ""
echo "TEST 5: Color Grading Presets"
echo "-----------------------------------"
python3 << 'PYEOF'
from montage_ai.ffmpeg_tools import _color_grade

test_presets = ['cinematic', 'teal_orange', 'vintage', 'golden_hour', 'horror']
for preset in test_presets:
    result = _color_grade(preset)
    if result:
        print(f"✅ {preset}: {len(result)} filters")
    else:
        print(f"❌ {preset}: No filters returned")
PYEOF

# Test 6: LUT directory
echo ""
echo "TEST 6: LUT Directory Setup"
echo "-----------------------------------"
if [ -d "/data/luts" ]; then
    echo "✅ /data/luts directory exists"
    lut_count=$(find /data/luts -type f \( -name "*.cube" -o -name "*.3dl" -o -name "*.dat" \) 2>/dev/null | wc -l)
    echo "   Found $lut_count LUT file(s)"
else
    echo "⚠️  /data/luts directory not mounted"
fi

# Test 7: Intelligent Selector (basic test)
echo ""
echo "TEST 7: Intelligent Clip Selector (Basic Test)"
echo "-----------------------------------"
LLM_CLIP_SELECTION=true python3 << 'PYEOF'
from montage_ai.clip_selector import IntelligentClipSelector, ClipCandidate

selector = IntelligentClipSelector(style="dynamic", use_llm=True)
print(f"✅ Selector initialized")
print(f"   LLM enabled: {selector.use_llm}")
print(f"   LLM backend: {selector.llm is not None}")

# Create minimal test
candidates = [
    ClipCandidate(
        path="test1.mp4",
        start_time=0.0,
        duration=2.0,
        heuristic_score=80,
        metadata={'action': 'high', 'shot': 'close', 'energy': 0.8}
    ),
    ClipCandidate(
        path="test2.mp4",
        start_time=0.0,
        duration=2.0,
        heuristic_score=70,
        metadata={'action': 'medium', 'shot': 'wide', 'energy': 0.5}
    ),
]

context = {
    'current_energy': 0.6,
    'position': 'build',
    'previous_clips': [],
    'beat_position': 0
}

best, reason = selector.select_best_clip(candidates, context)
print(f"✅ Selection completed")
print(f"   Selected: {best.path}")
print(f"   Reason: {reason}")
PYEOF

echo ""
echo "======================================================================="
echo "TEST SUMMARY"
echo "======================================================================="
echo "✅ All tests completed. Check output above for any failures."
echo ""
echo "To run a full montage test with all features enabled:"
echo "  LLM_CLIP_SELECTION=true STABILIZE=true ENHANCE=true ./montage-ai.sh run dynamic"
