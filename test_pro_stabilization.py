#!/usr/bin/env python3
"""
🎬 PRO STABILIZATION TEST & DEMO

Demonstrates the new ultra-smooth, professional-grade stabilization
with aggressive motion smoothing and image enhancement.

Usage:
    python3 test_pro_stabilization.py [input_video] [output_video]
    
    Or with defaults:
    PRO_STABILIZE_AGGRESSIVE=true python3 test_pro_stabilization.py

Examples:
    # Standard stabilization (smart profile selection)
    ./test_pro_stabilization.py footage.mp4 output.mp4
    
    # Aggressive mode (maximum smoothing on all clips)
    PRO_STABILIZE_AGGRESSIVE=true ./test_pro_stabilization.py footage.mp4 output.mp4
    
    # Extreme profile (vlogger/action footage)
    PRO_STABILIZE_PROFILE=extreme ./test_pro_stabilization.py footage.mp4 output.mp4
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from montage_ai.pro_stabilization_engine import (
    ProStabilizationEngine,
    PROFILE_EXTREME,
    PROFILE_VLOG_ACTION,
    PROFILE_DOCUMENTARY,
    PROFILE_CINEMATIC,
    PROFILE_BROADCAST,
)
from montage_ai.render_pipeline_stabilizer import (
    RenderPipelineStabilizer,
    SmartStabilizationManager,
)


def demo_profiles():
    """Display all available stabilization profiles."""
    print("\n" + "="*80)
    print("🎬 AVAILABLE STABILIZATION PROFILES")
    print("="*80)
    
    profiles = [
        ("EXTREME", PROFILE_EXTREME, "Severely shaky footage (action cam, handheld runs)"),
        ("VLOG_ACTION", PROFILE_VLOG_ACTION, "High-motion handheld (vloggers, POV action)"),
        ("BROADCAST", PROFILE_BROADCAST, "Professional broadcast (balanced quality/smoothing)"),
        ("DOCUMENTARY", PROFILE_DOCUMENTARY, "Interview/doc (moderate hand-held)"),
        ("CINEMATIC", PROFILE_CINEMATIC, "Tripod shots (minimal shake, detail preservation)"),
    ]
    
    for name, profile, description in profiles:
        print(f"\n  📌 {name}")
        print(f"     Description: {description}")
        print(f"     Vidstab shakiness: {profile.vidstab_shakiness}/10 (0=light, 10=aggressive)")
        print(f"     Vidstab accuracy:  {profile.vidstab_accuracy}/15")
        print(f"     Deshake threshold: {profile.deshake_threshold:.1f}")
        print(f"     Motion smoothing:  {profile.motion_smooth_factor:.1%}")
        print(f"     Denoise strength:  {profile.denoise_strength:.1%}")


def test_single_clip(
    input_path: str,
    output_path: str,
    profile_name: str = "vlog_action",
):
    """Test stabilization on a single clip."""
    
    # Map profile name to object
    profiles = {
        "extreme": PROFILE_EXTREME,
        "vlog": PROFILE_VLOG_ACTION,
        "broadcast": PROFILE_BROADCAST,
        "documentary": PROFILE_DOCUMENTARY,
        "cinematic": PROFILE_CINEMATIC,
    }
    
    profile = profiles.get(profile_name, PROFILE_VLOG_ACTION)
    
    print("\n" + "="*80)
    print(f"🎬 STABILIZATION TEST - {profile.name.upper()}")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Profile: {profile_name}")
    print()
    
    # Initialize engine
    engine = ProStabilizationEngine(profile=profile)
    
    print("Filter chain being applied:")
    print("-" * 80)
    print("1. [DETECT] Motion analysis pass → motion vectors")
    print("2. [STABILIZE] vidstabtransform → corrects detected motion")
    print("3. [STABILIZE] deshake → micro-stabilization")
    print("4. [STABILIZE] deflicker → removes frame flicker")
    print("5. [SMOOTH] minterpolate → optical flow smoothing (cinema-grade)")
    print("6. [ENHANCE] hqdn3d → adaptive denoising")
    print("7. [ENHANCE] normalize → adaptive contrast")
    print("8. [ENHANCE] unsharp → detail recovery")
    print("9. [ENHANCE] saturate → color vibrancy boost")
    print("10. [ENHANCE] curves → gamma correction")
    print("-" * 80)
    print()
    
    # Run stabilization
    start_time = time.time()
    success, message = engine.stabilize_video_twopass(
        input_path,
        output_path,
        include_motion_smooth=True,
        include_color_correction=True,
    )
    elapsed = time.time() - start_time
    
    if success:
        print(f"\n✅ SUCCESS ({elapsed:.1f}s)")
        print(f"   Message: {message}")
        
        # Show output stats
        if Path(output_path).exists():
            output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"   Output size: {output_size_mb:.1f} MB")
        
        print("\n📊 Applied Enhancements:")
        print("   ✓ Ultra-smooth motion (optical flow + frame interpolation)")
        print("   ✓ Adaptive denoising (preserves fine details)")
        print("   ✓ Automatic contrast correction")
        print("   ✓ Professional color grading")
        print("   ✓ Flicker removal (temporal stability)")
        
    else:
        print(f"\n❌ FAILED ({elapsed:.1f}s)")
        print(f"   Error: {message}")
        return False
    
    return True


def test_pipeline_manager(
    input_path: str,
    output_path: str,
):
    """Test the high-level pipeline manager."""
    
    print("\n" + "="*80)
    print("🎬 PIPELINE MANAGER TEST (AUTO MODE)")
    print("="*80)
    
    # Create manager from environment
    manager = SmartStabilizationManager.from_environment()
    
    # Simulate clip metadata
    metadata = {
        "shake_score": 0.6,  # Moderate shake
        "motion_type": "dynamic",
        "duration": 5.0,
    }
    
    # Apply stabilization
    success, msg = manager.apply_stabilization(input_path, output_path, metadata)
    
    if success:
        print(f"\n✅ Pipeline stabilization complete: {msg}")
    else:
        print(f"\n❌ Pipeline stabilization failed: {msg}")
        return False
    
    # Show stats
    print("\n📊 Pipeline Statistics:")
    for key, value in manager.get_stats().items():
        print(f"   {key}: {value}")
    
    return True


def main():
    """Main entry point."""
    
    # Parse arguments
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    profile = sys.argv[3] if len(sys.argv) > 3 else "vlog_action"
    
    # Show profiles by default
    demo_profiles()
    
    # If input/output provided, run test
    if input_file and output_file:
        if not Path(input_file).exists():
            print(f"\n❌ Input file not found: {input_file}")
            sys.exit(1)
        
        print(f"\n🎬 Running stabilization test...")
        success = test_single_clip(input_file, output_file, profile)
        sys.exit(0 if success else 1)
    else:
        print("\n" + "="*80)
        print("USAGE:")
        print("="*80)
        print("  python3 test_pro_stabilization.py <input.mp4> <output.mp4> [profile]")
        print()
        print("PROFILES: extreme, vlog, broadcast, documentary, cinematic")
        print()
        print("EXAMPLES:")
        print("  # Default profile (vlog_action)")
        print("  python3 test_pro_stabilization.py shaky.mp4 smooth.mp4")
        print()
        print("  # Extreme profile (maximum smoothing)")
        print("  PRO_STABILIZE_AGGRESSIVE=true python3 test_pro_stabilization.py shaky.mp4 smooth.mp4")
        print("="*80)


if __name__ == "__main__":
    main()
