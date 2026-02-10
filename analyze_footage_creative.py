#!/usr/bin/env python3
"""
Creative Footage Analysis & Cut Planning
Analyzes raw video material and generates an intelligent, creative cut plan
without relying on external LLM services (fallback-friendly).
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_INPUT_DIR = Path("/home/codeai/montage-ai/data/input")
MEDIA_FOLDER = Path("/home/codeai/montage-ai/data/input")

TARGET_DURATION_SECONDS = 45
ANALYSIS_SAMPLE_RATE = 1  # Analyze every Nth frame to speed up

# ============================================================================
# VIDEO METADATA EXTRACTION
# ============================================================================

def get_video_duration(filepath: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(filepath)
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Failed to get duration for {filepath}: {e}")
    return 0.0


def get_video_fps(filepath: Path) -> float:
    """Get video frame rate using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(filepath)
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            num, denom = result.stdout.strip().split('/')
            return float(num) / float(denom)
    except Exception as e:
        logger.warning(f"Failed to get FPS for {filepath}: {e}")
    return 24.0  # Default fallback


def get_video_resolution(filepath: Path) -> Tuple[int, int]:
    """Get video resolution (width, height) using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(filepath)
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            w, h = map(int, result.stdout.strip().split(','))
            return (w, h)
    except Exception as e:
        logger.warning(f"Failed to get resolution for {filepath}: {e}")
    return (1920, 1080)  # Default fallback


def analyze_video_brightness(filepath: Path, num_frames: int = 20) -> float:
    """Sample brightness levels across video (indicator of scene type)."""
    try:
        cmd = [
            "ffmpeg", "-i", str(filepath),
            "-vf", f"fps=1/({max(1, int(get_video_duration(filepath) / num_frames))})",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode == 0 and len(result.stdout) > 0:
            # Simple averaging of pixel values
            pixels = list(result.stdout)
            if pixels:
                avg_brightness = sum(pixels) / len(pixels) / 255.0
                return avg_brightness
    except Exception as e:
        logger.warning(f"Brightness analysis failed for {filepath}: {e}")
    
    return 0.5  # Neutral default


def detect_scene_changes(filepath: Path) -> List[float]:
    """Detect scene/cut points in a video (returns timestamps in seconds)."""
    try:
        # Use ffmpeg scenedetect-like approach via libscenedetect
        cmd = [
            "ffmpeg", "-i", str(filepath),
            "-vf", "select=gt(scene\\,0.3),fps=1",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Parse output (this is a simplified approach)
        # In real scenario, use PySceneDetect
        return []
    except Exception as e:
        logger.warning(f"Scene detection failed: {e}")
    return []


# ============================================================================
# CREATIVE ANALYSIS
# ============================================================================

def analyze_clip_characteristics(filepath: Path) -> Dict[str, Any]:
    """Analyze characteristics of a video clip."""
    return {
        "path": str(filepath),
        "filename": filepath.name,
        "duration": get_video_duration(filepath),
        "fps": get_video_fps(filepath),
        "resolution": get_video_resolution(filepath),
        "brightness": analyze_video_brightness(filepath),
        "size_mb": filepath.stat().st_size / (1024 * 1024),
    }


def categorize_clips(clips_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize clips by characteristics for creative arrangement."""
    categories = {
        "slow_motion": [],  # Low FPS or smoothness hints
        "high_energy": [],  # Short, high brightness
        "cinematic": [],    # Longer clips, mid-brightness
        "transition": [],   # Very short clips
        "establishing": [], # Longer, medium pacing
    }
    
    for clip in clips_data:
        duration = clip["duration"]
        brightness = clip["brightness"]
        
        # Simple heuristic categorization
        if duration < 1.0:
            categories["transition"].append(clip)
        elif duration > 5.0:
            categories["establishing"].append(clip)
        elif brightness > 0.7:
            categories["high_energy"].append(clip)
        elif brightness < 0.3:
            categories["cinematic"].append(clip)
        else:
            categories["transition"].append(clip)
    
    return categories


# ============================================================================
# CREATIVE CUT PLANNING (Heuristic-based, no LLM needed)
# ============================================================================

def generate_creative_cut_plan(
    clips_data: List[Dict[str, Any]],
    target_duration: float = 45.0,
    style: str = "dynamic_trailer"
) -> List[Dict[str, Any]]:
    """
    Generate a creative cut plan using heuristics.
    Styles: "dynamic_trailer", "cinematic", "montage", "highlights"
    """
    
    cuts = []
    elapsed_time = 0.0
    remaining_budget = target_duration
    used_clips = set()
    
    categorized = categorize_clips(clips_data)
    
    # Strategy 1: DYNAMIC TRAILER
    if style == "dynamic_trailer":
        # Phase 1: OPENING (5 seconds) - Establishing/cinematic
        logger.info("  Phase 1: Opening (5s)...")
        establishing = [c for c in categorized["establishing"] if c["filename"] not in used_clips]
        establishing.sort(key=lambda c: c["brightness"])  # Darker/moodier first
        
        if establishing:
            opening = establishing[0]
            cut_duration = min(5.0, opening["duration"] * 0.6)
            cuts.append({
                "clip_file": opening["filename"],
                "start": 0.0,
                "duration": cut_duration,
                "transition": "fade_in",
                "effect": "slow_zoom",
                "reason": "Cinematic opening"
            })
            elapsed_time += cut_duration
            remaining_budget -= cut_duration
            used_clips.add(opening["filename"])
        
        # Phase 2: BUILD-UP (10-15 seconds) - Medium-paced interesting clips
        logger.info("  Phase 2: Build-up (10-15s)...")
        buildup_clips = [c for c in clips_data if c["filename"] not in used_clips]
        buildup_clips = sorted(buildup_clips, key=lambda c: abs(c["brightness"] - 0.5))[:15]
        
        buildup_cut_pattern = [1.2, 1.0, 0.8, 1.1, 0.9, 1.0]
        pattern_idx = 0
        for clip in buildup_clips:
            if remaining_budget <= 15:
                break
            cut_duration = buildup_cut_pattern[pattern_idx % len(buildup_cut_pattern)]
            cut_duration = min(cut_duration, clip["duration"] * 0.7, remaining_budget - 15)
            
            cuts.append({
                "clip_file": clip["filename"],
                "start": 0.0,
                "duration": cut_duration,
                "transition": "cut",
                "effect": "none",
                "reason": f"Build-up segment {pattern_idx + 1}"
            })
            elapsed_time += cut_duration
            remaining_budget -= cut_duration
            used_clips.add(clip["filename"])
            pattern_idx += 1
        
        # Phase 3: CLIMAX (15-20 seconds) - Fast, intense cuts
        logger.info("  Phase 3: Climax (15-20s)...")
        climax_clips = [c for c in clips_data if c["filename"] not in used_clips]
        climax_clips = sorted(climax_clips, key=lambda c: c["brightness"], reverse=True)[:20]
        
        climax_pattern = [0.6, 0.5, 0.7, 0.55, 0.65, 0.5, 0.6, 0.4]
        pattern_idx = 0
        for clip in climax_clips:
            if remaining_budget <= 5:  # Keep 5s buffer for outro
                break
            cut_duration = climax_pattern[pattern_idx % len(climax_pattern)]
            cut_duration = min(cut_duration, clip["duration"] * 0.6, remaining_budget - 5)
            
            cuts.append({
                "clip_file": clip["filename"],
                "start": 0.0,
                "duration": cut_duration,
                "transition": "cut",
                "effect": "zoom_in" if pattern_idx % 2 == 0 else "none",
                "reason": f"Climax cut {pattern_idx + 1}"
            })
            elapsed_time += cut_duration
            remaining_budget -= cut_duration
            used_clips.add(clip["filename"])
            pattern_idx += 1
        
        # Phase 4: FINALE (5+ seconds) - Epic conclusion
        logger.info("  Phase 4: Finale...")
        finale_clips = [c for c in clips_data if c["filename"] not in used_clips]
        finale_clips = sorted(finale_clips, key=lambda c: c["duration"], reverse=True)
        
        for clip in finale_clips[:3]:
            if remaining_budget <= 1:
                break
            # Use remaining budget intelligently
            cut_duration = min(remaining_budget - 0.5, clip["duration"] * 0.5)
            if cut_duration > 0.3:
                cuts.append({
                    "clip_file": clip["filename"],
                    "start": 0.0,
                    "duration": cut_duration,
                    "transition": "fade",
                    "effect": "none",
                    "reason": "Epic finale"
                })
                elapsed_time += cut_duration
                remaining_budget -= cut_duration
                used_clips.add(clip["filename"])
    
    logger.info(f"Generated {len(cuts)} cuts, total duration: {elapsed_time:.1f}s (budget: {remaining_budget:.1f}s left)")
    
    return cuts


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    logger.info(f"🎬 Creative Footage Analysis Starting...")
    logger.info(f"Looking for videos in: {MEDIA_FOLDER}")
    
    # Find all video files
    video_extensions = {'.mp4', '.mov', '.mkv', '.avi', '.flv', '.wmv'}
    video_files = [
        f for f in MEDIA_FOLDER.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        logger.error(f"❌ No video files found in {MEDIA_FOLDER}")
        return
    
    logger.info(f"📼 Found {len(video_files)} video files")
    
    # Analyze each clip
    logger.info(f"🔍 Analyzing clips...")
    clips_data = []
    for video_file in sorted(video_files):
        logger.info(f"  → {video_file.name}")
        analysis = analyze_clip_characteristics(video_file)
        clips_data.append(analysis)
        logger.info(f"     Duration: {analysis['duration']:.1f}s, Brightness: {analysis['brightness']:.2f}")
    
    # Categorize
    logger.info(f"📊 Categorizing clips...")
    categorized = categorize_clips(clips_data)
    for category, clips in categorized.items():
        logger.info(f"   {category}: {len(clips)} clips")
    
    # Generate creative cut plan
    logger.info(f"🎨 Generating creative cut plan (target: {TARGET_DURATION_SECONDS}s)...")
    cut_plan = generate_creative_cut_plan(clips_data, TARGET_DURATION_SECONDS, "dynamic_trailer")
    
    # Export results
    output_json = Path("/tmp/creative_cut_plan.json")
    total_duration = sum(c["duration"] for c in cut_plan)
    
    result = {
        "style": "dynamic_trailer",
        "target_duration": TARGET_DURATION_SECONDS,
        "actual_duration": total_duration,
        "num_cuts": len(cut_plan),
        "clips_analyzed": len(clips_data),
        "cut_plan": cut_plan,
        "clip_analysis": clips_data,
        "categorization": {k: [c["filename"] for c in v] for k, v in categorized.items()}
    }
    
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✅ Creative cut plan generated!")
    logger.info(f"   Total cuts: {len(cut_plan)}")
    logger.info(f"   Total duration: {total_duration:.1f}s")
    logger.info(f"   Saved to: {output_json}")
    
    # Print summary
    print("\n" + "="*70)
    print("CREATIVE CUT PLAN SUMMARY")
    print("="*70)
    print(f"Style: Dynamic Trailer")
    print(f"Target Duration: {TARGET_DURATION_SECONDS}s")
    print(f"Actual Duration: {total_duration:.1f}s")
    print(f"Number of Cuts: {len(cut_plan)}\n")
    
    for i, cut in enumerate(cut_plan, 1):
        print(f"Cut #{i:2d}: {cut['clip_file']:40s} {cut['duration']:6.2f}s | {cut['reason']}")
    
    print("="*70)


if __name__ == "__main__":
    main()


def analyze_and_plan_creative_cut(target_duration: int = 45, style: str = "dynamic_trailer"):
    """
    Programmatic entry point for web UI integration.
    Returns JSON-serializable cut plan dict.
    """
    import json
    
    logger.info(f"🎬 Creative Footage Analysis (target: {target_duration}s, style: {style})")
    
    # Get all clips
    logger.info(f"📹 Scanning {MEDIA_FOLDER}...")
    video_files = sorted(MEDIA_FOLDER.glob("*.mp4")) + sorted(MEDIA_FOLDER.glob("*.mov"))
    if not video_files:
        logger.error(f"No video files found in {MEDIA_FOLDER}")
        return {"error": f"No video files found in {MEDIA_FOLDER}", "clips": [], "cuts": []}
    
    logger.info(f"   Found {len(video_files)} video files")
    
    # Analyze each clip
    clips_data = []
    for i, video_file in enumerate(video_files, 1):
        try:
            logger.info(f"   [{i}/{len(video_files)}] Analyzing {video_file.name}...")
            duration = get_video_duration(str(video_file))
            fps = get_video_fps(str(video_file))
            brightness = analyze_video_brightness(str(video_file))
            clips_data.append({
                "file": str(video_file),
                "filename": video_file.name,
                "duration": duration,
                "fps": fps,
                "brightness": brightness,
            })
        except Exception as e:
            logger.warning(f"   ⚠️  Error analyzing {video_file.name}: {str(e)}")
    
    if not clips_data:
        return {"error": "Failed to analyze any clips", "clips": [], "cuts": []}
    
    logger.info(f"📊 Categorizing clips...")
    categorized = categorize_clips(clips_data)
    
    # Generate creative cut plan
    logger.info(f"🎨 Generating creative cut plan (target: {target_duration}s)...")
    cut_plan = generate_creative_cut_plan(clips_data, target_duration, style)
    
    # Return as JSON-serializable dict
    return {
        "target_duration": target_duration,
        "style": style,
        "clips_analyzed": len(clips_data),
        "total_cuts": len(cut_plan),
        "cuts": cut_plan,
        "categorized": {k: len(v) for k, v in categorized.items()},
    }
