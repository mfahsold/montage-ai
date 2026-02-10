#!/usr/bin/env python3
"""
Render Creative Cut Plan
Converts the AI-generated cut plan into a final trailer video using FFmpeg.
"""

import json
import subprocess
import tempfile
from pathlib import Path
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
DATA_INPUT_DIR = Path("/home/codeai/montage-ai/data/input")
MUSIC_DIR = Path("/home/codeai/montage-ai/data/music")
OUTPUT_DIR = Path("/home/codeai/montage-ai/data/output")
CUT_PLAN_FILE = Path("/tmp/creative_cut_plan.json")

# ============================================================================
# FFMPEG RENDERING
# ============================================================================

def load_cut_plan(plan_file: Path) -> Dict:
    """Load the cut plan JSON."""
    with open(plan_file, 'r') as f:
        return json.load(f)


def normalize_clip(input_path: Path, output_path: Path, target_fps: float = 30.0) -> bool:
    """Normalize clip to constant frame rate and consistent codec."""
    logger.info(f"  Normalizing {input_path.name}...")
    
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),  # Force constant frame rate
        "-c:a", "aac",
        "-b:a", "128k",
        "-y",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def build_normalized_cuts(cuts: List[Dict], temp_dir: Path) -> str:
    """Build concat file with normalized clips (CFR + consistent codec)."""
    lines = []
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, cut in enumerate(cuts):
        clip_path = DATA_INPUT_DIR / cut["clip_file"]
        if not clip_path.exists():
            logger.warning(f"Clip not found: {clip_path}")
            continue
        
        # Prepare normalized segment
        norm_path = temp_dir / f"norm_{idx:03d}.mp4"
        
        if not norm_path.exists():
            if not normalize_clip(clip_path, norm_path):
                logger.warning(f"Failed to normalize {clip_path.name}")
                continue
        
        # Use normalized clip in concat
        lines.append(f"file '{norm_path.absolute()}'")
        lines.append(f"inpoint {cut['start']}")
        lines.append(f"outpoint {cut['start'] + cut['duration']}")
    
    return "\n".join(lines)


def build_concat_file(cuts: List[Dict]) -> str:
    """Build FFmpeg concat demuxer file (list of segments to concatenate)."""
    lines = []
    for cut in cuts:
        clip_path = DATA_INPUT_DIR / cut["clip_file"]
        if not clip_path.exists():
            logger.warning(f"Clip not found: {clip_path}")
            continue
        
        # For concat demuxer, use absolute path
        lines.append(f"file '{clip_path.absolute()}'")
        lines.append(f"inpoint {cut['start']}")
        lines.append(f"outpoint {cut['start'] + cut['duration']}")
    
    return "\n".join(lines)


def find_music_file() -> Path:
    """Find the first music file in music directory."""
    if not MUSIC_DIR.exists():
        logger.warning(f"Music directory not found: {MUSIC_DIR}")
        return None
    
    music_files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav")) + list(MUSIC_DIR.glob("*.aac"))
    if music_files:
        return music_files[0]
    return None


def render_creative_video(cut_plan: Dict, output_file: Path) -> bool:
    """
    Render the video using FFmpeg with the cut plan.
    Normalizes all clips first to ensure smooth playback (constant frame rate).
    """
    cuts = cut_plan["cut_plan"]
    target_duration = cut_plan["target_duration"]
    
    logger.info(f"🎬 Rendering creative cut video (with normalization)...")
    logger.info(f"   Cuts: {len(cuts)}")
    logger.info(f"   Target duration: {target_duration}s")
    logger.info(f"   Output: {output_file}")
    
    # Create temp directories
    temp_dir = Path(tempfile.gettempdir()) / "montage_normalized"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize all clips to CFR (this ensures smooth playback)
    logger.info(f"📊 Normalizing {len(cuts)} clips to constant 30 FPS...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_content = build_normalized_cuts(cuts, temp_dir)
        f.write(concat_content)
        concat_file = Path(f.name)
    
    try:
        # Find music
        music_file = find_music_file()
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
        ]
        
        # Add audio if available
        if music_file:
            cmd.extend(["-i", str(music_file)])
            cmd.extend([
                "-map", "0:v",
                "-map", "1:a",
                "-c:a", "aac",
                "-b:a", "192k",
            ])
        
        # Video encoding with CFR to prevent stuttering
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-profile:v", "high",
            "-level", "4.1",
            "-pix_fmt", "yuv420p",
            "-r", "30",  # Force constant output frame rate
        ])
        
        # Audio filtering for video with music
        if music_file:
            cmd.extend([
                "-af", f"atrim=0:{target_duration},volume=0.8",
            ])
        
        # Shortest to clip to target duration
        cmd.extend([
            "-shortest",
            "-movflags", "+faststart",
            str(output_file)
        ])
        
        logger.info(f"🎥 Concatenating and encoding...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed!")
            logger.error(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars
            return False
        
        logger.info(f"✅ Video rendered successfully!")
        
        # Verify output
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"   Output size: {size_mb:.1f} MB")
            
            # Verify output properties
            verify_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                str(output_file)
            ]
            result = subprocess.run(verify_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                fps = result.stdout.strip()
                logger.info(f"   Output frame rate: {fps}")
            
            return True
        
        return False
    
    finally:
        concat_file.unlink(missing_ok=True)
        # Keep normalized clips for potential reuse
        logger.info(f"   Temp normalized clips saved in: {temp_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("🎨 Creative Cut Renderer Starting...")
    
    # Load cut plan
    if not CUT_PLAN_FILE.exists():
        logger.error(f"Cut plan file not found: {CUT_PLAN_FILE}")
        return False
    
    cut_plan = load_cut_plan(CUT_PLAN_FILE)
    logger.info(f"Loaded cut plan: {cut_plan['num_cuts']} cuts, {cut_plan['actual_duration']:.1f}s")
    
    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "gallery_montage_creative_trailer_v1.mp4"
    
    # Render
    success = render_creative_video(cut_plan, output_file)
    
    if success:
        logger.info(f"✅ Done! Video saved to: {output_file}")
        return True
    else:
        logger.error(f"❌ Rendering failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


def render_with_plan(cut_plan: dict, plan_file: str = None) -> dict:
    """
    Programmatic entry point for web UI integration.
    Returns JSON-serializable result dict.
    
    Args:
        cut_plan: Dict with cuts array (or loads from plan_file if provided)
        plan_file: Optional path to cut_plan JSON file
    
    Returns:
        {"success": bool, "output_file": str, "file_size": int, "error": str}
    """
    import json
    
    try:
        # Load cut plan if file provided
        if plan_file:
            with open(plan_file, 'r') as f:
                cut_plan = json.load(f)
        
        if not cut_plan or 'cuts' not in cut_plan:
            return {"success": False, "error": "Invalid cut plan: missing 'cuts' array"}
        
        cuts = cut_plan['cuts']
        if not cuts:
            return {"success": False, "error": "No cuts in plan"}
        
        logger.info(f"🎬 Rendering creative cut ({len(cuts)} cuts, target duration)...")
        
        # Render video
        output_file = OUTPUT_DIR / "gallery_montage_creative_trailer_rendered.mp4"
        success = render_creative_video(cut_plan, output_file)
        
        if not success:
            return {"success": False, "error": "FFmpeg rendering failed"}
        
        if not output_file.exists():
            return {"success": False, "error": f"Output file not created: {output_file}"}
        
        file_size = output_file.stat().st_size
        logger.info(f"✅ Rendering complete! {file_size / (1024*1024):.1f}MB saved to {output_file}")
        
        return {
            "success": True,
            "output_file": str(output_file),
            "file_size": file_size,
            "total_cuts": len(cuts),
        }
        
    except Exception as e:
        logger.error(f"❌ Error during rendering: {str(e)}")
        return {"success": False, "error": str(e)}
