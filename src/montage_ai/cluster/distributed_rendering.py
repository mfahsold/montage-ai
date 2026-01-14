"""
Distributed Rendering - Shard-based parallel rendering across cluster nodes

Each worker renders a specific slice of the project timeline (Story Arc).
Segments are written to shared storage (NFS) and concatenated by the master.

Usage:
    # Render shards 0-3 of a total 4 shards
    python -m montage_ai.cluster.distributed_rendering \
        --clips-json /data/temp/clips_job123.json \
        --shard-index 0 --shard-count 4 \
        --output-dir /data/temp/segments_job123
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..logger import logger
from ..config import get_settings
from ..segment_writer import ProgressiveRenderer
from ..core.context import ClipMetadata

def render_shard(
    clips_metadata: List[Dict[str, Any]],
    shard_index: int,
    shard_count: int,
    output_dir: str,
    job_id: str,
    enable_xfade: bool = False,
    xfade_duration: float = 0.3,
    quality_profile: str = "standard"
):
    """
    Render a specific shard of the timeline.
    """
    settings = get_settings()
    
    # 1. Determine which clips this shard is responsible for
    total_clips = len(clips_metadata)
    clips_per_shard = (total_clips + shard_count - 1) // shard_count
    
    start_idx = shard_index * clips_per_shard
    end_idx = min(total_clips, (shard_index + 1) * clips_per_shard)
    
    if start_idx >= total_clips:
        logger.info(f"   ℹ️ Shard {shard_index} has no clips to process. Exiting.")
        return

    my_clips = clips_metadata[start_idx:end_idx]
    logger.info(f"   🎬 Shard {shard_index}/{shard_count} rendering {len(my_clips)} clips (indices {start_idx}-{end_idx-1})")

    # 2. Setup Progressive Renderer for this shard
    shard_output_dir = os.path.join(output_dir, f"shard_{shard_index}")
    os.makedirs(shard_output_dir, exist_ok=True)

    renderer = ProgressiveRenderer(
        batch_size=len(my_clips), 
        output_dir=shard_output_dir,
        job_id=f"{job_id}_s{shard_index}",
        enable_xfade=enable_xfade,
        xfade_duration=xfade_duration,
        ffmpeg_crf=settings.encoding.crf
    )

    # 3. Process Clips
    for i, clip_dict in enumerate(my_clips):
        # Reconstruct ClipMetadata
        meta = ClipMetadata(**clip_dict)
        
        logger.info(f"   ✂️ Shard {shard_index}: Rendering clip {i+1}/{len(my_clips)}: {os.path.basename(meta.source_path)}")
        temp_path = os.path.join(shard_output_dir, f"clip_{shard_index}_{i}.mp4")
        
        try:
            from ..ffmpeg_config import get_config
            cfg = get_config()
            
            hwaccel = []
            if os.environ.get("FFMPEG_HWACCEL") and os.environ.get("FFMPEG_HWACCEL") != "none":
                hwaccel = ["-hwaccel", os.environ.get("FFMPEG_HWACCEL")]

            import subprocess
            ffmpeg_cmd = [
                "ffmpeg", "-y"
            ] + hwaccel + [
                "-ss", f"{meta.start_time:.3f}", "-t", f"{meta.duration:.3f}",
                "-i", meta.source_path,
                "-vf", f"scale={settings.encoding.width}:{settings.encoding.height},setsar=1",
                "-c:v", os.environ.get("FFMPEG_ENCODER", cfg.codec),
                "-crf", str(settings.encoding.crf),
                "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                temp_path
            ]
            
            logger.debug(f"   Executing: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"   ❌ FFmpeg failed (code {result.returncode}): {result.stderr}")
                continue
                
            renderer.add_clip_path(temp_path)
            
        except Exception as e:
            logger.error(f"   ❌ Shard {shard_index}: Clip {i} failed: {e}")

    # 4. Finalize Shard Segment
    renderer.flush_batch()
    
    # Save shard report
    report = {
        "shard_index": shard_index,
        "clip_count": len(my_clips),
        "segments": [s.path for s in renderer.segment_writer.segments] if hasattr(renderer.segment_writer, 'segments') else []
    }
    
    with open(os.path.join(shard_output_dir, "shard_report.json"), "w") as f:
        json.dump(report, f)


def main():
    parser = argparse.ArgumentParser(description="Montage AI Distributed Renderer Worker")
    parser.add_argument("--clips-json", required=True, help="Path to JSON file containing ClipMetadata list")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", required=True, help="Shared directory for segments")
    parser.add_argument("--job-id", default="dist-render")
    parser.add_argument("--xfade", action="store_true", help="Enable crossfades")
    parser.add_argument("--xfade-duration", type=float, default=0.3)
    parser.add_argument("--quality", default="standard", choices=["preview", "standard", "high"])

    args = parser.parse_args()

    if not os.path.exists(args.clips_json):
        logger.error(f"Clips metadata file not found: {args.clips_json}")
        sys.exit(1)

    with open(args.clips_json, "r") as f:
        clips_metadata = json.load(f)

    render_shard(
        clips_metadata=clips_metadata,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        output_dir=args.output_dir,
        job_id=args.job_id,
        enable_xfade=args.xfade,
        xfade_duration=args.xfade_duration,
        quality_profile=args.quality
    )

if __name__ == "__main__":
    main()
