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
from ..core.clip_processor import process_clip_task
from ..resource_manager import get_resource_manager


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

    Args:
        clips_metadata: Full list of clip metadata for the project
        shard_index: This worker's index
        shard_count: Total number of workers
        output_dir: Shared output directory for segments
        job_id: Unique job ID
        enable_xfade: Enable transitions
        xfade_duration: Transition duration
        quality_profile: 'preview', 'standard', or 'high'
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
    # We want each shard to produce its own sequence of segments
    shard_output_dir = os.path.join(output_dir, f"shard_{shard_index}")
    os.makedirs(shard_output_dir, exist_ok=True)

    renderer = ProgressiveRenderer(
        batch_size=len(my_clips), # We want one big segment for the shard if possible, or use standard batching
        output_dir=shard_output_dir,
        job_id=f"{job_id}_s{shard_index}",
        enable_xfade=enable_xfade,
        xfade_duration=xfade_duration,
        ffmpeg_crf=settings.encoding.crf
    )

    # 3. Process Clips
    resource_manager = get_resource_manager()
    
    # In distributed mode, we can use more local threads since the node only handles one shard
    # max_threads = resource_manager.get_optimal_threads()
    
    render_start = time.time()
    for i, clip_dict in enumerate(my_clips):
        # Reconstruct ClipMetadata
        meta = ClipMetadata(**clip_dict)
        
        # We need to simulate the loop from MontageBuilder
        # Note: In distributed renderer, we assume the clip is already analyzed
        # and we just need to run the FFmpeg render part.
        
        logger.info(f"   ✂️ Shard {shard_index}: Rendering clip {i+1}/{len(my_clips)}: {os.path.basename(meta.source_path)}")
        
        # Simulating process_clip_task (simplified for worker)
        # Note: process_clip_task returns a path to a temporary video file
        temp_path = os.path.join(shard_output_dir, f"clip_{shard_index}_{i}.mp4")
        
        # If the worker script is running in a container, it has access to /data/input via NFS
        try:
            # We don't have the original 'scene' dict anymore, but ClipMetadata has source_path, start_time, duration.
            # We can use FFmpeg directly to extract the subclip.
            
            # Use the logic from clip_processor but simplified
            from ..ffmpeg_config import get_config
            cfg = get_config()
            
            # Build FFmpeg command for the subclip
            cmd = cfg.video_params() # Basic params
            
            # Extract subclip
            import subprocess
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-ss", str(meta.start_time), "-t", str(meta.duration),
                "-i", meta.source_path,
                "-vf", f"scale={settings.encoding.width}:{settings.encoding.height}",
                "-c:v", cfg.codec, "-crf", str(settings.encoding.crf), "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                temp_path
            ]
            
            logger.debug(f"   Executing: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
            renderer.add_clip_path(temp_path)
            
        except Exception as e:
            logger.error(f"   ❌ Shard {shard_index}: Clip {i} failed: {e}")

    # 4. Finalize Shard Segment
    # This will create one or more segments for this shard
    shard_segments = renderer.flush_batch()
    
    render_duration = time.time() - render_start
    logger.info(f"   ✅ Shard {shard_index} completed in {render_duration:.1f}s")
    
    # Save shard report
    report = {
        "shard_index": shard_index,
        "clip_count": len(my_clips),
        "duration": render_duration,
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
