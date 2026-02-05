"""
Distributed Scene Detection - Parallel Processing Across Cluster Nodes

Supports two sharding modes:
1. Time-based: Split single video into time segments (for large videos)
2. File-based: Distribute multiple videos across workers

Usage:
    # Time-based sharding (single video)
    python -m montage_ai.cluster.distributed_scene_detection \
        --video /data/input/large_video.mp4 \
        --shard-index 0 --shard-count 4

    # File-based sharding (multiple videos)
    python -m montage_ai.cluster.distributed_scene_detection \
        --videos /data/input/v1.mp4,/data/input/v2.mp4 \
        --shard-index 0 --shard-count 2

    # Aggregation (after all shards complete)
    python -m montage_ai.cluster.distributed_scene_detection \
        --aggregate --job-id scene-detect-abc123
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from ..scene_analysis import SceneDetector, Scene
from ..video_metadata import probe_metadata
from ..logger import logger


@dataclass
class ShardResult:
    """Result from a single shard's scene detection."""
    shard_index: int
    shard_count: int
    video_path: str
    time_range: Optional[Tuple[float, float]]  # (start, end) for time-based
    scenes: List[dict]
    duration_processed: float
    processing_time: float


def get_shard_time_range(
    duration: float,
    shard_index: int,
    shard_count: int,
    overlap_seconds: float = 2.0
) -> Tuple[float, float]:
    """
    Calculate time range for a shard with overlap for boundary detection.

    Args:
        duration: Total video duration in seconds
        shard_index: This shard's index (0-based)
        shard_count: Total number of shards
        overlap_seconds: Overlap between shards to catch boundary scenes

    Returns:
        (start_time, end_time) tuple
    """
    segment_duration = duration / shard_count

    start = max(0, shard_index * segment_duration - overlap_seconds)
    end = min(duration, (shard_index + 1) * segment_duration + overlap_seconds)

    return (start, end)


def get_shard_files(
    video_paths: List[str],
    shard_index: int,
    shard_count: int
) -> List[str]:
    """
    Get files assigned to this shard for file-based distribution.

    Args:
        video_paths: All video paths to process
        shard_index: This shard's index (0-based)
        shard_count: Total number of shards

    Returns:
        List of video paths for this shard
    """
    # Round-robin distribution for balanced load
    return [
        path for i, path in enumerate(video_paths)
        if i % shard_count == shard_index
    ]


def detect_scenes_shard(
    video_path: str,
    shard_index: int,
    shard_count: int,
    time_range: Optional[Tuple[float, float]] = None,
    threshold: float = 27.0
) -> ShardResult:
    """
    Run scene detection for a single shard.

    Args:
        video_path: Path to video file
        shard_index: This shard's index
        shard_count: Total number of shards
        time_range: Optional (start, end) for time-based sharding
        threshold: Scene detection threshold

    Returns:
        ShardResult with detected scenes
    """
    import time
    start_time = time.time()

    detector = SceneDetector(threshold=threshold)

    # Get video metadata
    meta = probe_metadata(video_path)
    if isinstance(meta, dict):
        duration = meta.get("duration", 0.0)
    else:
        duration = getattr(meta, "duration", 0.0) if meta is not None else 0.0

    if time_range:
        # Time-based sharding - detect in range
        # For now, detect full video and filter (TransNetV2 is fast enough)
        # TODO: Implement seek-based detection for very large videos
        logger.info(
            f"🔍 Shard {shard_index}/{shard_count}: Detecting scenes in "
            f"{os.path.basename(video_path)} [{time_range[0]:.1f}s - {time_range[1]:.1f}s]"
        )

        all_scenes = detector.detect(video_path)

        # Filter to time range
        scenes = [
            s for s in all_scenes
            if s.start >= time_range[0] and s.end <= time_range[1]
        ]
        duration_processed = time_range[1] - time_range[0]
    else:
        # Full video detection
        logger.info(
            f"🔍 Shard {shard_index}/{shard_count}: Detecting scenes in "
            f"{os.path.basename(video_path)}"
        )
        scenes = detector.detect(video_path)
        duration_processed = duration

    processing_time = time.time() - start_time

    logger.info(
        f"✅ Shard {shard_index}: Found {len(scenes)} scenes in "
        f"{duration_processed:.1f}s ({processing_time:.1f}s processing)"
    )

    return ShardResult(
        shard_index=shard_index,
        shard_count=shard_count,
        video_path=video_path,
        time_range=time_range,
        scenes=[s.to_dict() for s in scenes],
        duration_processed=duration_processed,
        processing_time=processing_time
    )


def save_shard_result(result: ShardResult, output_dir: str, job_id: str) -> str:
    """Save shard result to JSON file for later aggregation."""
    output_path = Path(output_dir) / f"shard_{job_id}_{result.shard_index}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "shard_index": result.shard_index,
        "shard_count": result.shard_count,
        "video_path": result.video_path,
        "time_range": result.time_range,
        "scenes": result.scenes,
        "duration_processed": result.duration_processed,
        "processing_time": result.processing_time
    }

    from ..utils import NumpyEncoder
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    logger.info(f"📁 Saved shard result to {output_path}")
    return str(output_path)


def aggregate_shard_results(output_dir: str, job_id: str) -> List[dict]:
    """
    Aggregate results from all shards into unified scene list.

    Handles:
    - Deduplication of overlapping boundary scenes
    - Sorting by start time
    - Merging adjacent scenes from different shards
    """
    output_path = Path(output_dir)
    shard_files = list(output_path.glob(f"shard_{job_id}_*.json"))

    if not shard_files:
        logger.warning(f"No shard results found for job {job_id}")
        return []

    logger.info(f"📊 Aggregating {len(shard_files)} shard results...")

    all_scenes = []
    total_processing_time = 0.0

    for shard_file in sorted(shard_files):
        with open(shard_file) as f:
            data = json.load(f)
            all_scenes.extend(data["scenes"])
            total_processing_time += data["processing_time"]

    # Deduplicate scenes (same video, overlapping times)
    unique_scenes = []
    for scene in sorted(all_scenes, key=lambda s: (s["path"], s["start"])):
        # Check if scene overlaps with last added scene from same video
        if unique_scenes and unique_scenes[-1]["path"] == scene["path"]:
            last = unique_scenes[-1]
            # If scenes overlap significantly, merge them
            if scene["start"] < last["end"] + 0.5:  # 0.5s tolerance
                last["end"] = max(last["end"], scene["end"])
                continue
        unique_scenes.append(scene)

    logger.info(
        f"✅ Aggregation complete: {len(unique_scenes)} unique scenes "
        f"(from {len(all_scenes)} total, {total_processing_time:.1f}s total processing)"
    )

    # Save aggregated result
    aggregated_path = output_path / f"scenes_{job_id}.json"
    with open(aggregated_path, "w") as f:
        json.dump({
            "job_id": job_id,
            "shard_count": len(shard_files),
            "total_scenes": len(unique_scenes),
            "total_processing_time": total_processing_time,
            "scenes": unique_scenes
        }, f, indent=2)

    logger.info(f"📁 Saved aggregated scenes to {aggregated_path}")
    return unique_scenes


def main():
    """CLI entry point for distributed scene detection."""
    parser = argparse.ArgumentParser(
        description="Distributed scene detection for cluster processing"
    )

    parser.add_argument(
        "--video", "-v",
        help="Single video path (for time-based sharding)"
    )
    parser.add_argument(
        "--videos",
        help="Comma-separated video paths (for file-based sharding)"
    )
    parser.add_argument(
        "--shard-index", "-i",
        type=int,
        default=int(os.environ.get("SHARD_INDEX", 0)),
        help="Shard index (0-based)"
    )
    parser.add_argument(
        "--shard-count", "-c",
        type=int,
        default=int(os.environ.get("SHARD_COUNT", 1)),
        help="Total shard count"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=os.environ.get("OUTPUT_DIR", "/data/output/scene_cache"),
        help="Output directory for shard results"
    )
    parser.add_argument(
        "--job-id", "-j",
        default=os.environ.get("JOB_ID", "default"),
        help="Job ID for result aggregation"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=27.0,
        help="Scene detection threshold"
    )
    parser.add_argument(
        "--aggregate", "-a",
        action="store_true",
        help="Aggregate shard results (run after all shards complete)"
    )
    parser.add_argument(
        "--shard-mode",
        action="store_true",
        help="Run in K8s indexed job mode (uses env vars)"
    )

    args = parser.parse_args()

    # Handle aggregation mode
    if args.aggregate:
        scenes = aggregate_shard_results(args.output_dir, args.job_id)
        print(f"Aggregated {len(scenes)} scenes")
        return 0

    # Determine video(s) to process
    if args.video:
        # Time-based sharding for single large video
        meta = probe_metadata(args.video)
        duration = meta.get("duration", 0.0)

        time_range = get_shard_time_range(
            duration,
            args.shard_index,
            args.shard_count
        )

        result = detect_scenes_shard(
            args.video,
            args.shard_index,
            args.shard_count,
            time_range=time_range,
            threshold=args.threshold
        )

        save_shard_result(result, args.output_dir, args.job_id)

    elif args.videos:
        # File-based sharding for multiple videos
        video_paths = [v.strip() for v in args.videos.split(",") if v.strip()]
        shard_videos = get_shard_files(video_paths, args.shard_index, args.shard_count)

        for video_path in shard_videos:
            result = detect_scenes_shard(
                video_path,
                args.shard_index,
                args.shard_count,
                threshold=args.threshold
            )
            save_shard_result(result, args.output_dir, args.job_id)

    elif args.shard_mode:
        # K8s indexed job mode - get videos from env or glob
        videos_env = os.environ.get("VIDEOS")
        video_path_env = os.environ.get("VIDEO_PATH")
        
        if videos_env:
            video_paths = [v.strip() for v in videos_env.split(",") if v.strip()]
        elif video_path_env and os.path.isfile(video_path_env):
            video_paths = [video_path_env]
        else:
            video_dir = video_path_env or "/data/input"
            video_paths = [
                str(p) for p in Path(video_dir).glob("*.mp4")
            ]

        if len(video_paths) == 1:
            # Automatic time-based sharding for single video in cluster mode
            video_path = video_paths[0]
            meta = probe_metadata(video_path)
            if isinstance(meta, dict):
                duration = meta.get("duration", 0.0)
            else:
                duration = getattr(meta, "duration", 0.0) if meta is not None else 0.0

            time_range = get_shard_time_range(
                duration,
                args.shard_index,
                args.shard_count
            )

            logger.info(
                f"🔧 K8s shard mode (Time-based): Processing {video_path} "
                f"range {time_range[0]:.2f}-{time_range[1]:.2f}s "
                f"(shard {args.shard_index}/{args.shard_count})"
            )

            result = detect_scenes_shard(
                video_path,
                args.shard_index,
                args.shard_count,
                time_range=time_range,
                threshold=args.threshold
            )
            save_shard_result(result, args.output_dir, args.job_id)
        else:
            # File-based sharding
            shard_videos = get_shard_files(video_paths, args.shard_index, args.shard_count)
            logger.info(
                f"🔧 K8s shard mode (File-based): Processing {len(shard_videos)} videos "
                f"(shard {args.shard_index}/{args.shard_count})"
            )

            for video_path in shard_videos:
                result = detect_scenes_shard(
                    video_path,
                    args.shard_index,
                    args.shard_count,
                    threshold=args.threshold
                )
                save_shard_result(result, args.output_dir, args.job_id)

    else:
        parser.error("Must specify --video, --videos, or --shard-mode")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
