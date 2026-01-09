"""
Export CLI for Montage AI - export montage timeline to NLE formats.

Usage:
    python -m montage_ai.export.cli [OPTIONS]
    
    Example:
    python -m montage_ai.export.cli \\
        --output-dir /data/output \\
        --formats otio edl premiere \\
        --project-name "My Project"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

from montage_ai.export import export_to_nle, create_export_summary
from montage_ai.export.otio_builder import TimelineClipInfo
from montage_ai.editing_parameters import EditingParameters
from montage_ai.logger import logger


def load_timeline_manifest(manifest_path: Path) -> List[TimelineClipInfo]:
    """
    Load timeline manifest JSON file.
    
    Expected format:
    {
        "clips": [
            {
                "source_path": "/data/input/clip1.mp4",
                "in_time": 0.0,
                "out_time": 5.0,
                "duration": 5.0,
                "sequence_number": 1,
                "applied_effects": {...},
                "recommended_effects": {...},
                "confidence_scores": {...}
            },
            ...
        ],
        "beat_timecodes": [[1.0, "beat_1"], [2.0, "beat_2"]],
        "section_markers": [[0.0, "intro"], [2.0, "build"]]
    }
    """
    logger.info(f"Loading timeline manifest: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    clips = []
    for clip_data in manifest.get("clips", []):
        clip = TimelineClipInfo(
            source_path=clip_data["source_path"],
            in_time=clip_data["in_time"],
            out_time=clip_data["out_time"],
            duration=clip_data["duration"],
            sequence_number=clip_data["sequence_number"],
            applied_effects=clip_data.get("applied_effects", {}),
            recommended_effects=clip_data.get("recommended_effects"),
            confidence_scores=clip_data.get("confidence_scores"),
            beat_markers=clip_data.get("beat_markers")
        )
        clips.append(clip)
    
    logger.info(f"Loaded {len(clips)} clips from manifest")
    return clips


def load_editing_parameters(params_path: Path) -> EditingParameters:
    """Load EditingParameters from JSON."""
    logger.info(f"Loading editing parameters: {params_path}")
    
    with open(params_path) as f:
        params_dict = json.load(f)
    
    params = EditingParameters()
    
    # Populate from JSON
    if "stabilization" in params_dict:
        for key, value in params_dict["stabilization"].items():
            setattr(params.stabilization, key, value)
    
    if "color_grading" in params_dict:
        for key, value in params_dict["color_grading"].items():
            setattr(params.color_grading, key, value)
    
    if "pacing" in params_dict:
        for key, value in params_dict["pacing"].items():
            setattr(params.pacing, key, value)
    
    return params


def main():
    """CLI entry point for export."""
    parser = argparse.ArgumentParser(
        description="Export Montage AI timeline to NLE formats (OTIO, EDL, Premiere, AAF)"
    )
    
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to timeline manifest JSON file (from montage render)"
    )
    
    parser.add_argument(
        "--params",
        type=Path,
        help="Path to editing parameters JSON file (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/output"),
        help="Output directory for export files (default: /data/output)"
    )
    
    parser.add_argument(
        "--project-name",
        type=str,
        default="Montage AI Project",
        help="Project name for timeline (default: 'Montage AI Project')"
    )
    
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["otio", "edl"],
        choices=["otio", "edl", "premiere", "aaf", "params_json"],
        help="Export formats (default: otio edl)"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (default: 30.0)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Video width in pixels (default: 1920)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Video height in pixels (default: 1080)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    logger.info("🎬 Montage AI Export to NLE")
    logger.info(f"   Project: {args.project_name}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   Formats: {', '.join(args.formats)}")
    
    # Load manifest
    if not args.manifest.exists():
        logger.error(f"Manifest file not found: {args.manifest}")
        return 1
    
    try:
        clips = load_timeline_manifest(args.manifest)
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        return 1
    
    # Load editing parameters (optional)
    params = EditingParameters()
    if args.params and args.params.exists():
        try:
            params = load_editing_parameters(args.params)
        except Exception as e:
            logger.warning(f"Failed to load parameters, using defaults: {e}")
    
    # Export
    try:
        results = export_to_nle(
            timeline_clips=clips,
            editing_params=params,
            output_dir=args.output_dir,
            formats=args.formats,
            project_name=args.project_name,
            fps=args.fps
        )
        
        # Print summary
        summary = create_export_summary(results)
        logger.info("\n" + summary)
        
        # Count successes
        success_count = sum(1 for success, _ in results.values() if success)
        if success_count == len(results):
            logger.info("✅ Export completed successfully!")
            return 0
        else:
            logger.warning(f"⚠️ Export partially successful ({success_count}/{len(results)})")
            return 1
    
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
