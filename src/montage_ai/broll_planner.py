"""
B-Roll Planner - Script-to-Clip Matching

DRY: Reuses video_agent.py for semantic search.
KISS: Simple script parsing, direct clip suggestions.

Usage (CLI):
    python -m montage_ai.broll_planner "The athlete trains hard. Victory celebration."

Usage (API):
    from montage_ai.broll_planner import plan_broll
    suggestions = plan_broll("The athlete trains hard.")
"""

import re
import os
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .logger import logger
from .config import get_settings

# Reuse video_agent for semantic search (DRY)
try:
    from .video_agent import create_video_agent, VideoAgentAdapter
    VIDEO_AGENT_AVAILABLE = True
except ImportError:
    VIDEO_AGENT_AVAILABLE = False


@dataclass
class BRollSuggestion:
    """Single B-roll suggestion for a script segment."""
    segment_text: str
    clip_path: str
    start_time: float
    end_time: float
    score: float
    caption: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment": self.segment_text,
            "clip": self.clip_path,
            "start": self.start_time,
            "end": self.end_time,
            "score": self.score,
            "caption": self.caption
        }


def split_script(script: str) -> List[str]:
    """Split script into searchable segments.

    KISS: Split on sentences. No complex NLP.
    """
    # Split on sentence endings
    segments = re.split(r'[.!?]+', script)
    # Clean and filter empty
    return [s.strip() for s in segments if s.strip() and len(s.strip()) > 3]


def plan_broll(
    script: str,
    input_dir: Optional[str] = None,
    top_k: int = 3,
    analyze_first: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate B-roll suggestions for a script.

    Args:
        script: The script/voiceover text
        input_dir: Directory with video files (default: /data/input)
        top_k: Number of suggestions per segment
        analyze_first: Run video analysis if memory is empty

    Returns:
        List of suggestions, one per script segment
    """
    if not VIDEO_AGENT_AVAILABLE:
        logger.error("Video Agent not available")
        return [{"error": "Video Agent not available"}]

    if input_dir is None:
        input_dir = str(get_settings().paths.input_dir)

    # Initialize agent
    agent = create_video_agent()

    # Check if we need to analyze footage first
    stats = agent.get_memory_stats()
    if analyze_first and stats.get("temporal_entries", 0) == 0:
        # Auto-analyze all videos in input_dir
        from pathlib import Path
        video_files = list(Path(input_dir).glob("*.mp4")) + list(Path(input_dir).glob("*.mov"))
        logger.info(f"Analyzing {len(video_files)} videos for B-roll planning...")
        for vf in video_files:
            agent.analyze_video(str(vf))

    # Split script into segments
    segments = split_script(script)
    logger.info(f"Split script into {len(segments)} segments")

    # Get suggestions for each segment
    results = []
    for segment in segments:
        matches = agent.caption_retrieval(segment, top_k=top_k)

        segment_result = {
            "segment": segment,
            "suggestions": []
        }

        for match in matches:
            segment_result["suggestions"].append({
                "clip": match.get("video_path", ""),
                "start": match.get("start_time", 0),
                "end": match.get("end_time", 0),
                "score": match.get("similarity_score", 0),
                "caption": match.get("caption", "")
            })

        results.append(segment_result)

    return results


def format_plan(results: List[Dict[str, Any]], verbose: bool = False) -> str:
    """Format B-roll plan for terminal output.

    KISS: Simple text formatting, no fancy rendering.
    """
    lines = ["B-ROLL PLAN", "=" * 40]

    for i, segment in enumerate(results, 1):
        lines.append(f"\n[{i}] \"{segment['segment']}\"")

        if segment.get("suggestions"):
            for j, sug in enumerate(segment["suggestions"], 1):
                clip_name = sug["clip"].split("/")[-1] if sug["clip"] else "unknown"
                score_pct = int(sug["score"] * 100)
                lines.append(f"    {j}. {clip_name} [{sug['start']:.1f}s-{sug['end']:.1f}s] ({score_pct}%)")
                if verbose and sug.get("caption"):
                    lines.append(f"       > {sug['caption']}")
        else:
            lines.append("    (no matches)")

    return "\n".join(lines)


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B-Roll Planner - Script-to-Clip Matching")
    parser.add_argument("script", nargs="+", help="The script/voiceover text")
    parser.add_argument("--input-dir", help="Directory with video files")
    parser.add_argument("--top-k", type=int, default=3, help="Number of suggestions per segment")
    parser.add_argument("--no-analyze", action="store_false", dest="analyze", help="Skip initial analysis")
    
    args = parser.parse_args()
    
    script_text = " ".join(args.script)
    logger.info(f"Planning B-roll for: \"{script_text[:50]}...\"")

    results = plan_broll(
        script_text, 
        input_dir=args.input_dir,
        top_k=args.top_k,
        analyze_first=args.analyze
    )
    
    print(format_plan(results, verbose=True))
