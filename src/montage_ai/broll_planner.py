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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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
        return [{"error": "Video Agent not available"}]

    input_dir = input_dir or os.environ.get("INPUT_DIR", "/data/input")

    # Initialize agent
    agent = create_video_agent()

    # Check if we need to analyze footage first
    stats = agent.get_memory_stats()
    if analyze_first and stats.get("temporal_entries", 0) == 0:
        # Auto-analyze all videos in input_dir
        from pathlib import Path
        video_files = list(Path(input_dir).glob("*.mp4")) + list(Path(input_dir).glob("*.mov"))
        for vf in video_files:
            agent.analyze_video(str(vf))

    # Split script into segments
    segments = split_script(script)

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
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m montage_ai.broll_planner \"Your script text here.\"")
        print("\nExample:")
        print("  python -m montage_ai.broll_planner \"The athlete trains. Victory moment.\"")
        sys.exit(1)

    script_text = " ".join(sys.argv[1:])
    print(f"> Planning B-roll for: \"{script_text[:50]}...\"")
    print()

    results = plan_broll(script_text)
    print(format_plan(results, verbose=True))
