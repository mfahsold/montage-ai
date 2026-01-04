"""
Caption Burner - Burn subtitles into video (hardcoded captions).

Supports multiple caption styles optimized for social media:
- TikTok: Large, centered, word-by-word highlighting
- YouTube: Classic bottom subtitles
- Minimal: Clean, small text
- Karaoke: Word-by-word color change

Usage:
    from montage_ai.caption_burner import CaptionBurner, CaptionStyle

    burner = CaptionBurner(style=CaptionStyle.TIKTOK)
    output = burner.burn("video.mp4", "subtitles.srt")
"""

import json
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

from .ffmpeg_utils import build_ffmpeg_cmd


class CaptionStyle(Enum):
    """Predefined caption styles for different platforms."""
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    MINIMAL = "minimal"
    KARAOKE = "karaoke"
    BOLD = "bold"
    CINEMATIC = "cinematic"


@dataclass
class CaptionSegment:
    """A single caption segment with timing."""
    start: float  # seconds
    end: float    # seconds
    text: str
    words: Optional[List[Dict[str, Any]]] = None  # word-level timing


@dataclass
class StyleConfig:
    """Configuration for caption rendering."""
    fontsize: int = 48
    fontcolor: str = "white"
    fontfile: Optional[str] = None  # Use default if None
    borderw: int = 2
    bordercolor: str = "black"
    shadowcolor: str = "black@0.5"
    shadowx: int = 2
    shadowy: int = 2
    box: bool = False
    boxcolor: str = "black@0.6"
    boxborderw: int = 10
    x_expr: str = "(w-text_w)/2"  # centered
    y_expr: str = "h-100"         # near bottom
    line_spacing: int = 10


# Predefined style configurations
STYLE_CONFIGS: Dict[CaptionStyle, StyleConfig] = {
    CaptionStyle.TIKTOK: StyleConfig(
        fontsize=64,
        fontcolor="white",
        borderw=4,
        bordercolor="black",
        shadowx=0,
        shadowy=0,
        box=False,
        x_expr="(w-text_w)/2",
        y_expr="(h-text_h)/2",  # centered vertically
    ),
    CaptionStyle.YOUTUBE: StyleConfig(
        fontsize=42,
        fontcolor="white",
        borderw=2,
        bordercolor="black",
        shadowx=2,
        shadowy=2,
        box=True,
        boxcolor="black@0.7",
        boxborderw=8,
        x_expr="(w-text_w)/2",
        y_expr="h-80",
    ),
    CaptionStyle.MINIMAL: StyleConfig(
        fontsize=36,
        fontcolor="white",
        borderw=1,
        bordercolor="black@0.5",
        shadowx=1,
        shadowy=1,
        box=False,
        x_expr="(w-text_w)/2",
        y_expr="h-60",
    ),
    CaptionStyle.KARAOKE: StyleConfig(
        fontsize=56,
        fontcolor="yellow",
        borderw=3,
        bordercolor="black",
        shadowx=0,
        shadowy=0,
        box=False,
        x_expr="(w-text_w)/2",
        y_expr="(h-text_h)/2",
    ),
    CaptionStyle.BOLD: StyleConfig(
        fontsize=72,
        fontcolor="white",
        borderw=5,
        bordercolor="black",
        shadowx=3,
        shadowy=3,
        box=False,
        x_expr="(w-text_w)/2",
        y_expr="(h-text_h)/2",
    ),
    CaptionStyle.CINEMATIC: StyleConfig(
        fontsize=38,
        fontcolor="white@0.9",
        borderw=0,
        box=True,
        boxcolor="black@0.4",
        boxborderw=12,
        shadowx=0,
        shadowy=0,
        x_expr="(w-text_w)/2",
        y_expr="h*0.85",
    ),
}


def parse_srt(srt_path: Path) -> List[CaptionSegment]:
    """Parse SRT subtitle file into segments."""
    segments = []
    content = srt_path.read_text(encoding="utf-8")

    # SRT format: index, timestamp, text, blank line
    pattern = re.compile(
        r"(\d+)\s*\n"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*\n"
        r"(.*?)(?=\n\n|\n*$)",
        re.DOTALL
    )

    for match in pattern.finditer(content):
        start = (
            int(match.group(2)) * 3600 +
            int(match.group(3)) * 60 +
            int(match.group(4)) +
            int(match.group(5)) / 1000
        )
        end = (
            int(match.group(6)) * 3600 +
            int(match.group(7)) * 60 +
            int(match.group(8)) +
            int(match.group(9)) / 1000
        )
        text = match.group(10).strip().replace("\n", " ")

        segments.append(CaptionSegment(start=start, end=end, text=text))

    return segments


def parse_vtt(vtt_path: Path) -> List[CaptionSegment]:
    """Parse WebVTT subtitle file into segments."""
    segments = []
    content = vtt_path.read_text(encoding="utf-8")

    # Skip WEBVTT header
    lines = content.split("\n")
    i = 0
    while i < len(lines) and not "-->" in lines[i]:
        i += 1

    # Parse cues
    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
    )

    while i < len(lines):
        line = lines[i].strip()
        match = pattern.match(line)
        if match:
            start = (
                int(match.group(1)) * 3600 +
                int(match.group(2)) * 60 +
                int(match.group(3)) +
                int(match.group(4)) / 1000
            )
            end = (
                int(match.group(5)) * 3600 +
                int(match.group(6)) * 60 +
                int(match.group(7)) +
                int(match.group(8)) / 1000
            )

            # Collect text lines until blank line
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            text = " ".join(text_lines)
            segments.append(CaptionSegment(start=start, end=end, text=text))
        i += 1

    return segments


def parse_whisper_json(json_path: Path) -> List[CaptionSegment]:
    """
    Parse Whisper JSON output with word-level timestamps.

    Whisper JSON format:
    {
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.0},
                    {"word": "world", "start": 1.2, "end": 2.5}
                ]
            }
        ]
    }
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = []

    for seg in data.get("segments", []):
        segment = CaptionSegment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip(),
            words=seg.get("words")
        )
        segments.append(segment)

    return segments


def load_captions(caption_path: str) -> List[CaptionSegment]:
    """Load captions from SRT, VTT, or Whisper JSON."""
    path = Path(caption_path)
    suffix = path.suffix.lower()

    if suffix == ".srt":
        return parse_srt(path)
    elif suffix == ".vtt":
        return parse_vtt(path)
    elif suffix == ".json":
        return parse_whisper_json(path)
    else:
        raise ValueError(f"Unsupported caption format: {suffix}")


def escape_ffmpeg_text(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    # FFmpeg drawtext requires escaping these characters
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("%", "\\%")
    return text


class CaptionBurner:
    """
    Burns captions into video using FFmpeg drawtext filter.

    Supports multiple styles optimized for different platforms.
    """

    def __init__(
        self,
        style: CaptionStyle = CaptionStyle.YOUTUBE,
        custom_config: Optional[StyleConfig] = None
    ):
        """
        Initialize caption burner.

        Args:
            style: Predefined style preset
            custom_config: Override style config (optional)
        """
        self.style = style
        self.config = custom_config or STYLE_CONFIGS[style]

    def _build_drawtext_filter(
        self,
        segment: CaptionSegment,
        config: StyleConfig
    ) -> str:
        """Build FFmpeg drawtext filter string for a segment."""
        text = escape_ffmpeg_text(segment.text)

        parts = [
            f"drawtext=text='{text}'",
            f"fontsize={config.fontsize}",
            f"fontcolor={config.fontcolor}",
            f"borderw={config.borderw}",
            f"bordercolor={config.bordercolor}",
            f"x={config.x_expr}",
            f"y={config.y_expr}",
            f"enable='between(t,{segment.start:.3f},{segment.end:.3f})'"
        ]

        if config.fontfile:
            parts.append(f"fontfile={config.fontfile}")

        if config.shadowx or config.shadowy:
            parts.append(f"shadowcolor={config.shadowcolor}")
            parts.append(f"shadowx={config.shadowx}")
            parts.append(f"shadowy={config.shadowy}")

        if config.box:
            parts.append("box=1")
            parts.append(f"boxcolor={config.boxcolor}")
            parts.append(f"boxborderw={config.boxborderw}")

        return ":".join(parts)

    def _build_karaoke_filters(
        self,
        segment: CaptionSegment,
        config: StyleConfig
    ) -> List[str]:
        """
        Build Karaoke-style word-by-word highlighting filters.
        
        Returns list of drawtext filters - one base layer + one highlight layer per word.
        """
        if not segment.words or len(segment.words) == 0:
            # Fallback to regular caption if no word timing
            return [self._build_drawtext_filter(segment, config)]
        
        filters = []
        
        # Base layer: Full sentence in default color (gray)
        base_text = escape_ffmpeg_text(segment.text)
        base_parts = [
            f"drawtext=text='{base_text}'",
            f"fontsize={config.fontsize}",
            f"fontcolor='gray'",
            f"borderw={config.borderw}",
            f"bordercolor={config.bordercolor}",
            f"x={config.x_expr}",
            f"y={config.y_expr}",
            f"enable='between(t,{segment.start:.3f},{segment.end:.3f})'"
        ]
        filters.append(":".join(base_parts))
        
        # Highlight layer: Each word in highlight color during its timing
        for i, word_data in enumerate(segment.words):
            word_text = escape_ffmpeg_text(word_data.get('word', '').strip())
            if not word_text:
                continue
                
            word_start = word_data.get('start', segment.start)
            word_end = word_data.get('end', segment.end)
            
            # Calculate word position offset (approximate)
            # This is a simple approach - word appears at same X as full text
            # For perfect positioning, would need to measure text width per word
            
            word_parts = [
                f"drawtext=text='{word_text}'",
                f"fontsize={config.fontsize}",
                f"fontcolor={config.fontcolor}",  # Highlight color (yellow)
                f"borderw={config.borderw}",
                f"bordercolor={config.bordercolor}",
                f"x={config.x_expr}",
                f"y={config.y_expr}",
                f"enable='between(t,{word_start:.3f},{word_end:.3f})'"
            ]
            
            filters.append(":".join(word_parts))
        
        return filters

    def _build_filter_chain(self, segments: List[CaptionSegment]) -> str:
        """Build complete FFmpeg filter chain for all segments."""
        filters = []
        
        # Check if Karaoke style (word-by-word)
        if self.style == CaptionStyle.KARAOKE:
            for segment in segments:
                karaoke_filters = self._build_karaoke_filters(segment, self.config)
                filters.extend(karaoke_filters)
        else:
            # Standard styles: one filter per segment
            for segment in segments:
                filter_str = self._build_drawtext_filter(segment, self.config)
                filters.append(filter_str)

        return ",".join(filters)

    def burn(
        self,
        video_path: str,
        caption_path: str,
        output_path: Optional[str] = None,
        codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium"
    ) -> str:
        """
        Burn captions into video.

        Args:
            video_path: Input video file
            caption_path: Caption file (SRT, VTT, or Whisper JSON)
            output_path: Output file (default: video_captioned.mp4)
            codec: Video codec (libx264, libx265, etc.)
            crf: Quality (lower = better, 18-28 typical)
            preset: Encoding speed preset

        Returns:
            Path to output video
        """
        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        captions = Path(caption_path)
        if not captions.exists():
            raise FileNotFoundError(f"Captions not found: {caption_path}")

        if output_path is None:
            output_path = str(video.parent / f"{video.stem}_captioned.mp4")

        # Load and parse captions
        segments = load_captions(caption_path)
        if not segments:
            raise ValueError("No caption segments found")

        print(f"   Loaded {len(segments)} caption segments")

        # Build filter chain
        filter_chain = self._build_filter_chain(segments)

        # Run FFmpeg
        cmd = build_ffmpeg_cmd(
            [
                "-hide_banner", "-loglevel", "error",
                "-i", str(video),
                "-vf", filter_chain,
                "-c:v", codec,
                "-crf", str(crf),
                "-preset", preset,
                "-c:a", "copy",
                output_path
            ]
        )

        print(f"   Burning captions ({self.style.value} style)...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        print(f"   Output: {output_path}")
        return output_path

    def burn_with_srt_filter(
        self,
        video_path: str,
        srt_path: str,
        output_path: Optional[str] = None,
        force_style: Optional[str] = None
    ) -> str:
        """
        Alternative method using FFmpeg's subtitles filter.

        This uses ASS styling which can be more flexible for some use cases.
        Requires the subtitles filter and libass.

        Args:
            video_path: Input video
            srt_path: SRT subtitle file
            output_path: Output file
            force_style: ASS style string (optional)

        Returns:
            Path to output video
        """
        video = Path(video_path)
        if output_path is None:
            output_path = str(video.parent / f"{video.stem}_captioned.mp4")

        # Build subtitles filter with styling
        style = self._get_ass_style() if force_style is None else force_style
        subtitle_filter = f"subtitles={srt_path}:force_style='{style}'"

        cmd = build_ffmpeg_cmd(
            [
                "-hide_banner", "-loglevel", "error",
                "-i", str(video),
                "-vf", subtitle_filter,
                "-c:v", "libx264",
                "-crf", "23",
                "-c:a", "copy",
                output_path
            ]
        )

        print(f"   Burning captions (subtitles filter)...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        return output_path

    def _get_ass_style(self) -> str:
        """Convert StyleConfig to ASS style string."""
        cfg = self.config

        # Map style to ASS format
        parts = [
            f"FontSize={cfg.fontsize}",
            f"PrimaryColour=&H00FFFFFF",  # White (ABGR format)
            f"OutlineColour=&H00000000",  # Black outline
            f"Outline={cfg.borderw}",
            f"Shadow={cfg.shadowx}",
            "Alignment=2",  # Bottom center
        ]

        if self.style == CaptionStyle.TIKTOK:
            parts.append("Alignment=5")  # Middle center

        return ",".join(parts)


# =============================================================================
# Convenience Functions
# =============================================================================

def burn_captions(
    video_path: str,
    caption_path: str,
    style: str = "youtube",
    output_path: Optional[str] = None
) -> str:
    """
    Quick caption burning helper.

    Args:
        video_path: Input video file
        caption_path: Caption file (SRT, VTT, or JSON)
        style: Caption style (tiktok, youtube, minimal, karaoke, bold, cinematic)
        output_path: Output file (optional)

    Returns:
        Path to output video with burned captions
    """
    try:
        caption_style = CaptionStyle(style.lower())
    except ValueError:
        print(f"   Unknown style '{style}', using youtube")
        caption_style = CaptionStyle.YOUTUBE

    burner = CaptionBurner(style=caption_style)
    return burner.burn(video_path, caption_path, output_path)


def list_caption_styles() -> List[Dict[str, str]]:
    """List available caption styles with descriptions."""
    return [
        {"name": "tiktok", "description": "Large centered text, social media optimized"},
        {"name": "youtube", "description": "Classic bottom subtitles with background"},
        {"name": "minimal", "description": "Clean, small, unobtrusive text"},
        {"name": "karaoke", "description": "Yellow centered text for karaoke style"},
        {"name": "bold", "description": "Extra large, high contrast text"},
        {"name": "cinematic", "description": "Subtle, semi-transparent box style"},
    ]


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys

    def print_usage():
        print("Caption Burner - Burn subtitles into video")
        print()
        print("Usage:")
        print("  python -m montage_ai.caption_burner <video> <captions> [style] [output]")
        print()
        print("Styles:")
        for s in list_caption_styles():
            print(f"  {s['name']:12} - {s['description']}")
        print()
        print("Examples:")
        print("  python -m montage_ai.caption_burner video.mp4 subs.srt")
        print("  python -m montage_ai.caption_burner video.mp4 subs.srt tiktok")
        print("  python -m montage_ai.caption_burner video.mp4 whisper.json bold output.mp4")

    if len(sys.argv) < 3 or sys.argv[1] in ["-h", "--help"]:
        print_usage()
        sys.exit(0)

    video = sys.argv[1]
    captions = sys.argv[2]
    style = sys.argv[3] if len(sys.argv) > 3 else "youtube"
    output = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        result = burn_captions(video, captions, style, output)
        print(f"\n Captions burned: {result}")
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)
