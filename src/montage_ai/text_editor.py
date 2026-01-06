"""
Text-Based Video Editor - Edit video by editing transcript.

The core concept: Delete text → Delete corresponding video.

This enables Descript-style editing where users work with text
instead of timelines. Powered by Whisper word-level timestamps.

Usage:
    from montage_ai.text_editor import TextEditor

    editor = TextEditor("video.mp4")
    editor.load_transcript("transcript.json")  # Whisper JSON with word timestamps

    # Mark segments to remove
    editor.remove_filler_words()  # Auto-remove "um", "uh", etc.
    editor.remove_segment(start=10.5, end=15.2)  # Manual removal
    editor.remove_words(["um", "uh", "like"])  # Remove specific words

    # Export
    editor.export("output.mp4")  # Render edited video
    editor.export_edl("cuts.edl")  # Export cut list for NLE
"""

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

from .ffmpeg_config import get_config
from .ffmpeg_utils import build_ffmpeg_cmd
from .logger import logger
from .video_metadata import probe_metadata

# Import Timeline Exporter for Pro Handoff
try:
    from .timeline_exporter import TimelineExporter, Timeline, Clip
    TIMELINE_EXPORTER_AVAILABLE = True
except ImportError:
    TIMELINE_EXPORTER_AVAILABLE = False
    logger.warning("TimelineExporter not available")


@dataclass
class Word:
    """A single word with timing information."""
    text: str
    start: float
    end: float
    confidence: float = 1.0
    removed: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Segment:
    """A transcript segment (typically a sentence)."""
    id: int
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def active_words(self) -> List[Word]:
        """Words not marked for removal."""
        return [w for w in self.words if not w.removed]

    @property
    def active_text(self) -> str:
        """Text with removed words excluded."""
        return " ".join(w.text for w in self.active_words)


@dataclass
class EditRegion:
    """A region to keep or remove in the final video."""
    start: float
    end: float
    keep: bool = True
    reason: str = ""


# Common filler words to auto-detect
FILLER_WORDS = {
    # English
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "actually", "literally", "honestly", "right", "so", "well",
    # German
    "äh", "ähm", "halt", "quasi", "sozusagen", "eigentlich",
    # French
    "euh", "ben", "genre", "quoi",
}


class TextEditor:
    """
    Edit video by editing its transcript.

    Workflow:
    1. Load video and its Whisper transcript (with word timestamps)
    2. Mark words/segments for removal
    3. Generate cut list
    4. Render or export to NLE
    """

    def __init__(self, video_path: str):
        """
        Initialize text editor.

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.segments: List[Segment] = []
        self._video_duration: Optional[float] = None
        self._removed_words: Set[int] = set()  # Track removed word indices

    @property
    def video_duration(self) -> float:
        """Get video duration using cached metadata probe."""
        if self._video_duration is None:
            metadata = probe_metadata(str(self.video_path))
            self._video_duration = metadata.duration if metadata else 0.0
        return self._video_duration

    def load_transcript(self, transcript_path: str) -> int:
        """
        Load Whisper JSON transcript with word-level timestamps.

        Expected format:
        {
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 1.0, "probability": 0.99},
                        {"word": "world", "start": 1.2, "end": 2.5, "probability": 0.95}
                    ]
                }
            ]
        }

        Args:
            transcript_path: Path to Whisper JSON output

        Returns:
            Number of segments loaded
        """
        path = Path(transcript_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        self.segments = []

        for seg_data in data.get("segments", []):
            words = []
            for w in seg_data.get("words", []):
                word = Word(
                    text=w.get("word", "").strip(),
                    start=w.get("start", 0),
                    end=w.get("end", 0),
                    confidence=w.get("probability", w.get("confidence", 1.0))
                )
                if word.text:  # Skip empty words
                    words.append(word)

            segment = Segment(
                id=seg_data.get("id", len(self.segments)),
                start=seg_data["start"],
                end=seg_data["end"],
                text=seg_data.get("text", "").strip(),
                words=words
            )
            self.segments.append(segment)

        return len(self.segments)

    def load_srt(self, srt_path: str) -> int:
        """
        Load SRT file (without word-level timing).

        Note: Word-level editing won't be available, only segment-level.

        Args:
            srt_path: Path to SRT file

        Returns:
            Number of segments loaded
        """
        path = Path(srt_path)
        content = path.read_text(encoding="utf-8")
        self.segments = []

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

            # Create single word spanning entire segment (no word-level timing)
            word = Word(text=text, start=start, end=end)

            segment = Segment(
                id=int(match.group(1)),
                start=start,
                end=end,
                text=text,
                words=[word]
            )
            self.segments.append(segment)

        return len(self.segments)

    # =========================================================================
    # Editing Operations
    # =========================================================================

    def remove_segment(self, segment_id: int) -> None:
        """Remove an entire segment by ID."""
        for seg in self.segments:
            if seg.id == segment_id:
                for word in seg.words:
                    word.removed = True
                break

    def remove_segment_by_time(self, start: float, end: float) -> int:
        """
        Remove all content within a time range.

        Args:
            start: Start time in seconds
            end: End time in seconds

        Returns:
            Number of words removed
        """
        removed_count = 0
        for seg in self.segments:
            for word in seg.words:
                if word.start >= start and word.end <= end:
                    word.removed = True
                    removed_count += 1
        return removed_count

    def remove_words(self, words_to_remove: List[str]) -> int:
        """
        Remove specific words (case-insensitive).

        Args:
            words_to_remove: List of words to remove

        Returns:
            Number of words removed
        """
        target_words = {w.lower() for w in words_to_remove}
        removed_count = 0

        for seg in self.segments:
            for word in seg.words:
                if word.text.lower().strip(".,!?;:") in target_words:
                    word.removed = True
                    removed_count += 1

        return removed_count

    def remove_filler_words(self, custom_fillers: Optional[Set[str]] = None) -> int:
        """
        Remove common filler words ("um", "uh", etc.).

        Args:
            custom_fillers: Additional filler words to remove

        Returns:
            Number of filler words removed
        """
        fillers = FILLER_WORDS.copy()
        if custom_fillers:
            fillers.update(custom_fillers)

        return self.remove_words(list(fillers))

    def toggle_word(self, global_index: int, removed: bool) -> bool:
        """
        Set the removed state of a word by its global index.

        Args:
            global_index: 0-based index of the word in the entire transcript
            removed: True to remove (strike), False to keep

        Returns:
            True if word was found, False otherwise
        """
        current_idx = 0
        for seg in self.segments:
            for word in seg.words:
                if current_idx == global_index:
                    word.removed = removed
                    return True
                current_idx += 1
        return False

    def remove_low_confidence(self, threshold: float = 0.5) -> int:
        """
        Remove words with low transcription confidence.

        Args:
            threshold: Confidence threshold (0-1)

        Returns:
            Number of words removed
        """
        removed_count = 0
        for seg in self.segments:
            for word in seg.words:
                if word.confidence < threshold:
                    word.removed = True
                    removed_count += 1
        return removed_count

    def remove_silence(self, min_gap: float = 1.0) -> int:
        """
        Mark long silences for removal.

        Note: This marks the gaps, not words. The cut list generator
        will skip these silent regions.

        Args:
            min_gap: Minimum gap duration to consider as removable silence

        Returns:
            Number of silent gaps identified
        """
        # This is tracked in get_cut_list() rather than word removal
        # Just return count of gaps for feedback
        gaps = self._find_gaps(min_gap)
        return len(gaps)

    def restore_all(self) -> None:
        """Restore all removed content."""
        for seg in self.segments:
            for word in seg.words:
                word.removed = False

    # =========================================================================
    # Analysis
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get editing statistics."""
        total_words = 0
        removed_words = 0
        total_duration = 0.0
        removed_duration = 0.0
        removed_segments = sum(1 for seg in self.segments if all(w.removed for w in seg.words) if seg.words)

        for seg in self.segments:
            for word in seg.words:
                total_words += 1
                total_duration += word.duration
                if word.removed:
                    removed_words += 1
                    removed_duration += word.duration

        return {
            "total_segments": len(self.segments),
            "removed_segments": removed_segments,
            "total_words": total_words,
            "removed_words": removed_words,
            "kept_words": total_words - removed_words,
            "total_duration": total_duration,
            "removed_duration": removed_duration,
            "kept_duration": total_duration - removed_duration,
            "time_saved_seconds": removed_duration,
            "removal_percentage": (removed_words / total_words * 100) if total_words else 0
        }

    def get_transcript(self, include_removed: bool = False) -> str:
        """
        Get current transcript text.

        Args:
            include_removed: Include removed words (struck through)

        Returns:
            Transcript text
        """
        lines = []
        for seg in self.segments:
            if include_removed:
                words = []
                for w in seg.words:
                    if w.removed:
                        words.append(f"~~{w.text}~~")
                    else:
                        words.append(w.text)
                lines.append(" ".join(words))
            else:
                lines.append(seg.active_text)
        return "\n".join(lines)

    def _find_gaps(self, min_gap: float = 0.5) -> List[Tuple[float, float]]:
        """Find gaps between words/segments."""
        gaps = []
        all_words = []

        for seg in self.segments:
            for word in seg.words:
                if not word.removed:
                    all_words.append((word.start, word.end))

        all_words.sort(key=lambda x: x[0])

        for i in range(len(all_words) - 1):
            gap_start = all_words[i][1]
            gap_end = all_words[i + 1][0]
            if gap_end - gap_start >= min_gap:
                gaps.append((gap_start, gap_end))

        return gaps

    # =========================================================================
    # Cut List Generation
    # =========================================================================

    def get_cut_list(
        self,
        padding: float = 0.05,
        merge_threshold: float = 0.2,
        remove_silence: bool = True,
        silence_threshold: float = 1.0
    ) -> List[EditRegion]:
        """
        Generate cut list from current edit state.

        Args:
            padding: Time padding around kept words (seconds)
            merge_threshold: Merge adjacent regions closer than this
            remove_silence: Also remove long silent gaps
            silence_threshold: Minimum silence duration to remove

        Returns:
            List of EditRegions representing the final cut
        """
        # Collect all kept word regions
        kept_regions: List[Tuple[float, float]] = []

        for seg in self.segments:
            for word in seg.words:
                if not word.removed:
                    start = max(0, word.start - padding)
                    end = word.end + padding
                    kept_regions.append((start, end))

        if not kept_regions:
            return []

        # Sort by start time
        kept_regions.sort(key=lambda x: x[0])

        # Merge overlapping/adjacent regions
        merged: List[Tuple[float, float]] = [kept_regions[0]]
        for start, end in kept_regions[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + merge_threshold:
                # Extend previous region
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        # Optionally remove long silences within kept regions
        if remove_silence:
            final_regions = []
            for start, end in merged:
                # Check for internal silences
                gaps = self._find_gaps(silence_threshold)
                region_gaps = [(g[0], g[1]) for g in gaps if g[0] >= start and g[1] <= end]

                if not region_gaps:
                    final_regions.append((start, end))
                else:
                    # Split around gaps
                    current_start = start
                    for gap_start, gap_end in region_gaps:
                        if gap_start > current_start:
                            final_regions.append((current_start, gap_start))
                        current_start = gap_end
                    if current_start < end:
                        final_regions.append((current_start, end))
            merged = final_regions

        # Convert to EditRegions
        edit_regions = []
        for start, end in merged:
            edit_regions.append(EditRegion(
                start=start,
                end=min(end, self.video_duration),
                keep=True
            ))

        return edit_regions

    # =========================================================================
    # Export
    # =========================================================================

    def render_preview(self, output_path: str) -> str:
        """
        Render a fast preview of the current edit.
        
        Optimized for speed:
        - 360p resolution
        - Ultrafast preset
        - High CRF (lower quality)
        - Short audio crossfades (20ms)
        
        Args:
            output_path: Path to save preview video
            
        Returns:
            Path to output video
        """
        # Import preview constants to avoid circular imports or duplication
        from .ffmpeg_config import PREVIEW_PRESET, PREVIEW_CRF, PREVIEW_WIDTH, PREVIEW_HEIGHT
        
        return self.export(
            output_path,
            preset=PREVIEW_PRESET,
            crf=PREVIEW_CRF,
            audio_crossfade_ms=20,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT
        )

    def export(
        self,
        output_path: str,
        codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium",
        audio_crossfade_ms: int = 50,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> str:
        """
        Render edited video with smooth audio crossfades.

        Uses micro-crossfades (default 50ms) at cut points to avoid
        hard audio pops and make edits invisible.
        
        Optimized for performance:
        - Uses FFmpeg filter_complex for single-pass rendering
        - Uses hardware acceleration if available (via FFmpegConfig)
        - Uses input seeking (-ss) to avoid decoding unnecessary frames

        Args:
            output_path: Output file path
            codec: Video codec (default: libx264, but will use GPU if available)
            crf: Quality (lower = better)
            preset: Encoding preset
            audio_crossfade_ms: Audio crossfade duration in milliseconds
            width: Optional output width (for scaling)
            height: Optional output height (for scaling)

        Returns:
            Path to output video
        """
        cut_list = self.get_cut_list()
        if not cut_list:
            raise ValueError("No content to export (everything removed?)")

        logger.info(f"   {len(cut_list)} segments to keep")
        logger.info(f"   🔊 Using {audio_crossfade_ms}ms audio crossfades for smooth cuts")

        # Get hardware optimized config
        # If codec is default libx264, allow auto-detection of GPU
        use_hw_accel = (codec == "libx264")
        config = get_config("auto" if use_hw_accel else "none")
        
        if use_hw_accel and config.is_gpu_accelerated:
            logger.info(f"   🚀 Using GPU acceleration: {config.gpu_encoder_type}")
        
        # Prepare inputs and filters
        inputs = []
        filter_parts = []
        concat_v = []
        concat_a = []
        
        fade_sec = audio_crossfade_ms / 1000.0
        half_fade = fade_sec / 2.0
        
        for i, region in enumerate(cut_list):
            duration = region.end - region.start
            
            # Input args: Fast seek to start, limit duration
            # Note: We use string conversion for path to handle Path objects
            inputs.extend([
                "-ss", f"{region.start:.3f}",
                "-t", f"{duration:.3f}",
                "-i", str(self.video_path)
            ])
            
            # Video filter: Reset PTS
            # We don't fade video, just audio
            filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
            concat_v.append(f"[v{i}]")
            
            # Audio filter: Reset PTS + Crossfade
            # Fade in start, fade out end
            afade = f"afade=t=in:st=0:d={half_fade},afade=t=out:st={duration-half_fade}:d={half_fade}"
            filter_parts.append(f"[{i}:a]asetpts=PTS-STARTPTS,{afade}[a{i}]")
            concat_a.append(f"[a{i}]")

        # Concat filter
        n_segments = len(cut_list)
        concat_filter = f"{''.join(concat_v)}{''.join(concat_a)}concat=n={n_segments}:v=1:a=1[v_concat][outa]"
        filter_parts.append(concat_filter)
        
        # Scaling (if requested)
        if width and height:
            # Use scale filter with force_original_aspect_ratio to avoid distortion
            # and pad to fill the box if needed (though usually we just scale)
            # For preview, simple scaling is often enough, but let's be safe.
            # scale=w:h:force_original_aspect_ratio=decrease,pad=w:h:(ow-iw)/2:(oh-ih)/2
            scale_filter = f"[v_concat]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2[outv]"
            filter_parts.append(scale_filter)
        else:
            # Just rename the label
            filter_parts.append(f"[v_concat]null[outv]")
        
        full_filter = ";".join(filter_parts)
        output_video_label = "[outv]"
        if config.hwupload_filter:
            full_filter = f"{full_filter};[outv]{config.hwupload_filter}[outv_hw]"
            output_video_label = "[outv_hw]"
        
        # Build command
        cmd = build_ffmpeg_cmd([], hide_banner=True, loglevel="error")
        
        # Add hardware decoding args if applicable
        if config.is_gpu_accelerated:
            cmd.extend(config.hwaccel_input_params())
            
        # Add all inputs
        cmd.extend(inputs)
        
        # Add filter complex
        cmd.extend(["-filter_complex", full_filter])
        
        # Map output
        cmd.extend(["-map", output_video_label, "-map", "[outa]"])
        
        # Encoding parameters
        if use_hw_accel:
            # Use config's optimized params
            cmd.extend(config.video_params(crf=crf, preset=preset))
        else:
            # Use manual params
            cmd.extend(["-c:v", codec, "-crf", str(crf), "-preset", preset])
            
        # Audio encoding
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        
        # Output path
        cmd.append(output_path)

        logger.info(f"   Rendering...")
        # logger.debug(" ".join(cmd)) # Debug
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # Fallback for "Argument list too long" or other errors
            logger.warning(f"   ⚠️ Render failed: {e}")
            logger.warning("   Falling back to segment-based rendering...")
            return self._export_fallback(output_path, codec, crf, preset, audio_crossfade_ms)

        stats = self.get_stats()
        logger.info(f"   Removed {stats['removed_duration']:.1f}s ({stats['removal_percentage']:.1f}%)")
        logger.info(f"   Output: {output_path}")

        return output_path

    def _export_fallback(
        self,
        output_path: str,
        codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium",
        audio_crossfade_ms: int = 50
    ) -> str:
        """Fallback method using intermediate files (original implementation)."""
        cut_list = self.get_cut_list()
        
        # Extract each segment with audio fade handles
        segment_files = []
        fade_ms = audio_crossfade_ms / 2  # Half on each side

        for i, region in enumerate(cut_list):
            seg_file = tempfile.mktemp(suffix=f"_seg{i}.mp4")
            segment_files.append(seg_file)

            duration = region.end - region.start
            fade_in = fade_ms / 1000  # Convert to seconds
            fade_out = fade_ms / 1000

            # Build audio filter for micro-crossfades
            # afade in at start, afade out at end
            af_filter = f"afade=t=in:st=0:d={fade_in},afade=t=out:st={duration - fade_out}:d={fade_out}"

            # Extract segment with audio fades
            cmd = build_ffmpeg_cmd(
                [
                    "-hide_banner", "-loglevel", "error",
                    "-ss", str(region.start),
                    "-i", str(self.video_path),
                    "-t", str(duration),
                    "-c:v", "copy",  # Copy video (no re-encode)
                    "-af", af_filter,  # Apply audio crossfades
                    "-c:a", "aac",
                    "-b:a", "192k",
                    seg_file
                ]
            )
            subprocess.run(cmd, check=True)

        # Create concat file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_file = f.name
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")

        # Concatenate all segments
        cmd = build_ffmpeg_cmd(
            [
                "-hide_banner", "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c:v", codec,
                "-crf", str(crf),
                "-preset", preset,
                "-c:a", "copy",  # Audio already faded
                output_path
            ]
        )

        logger.info(f"   Rendering (fallback)...")
        subprocess.run(cmd, check=True)

        # Cleanup temp files
        Path(concat_file).unlink()
        for seg_file in segment_files:
            Path(seg_file).unlink()

        stats = self.get_stats()
        logger.info(f"   Removed {stats['removed_duration']:.1f}s ({stats['removal_percentage']:.1f}%)")
        logger.info(f"   Output: {output_path}")

        return output_path


    def export_edl(self, output_path: str) -> str:
        """
        Export Edit Decision List for NLE import.
        Uses TimelineExporter for robust CMX 3600 generation.

        Args:
            output_path: Output EDL file path

        Returns:
            Path to EDL file
        """
        if not TIMELINE_EXPORTER_AVAILABLE:
            logger.warning("TimelineExporter not available, falling back to legacy EDL export")
            return self._export_edl_legacy(output_path)

        # Convert to Timeline object
        timeline = self._create_timeline(project_name="text_edit_export")
        
        # Export
        exporter = TimelineExporter(output_dir=os.path.dirname(output_path))
        result = exporter.export_timeline(
            timeline,
            generate_proxies=False,
            export_otio=False,
            export_edl=True,
            export_xml=False,
            export_csv=False
        )
        
        # Move/Rename if necessary (TimelineExporter generates its own name)
        generated_edl = result.get('edl')
        if generated_edl and generated_edl != output_path:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(generated_edl, output_path)
            return output_path
        
        return generated_edl or output_path

    def export_otio(self, output_path: str) -> str:
        """
        Export OpenTimelineIO file for DaVinci/Premiere.

        Args:
            output_path: Output OTIO file path

        Returns:
            Path to OTIO file
        """
        if not TIMELINE_EXPORTER_AVAILABLE:
            raise RuntimeError("TimelineExporter not available")

        timeline = self._create_timeline(project_name="text_edit_export")
        exporter = TimelineExporter(output_dir=os.path.dirname(output_path))
        
        result = exporter.export_timeline(
            timeline,
            generate_proxies=False,
            export_otio=True,
            export_edl=False,
            export_xml=False,
            export_csv=False
        )
        
        generated_otio = result.get('otio')
        if generated_otio and generated_otio != output_path:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(generated_otio, output_path)
            return output_path
            
        return generated_otio or output_path

    def _create_timeline(self, project_name: str) -> 'Timeline':
        """Convert current edit state to a Timeline object."""
        cut_list = self.get_cut_list()
        clips = []
        
        current_timeline_time = 0.0
        
        for region in cut_list:
            duration = region.end - region.start
            
            clips.append(Clip(
                source_path=str(self.video_path),
                start_time=region.start,
                duration=duration,
                timeline_start=current_timeline_time,
                metadata={
                    "text_content": " ".join(w.text for w in self.get_words_in_range(region.start, region.end))
                }
            ))
            current_timeline_time += duration
            
        return Timeline(
            clips=clips,
            audio_path=str(self.video_path), # Use video as audio source
            total_duration=current_timeline_time,
            project_name=project_name
        )

    def _export_edl_legacy(self, output_path: str) -> str:
        """Legacy EDL export (fallback)."""
        cut_list = self.get_cut_list()
        if not cut_list:
            raise ValueError("No content to export")

        def tc(seconds: float, fps: float = 30.0) -> str:
            """Convert seconds to timecode HH:MM:SS:FF"""
            total_frames = int(seconds * fps)
            frames = total_frames % int(fps)
            total_seconds = total_frames // int(fps)
            secs = total_seconds % 60
            mins = (total_seconds // 60) % 60
            hours = total_seconds // 3600
            return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"

        lines = [
            "TITLE: Text-Based Edit",
            f"FCM: NON-DROP FRAME",
            ""
        ]

        record_in = 0.0
        for i, region in enumerate(cut_list):
            edit_num = i + 1
            source_in = tc(region.start)
            source_out = tc(region.end)
            rec_in = tc(record_in)
            rec_out = tc(record_in + (region.end - region.start))

            lines.append(
                f"{edit_num:03d}  001      V     C        "
                f"{source_in} {source_out} {rec_in} {rec_out}"
            )
            record_in += region.end - region.start

        Path(output_path).write_text("\n".join(lines))
        logger.info(f"   EDL exported: {output_path}")
        return output_path

    def export_json(self, output_path: str) -> str:
        """
        Export cut list as JSON for custom processing.

        Args:
            output_path: Output JSON file path

        Returns:
            Path to JSON file
        """
        cut_list = self.get_cut_list()
        data = {
            "source": str(self.video_path),
            "source_duration": self.video_duration,
            "stats": self.get_stats(),
            "regions": [
                {
                    "start": r.start,
                    "end": r.end,
                    "duration": r.end - r.start,
                    "keep": r.keep
                }
                for r in cut_list
            ]
        }

        Path(output_path).write_text(json.dumps(data, indent=2))
        return output_path


# =============================================================================
# Interactive Mode (CLI with $EDITOR)
# =============================================================================

def edit_transcript_interactive(
    video_path: str,
    transcript_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Open transcript in $EDITOR for manual editing.

    User deletes lines/words, saves, and video is rendered.

    Args:
        video_path: Input video
        transcript_path: Whisper JSON transcript
        output_path: Output video (optional)

    Returns:
        Path to edited video
    """
    import os

    editor = TextEditor(video_path)
    editor.load_transcript(transcript_path)

    # Save transcript to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        temp_transcript = f.name
        f.write("# Delete lines or words you want to remove from the video\n")
        f.write("# Lines starting with # are ignored\n")
        f.write("#\n\n")

        for seg in editor.segments:
            f.write(f"[{seg.start:.2f} - {seg.end:.2f}]\n")
            f.write(f"{seg.text}\n\n")

    # Open in editor
    editor_cmd = os.environ.get("EDITOR", "nano")
    subprocess.run([editor_cmd, temp_transcript])

    # Parse edited transcript (simplified - just check which segments were deleted)
    edited_content = Path(temp_transcript).read_text()
    Path(temp_transcript).unlink()

    # Find which segments are still present
    kept_times = set()
    for line in edited_content.split("\n"):
        match = re.match(r"\[(\d+\.?\d*)\s*-", line)
        if match:
            kept_times.add(float(match.group(1)))

    # Remove segments not in kept_times
    for seg in editor.segments:
        if seg.start not in kept_times:
            editor.remove_segment(seg.id)

    # Export
    if output_path is None:
        output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_edited.mp4")

    return editor.export(output_path)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys

    def print_usage():
        print("Text-Based Video Editor")
        print()
        print("Usage:")
        print("  python -m montage_ai.text_editor <video> <transcript.json> [output]")
        print()
        print("Options:")
        print("  --remove-fillers    Auto-remove filler words (um, uh, etc.)")
        print("  --interactive       Open transcript in $EDITOR")
        print("  --export-edl FILE   Export EDL cut list")
        print("  --stats             Show editing statistics only")
        print()
        print("Examples:")
        print("  python -m montage_ai.text_editor video.mp4 whisper.json")
        print("  python -m montage_ai.text_editor video.mp4 whisper.json --remove-fillers")
        print("  python -m montage_ai.text_editor video.mp4 whisper.json --interactive")

    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print_usage()
        sys.exit(0)

    video = sys.argv[1]
    transcript = None
    args = []

    # Check if second arg is transcript (file) or option
    if len(sys.argv) > 2:
        if not sys.argv[2].startswith("--") and (sys.argv[2].endswith(".json") or sys.argv[2].endswith(".srt")):
            transcript = sys.argv[2]
            args = sys.argv[3:]
        else:
            args = sys.argv[2:]

    # Parse options
    remove_fillers = "--remove-fillers" in args
    interactive = "--interactive" in args
    stats_only = "--stats" in args
    edl_export = None
    otio_export = None

    for i, arg in enumerate(args):
        if arg == "--export-edl" and i + 1 < len(args):
            edl_export = args[i + 1]
        elif arg == "--export-otio" and i + 1 < len(args):
            otio_export = args[i + 1]

    # Find output path (first non-flag argument)
    output = None
    for arg in args:
        if not arg.startswith("--") and arg != edl_export and arg != otio_export:
            output = arg
            break

    try:
        # Auto-generate transcript if missing
        if transcript is None:
            print("   No transcript provided. Attempting to generate via cgpu...")
            try:
                from .transcriber import transcribe_audio
                # Use json format for word-level timestamps (needed for editing)
                transcript = transcribe_audio(video, output_format="json")
                if not transcript:
                    print("   ❌ Failed to generate transcript (is cgpu running?).")
                    sys.exit(1)
                print(f"   ✅ Generated transcript: {transcript}")
            except ImportError:
                print("   ❌ Transcriber module not found.")
                sys.exit(1)
            except Exception as e:
                print(f"   ❌ Transcription error: {e}")
                sys.exit(1)

        if interactive:
            result = edit_transcript_interactive(video, transcript, output)
            print(f"\n Edited video: {result}")
        else:
            editor = TextEditor(video)
            loaded = editor.load_transcript(transcript)
            print(f"   Loaded {loaded} segments")

            if remove_fillers:
                removed = editor.remove_filler_words()
                print(f"   Removed {removed} filler words")

            if stats_only:
                stats = editor.get_stats()
                print(f"\n   Statistics:")
                print(f"   - Total words: {stats['total_words']}")
                print(f"   - Removed: {stats['removed_words']}")
                print(f"   - Duration removed: {stats['removed_duration']:.1f}s")
                sys.exit(0)

            if edl_export:
                editor.export_edl(edl_export)
                print(f"   ✅ Exported EDL: {edl_export}")

            if otio_export:
                try:
                    editor.export_otio(otio_export)
                    print(f"   ✅ Exported OTIO: {otio_export}")
                except Exception as e:
                    print(f"   ❌ OTIO Export failed: {e}")

            if output or (not edl_export and not otio_export):
                if output is None:
                    output = str(Path(video).parent / f"{Path(video).stem}_edited.mp4")
                result = editor.export(output)
                print(f"\n Edited video: {result}")

    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)
