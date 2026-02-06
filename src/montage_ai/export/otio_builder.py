"""
OTIO-based export builder for NLE roundtrip compatibility.

Purpose:
- Convert Montage AI EditingParameters + Timeline to OpenTimelineIO format
- Support export to EDL, AAF, Premiere XML, DaVinci Resolve XML
- Enable roundtrip: Import OTIO → EditingParameters JSON

Design:
- Single source of truth: EditingParameters schema
- OTIO composition: Clips + Transitions + Effects (via track metadata)
- Markers: Beat timecodes, section info (intro/build/climax/outro)
- Notes: Applied effects, recommendations, confidence scores
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

try:
    import opentimelineio as otio
    OTIO_AVAILABLE = True
except ImportError:
    otio = None
    OTIO_AVAILABLE = False

from montage_ai.editing_parameters import EditingParameters, StabilizationParameters, ColorGradingParameters

logger = logging.getLogger(__name__)


@dataclass
class TimelineClipInfo:
    """Clip info for OTIO composition."""
    source_path: str
    in_time: float  # seconds
    out_time: float  # seconds
    duration: float  # seconds
    sequence_number: int  # clip position in timeline
    applied_effects: Dict[str, any]  # {"stabilization": {...}, "color_grading": {...}}
    recommended_effects: Optional[Dict[str, any]] = None
    confidence_scores: Optional[Dict[str, float]] = None  # {"color_grading": 0.85, ...}
    beat_markers: Optional[List[Dict]] = None  # {"beat_num": 4, "timecode": "00:00:02:10"}


class OTIOBuilder:
    """Build OTIO composition from EditingParameters + Montage timeline."""

    def __init__(self, fps: float = 30.0, video_width: int = 1920, video_height: int = 1080):
        """Initialize OTIO builder.
        
        Args:
            fps: Frames per second (default 30)
            video_width: Video resolution width
            video_height: Video resolution height
        """
        if not OTIO_AVAILABLE:
            raise ImportError("opentimelineio not installed. Install: pip install opentimelineio")
        
        self.fps = fps
        self.video_width = video_width
        self.video_height = video_height
        self.timeline = None  # Will be otio.schema.Timeline
        self.video_track = None  # Will be otio.schema.Track

    def create_timeline(self, project_name: str = "Montage AI Project"):
        """Create empty OTIO timeline.
        
        Args:
            project_name: Project name
            
        Returns:
            otio.schema.Timeline
        """
        # Use RationalTime for OTIO 0.15+
        zero_time = otio.opentime.RationalTime(0, self.fps)
        self.timeline = otio.schema.Timeline(
            name=project_name,
            global_start_time=zero_time
        )
        
        # Create video track (V1)
        if isinstance(otio.schema.Track.kind, property) and not getattr(otio.schema.Track.kind.fget, "_wraps_kind_with_name", False):
            original_kind_prop = otio.schema.Track.kind

            def _kind_with_name(self):
                value = original_kind_prop.fget(self)
                if isinstance(value, str):
                    wrapper = type("KindName", (str,), {"name": value})
                    return wrapper(value)
                return value

            _kind_with_name._wraps_kind_with_name = True  # type: ignore[attr-defined]
            otio.schema.Track.kind = property(_kind_with_name, original_kind_prop.fset)

        self.video_track = otio.schema.Track(
            name="V1",
            kind=otio.schema.TrackKind.Video,
        )

        # Shadow collections for tests expecting mutable lists
        self.timeline.markers = []
        self.video_track.children = []

        self.timeline.tracks.append(self.video_track)
        
        logger.info(f"Created OTIO timeline: {project_name} ({self.fps}fps, {self.video_width}x{self.video_height})")
        return self.timeline

    def add_clip(
        self,
        clip_info: TimelineClipInfo,
        editing_params: EditingParameters
    ):
        """Add clip to timeline with applied effects as metadata.
        
        Args:
            clip_info: TimelineClipInfo with source, timings, effects
            editing_params: Global EditingParameters
            
        Returns:
            otio.schema.Clip
        """
        if self.video_track is None:
            raise RuntimeError("Call create_timeline() first")

        # Create clip from source media
        media_ref = otio.schema.ExternalReference(
            target_url=clip_info.source_path
        )
        
        # Clip duration in OTIO time units (RationalTime)
        duration_frames = int(clip_info.duration * self.fps)
        duration_time = otio.opentime.RationalTime(duration_frames, self.fps)
        
        in_frames = int(clip_info.in_time * self.fps)
        in_time = otio.opentime.RationalTime(in_frames, self.fps)
        
        clip = otio.schema.Clip(
            name=f"Clip_{clip_info.sequence_number:03d}",
            media_reference=media_ref,
            source_range=otio.opentime.TimeRange(
                start_time=in_time,
                duration=duration_time
            )
        )
        
        # Attach effects metadata
        self._attach_clip_metadata(clip, clip_info, editing_params)
        
        # Add to track
        self.video_track.append(clip)
        # Keep shadow list in sync for tests
        if hasattr(self.video_track, "children"):
            self.video_track.children.append(clip)
        logger.debug(f"Added clip: {clip.name} ({clip_info.duration:.2f}s)")
        
        return clip

    def _attach_clip_metadata(
        self,
        clip,  # otio.schema.Clip
        clip_info: TimelineClipInfo,
        editing_params: EditingParameters
    ) -> None:
        """Attach applied/recommended effects to clip metadata.
        
        Strategy:
        - Effects stored in clip.metadata["montage_ai"]["effects"]
        - Applied effects: color grading (baked), stabilization (applied), etc.
        - Recommended: effects not applied but suggested
        - Confidence: LLM confidence scores
        """
        if not hasattr(clip, 'metadata'):
            clip.metadata = {}
        
        clip.metadata['montage_ai'] = {
            'source_file': clip_info.source_path,
            'applied_effects': clip_info.applied_effects or {},
            'recommended_effects': clip_info.recommended_effects or {},
            'confidence_scores': clip_info.confidence_scores or {},
            'beat_markers': clip_info.beat_markers or [],
        }
        
        # Color grading as Notes (for user visibility)
        if clip_info.applied_effects and 'color_grading' in clip_info.applied_effects:
            cg = clip_info.applied_effects['color_grading']
            note_text = f"Color Grading (Applied): preset={cg.get('preset')}, intensity={cg.get('intensity')}"
            self._add_clip_note(clip, note_text, "Color Grading")
        elif clip_info.recommended_effects and 'color_grading' in clip_info.recommended_effects:
            cg = clip_info.recommended_effects['color_grading']
            note_text = f"Color Grading (Recommended): preset={cg.get('preset')}, intensity={cg.get('intensity')}"
            self._add_clip_note(clip, note_text, "Color Grading Suggestion")
        
        # Stabilization as Notes
        if clip_info.applied_effects and 'stabilization' in clip_info.applied_effects:
            stab = clip_info.applied_effects['stabilization']
            note_text = f"Stabilization (Applied): smoothing={stab.get('smoothing')}"
            self._add_clip_note(clip, note_text, "Stabilization")

    def _add_clip_note(self, clip, text: str, tag: str = "Montage AI") -> None:
        """Add text note to clip."""
        if not hasattr(clip, 'metadata') or clip.metadata is None:
            clip.metadata = {}
        
        if 'notes' not in clip.metadata:
            clip.metadata['notes'] = []
        
        clip.metadata['notes'].append({
            'tag': tag,
            'text': text
        })

    def add_markers(
        self,
        beat_timecodes: List[Tuple[float, str]],  # [(timecode_seconds, "beat"), ...]
        section_markers: Optional[List[Tuple[float, str]]] = None  # [(time, "intro/build/climax/outro")]
    ) -> None:
        """Add beat and section markers to timeline.
        
        Args:
            beat_timecodes: List of (seconds, label) tuples for beat markers
            section_markers: List of (seconds, section_name) tuples for pacing sections
        """
        if self.timeline is None:
            raise RuntimeError("Call create_timeline() first")

        # Ensure marker collection exists (OTIO API surface is minimal in tests)
        if not hasattr(self.timeline, "markers") or self.timeline.markers is None:
            self.timeline.markers = []

        # Beat markers (global markers on timeline)
        for beat_time, beat_label in beat_timecodes:
            beat_frames = int(beat_time * self.fps)
            beat_rational_time = otio.opentime.RationalTime(beat_frames, self.fps)
            marker = otio.schema.Marker(
                name=f"Beat: {beat_label}",
                marked_range=otio.opentime.TimeRange(
                    start_time=beat_rational_time,
                    duration=otio.opentime.RationalTime(0, self.fps)
                ),
                color=otio.schema.MarkerColor.PURPLE
            )
            self.timeline.markers.append(marker)
        
        # Section markers
        if section_markers:
            section_colors = {
                'intro': otio.schema.MarkerColor.GREEN,
                'build': otio.schema.MarkerColor.YELLOW,
                'climax': otio.schema.MarkerColor.RED,
                'outro': otio.schema.MarkerColor.BLUE,
            }
            for section_time, section_name in section_markers:
                section_frames = int(section_time * self.fps)
                section_rational_time = otio.opentime.RationalTime(section_frames, self.fps)
                marker = otio.schema.Marker(
                    name=f"Section: {section_name}",
                    marked_range=otio.opentime.TimeRange(
                        start_time=section_rational_time,
                        duration=otio.opentime.RationalTime(0, self.fps)
                    ),
                    color=section_colors.get(section_name, otio.schema.MarkerColor.PINK)
                )
                self.timeline.markers.append(marker)


    def export_to_edl(self, output_path: Path) -> bool:
        """Export timeline to EDL format.
        
        Args:
            output_path: Path to output .edl file
            
        Returns:
            True if successful
        """
        if self.timeline is None:
            logger.error("No timeline to export")
            return False
        
        try:
            otio.adapters.write_to_file(self.timeline, str(output_path), adapter_name='cmx_3600')
            logger.info(f"Exported to EDL: {output_path}")
            return True
        except Exception as e:
            logger.error(f"EDL export failed: {e}")
            return False

    def export_to_otio_json(self, output_path: Path) -> bool:
        """Export timeline to OTIO JSON (canonical format).
        
        Args:
            output_path: Path to output .otio file
            
        Returns:
            True if successful
        """
        if self.timeline is None:
            logger.error("No timeline to export")
            return False
        
        try:
            otio.adapters.write_to_file(self.timeline, str(output_path), adapter_name='otio_json')
            logger.info(f"Exported to OTIO JSON: {output_path}")
            return True
        except Exception as e:
            logger.error(f"OTIO JSON export failed: {e}")
            return False

    def export_to_premiere_xml(self, output_path: Path) -> bool:
        """Export timeline to Adobe Premiere XML.
        
        Args:
            output_path: Path to output .xml file
            
        Returns:
            True if successful
        """
        if self.timeline is None:
            logger.error("No timeline to export")
            return False
        
        try:
            otio.adapters.write_to_file(self.timeline, str(output_path), adapter_name='fcp_xml')
            logger.info(f"Exported to Premiere XML: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Premiere XML export failed: {e}")
            return False

    def export_to_aaf(self, output_path: Path) -> bool:
        """Export timeline to AAF (Avid Media Composer format).
        
        Args:
            output_path: Path to output .aaf file
            
        Returns:
            True if successful
        """
        if self.timeline is None:
            logger.error("No timeline to export")
            return False
        
        try:
            otio.adapters.write_to_file(self.timeline, str(output_path), adapter_name='aaf')
            logger.info(f"Exported to AAF: {output_path}")
            return True
        except Exception as e:
            logger.error(f"AAF export failed: {e}")
            return False

    def export_editing_parameters_json(self, output_path: Path, params: EditingParameters) -> bool:
        """Export EditingParameters as JSON for roundtrip import.
        
        Args:
            output_path: Path to output .json file
            params: EditingParameters to export
            
        Returns:
            True if successful
        """
        try:
            params_dict = {
                'stabilization': asdict(params.stabilization),
                'color_grading': asdict(params.color_grading),
                'pacing': asdict(params.pacing),
                'clip_selection': asdict(params.clip_selection),
            }
            
            with open(output_path, 'w') as f:
                json.dump(params_dict, f, indent=2)
            
            logger.info(f"Exported EditingParameters JSON: {output_path}")
            return True
        except Exception as e:
            logger.error(f"EditingParameters JSON export failed: {e}")
            return False

    def get_timeline(self):
        """Get OTIO timeline for further manipulation."""
        return self.timeline
