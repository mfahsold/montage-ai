"""
Timeline Exporter: Export Montage AI edits to professional NLE formats

# STATUS: Beta - Feature Complete (Relinking & Proxies supported)

Supports:
- OpenTimelineIO (.otio) - Industry standard for DaVinci, Premiere, FCP, Avid
- CMX 3600 EDL (.edl) - Universal fallback for all NLEs
- Proxy generation (H.264 low-res) for smooth editing workflow
- Source Relinking (Switch between Proxy and Source)

Based on 2024/2025 research:
- OpenTimelineIO is Academy Software Foundation standard (Oscar-winning tech)
- CMX EDL is oldest but most compatible (1970s, still works everywhere)
- Modern workflow: Generate proxies, edit with them, relink to originals for export

Architecture:
  Fluxibri Montage Metadata → Timeline Builder → OTIO/EDL Export →
  Import into DaVinci/Premiere/FCP → Professional color/finishing
"""

import os
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from .logger import logger
from .ffmpeg_utils import build_ffmpeg_cmd
from .proxy_generator import ProxyGenerator

VERSION = "0.1.0"

# Try importing OpenTimelineIO (optional dependency)
try:
    import opentimelineio as otio
    OTIO_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ OpenTimelineIO not installed. OTIO export disabled.")
    logger.warning("   Install: pip install OpenTimelineIO")
    OTIO_AVAILABLE = False


@dataclass
class Clip:
    """Represents a single video clip in the timeline."""
    source_path: str  # Original video file
    start_time: float  # Start time in source (seconds)
    duration: float  # Duration of clip (seconds)
    timeline_start: float  # Where this clip starts in final timeline (seconds)
    proxy_path: Optional[str] = None  # Path to proxy file (if generated)
    metadata: Dict = None  # Additional metadata (energy, motion score, etc.)
    enhancement_decision: Optional[Any] = None  # EnhancementDecision from enhancement_tracking.py


@dataclass
class Timeline:
    """Represents a complete edited timeline."""
    clips: List[Clip]
    audio_path: str
    total_duration: float
    fps: float = 30.0
    resolution: Tuple[int, int] = (1080, 1920)  # 9:16 vertical
    project_name: str = "fluxibri_montage"

    def __post_init__(self):
        # Load defaults from config if not specified
        from .config import get_settings
        settings = get_settings()
        if self.fps == 30.0: # Default
             self.fps = settings.export.fps
        if self.resolution == (1080, 1920): # Default
             self.resolution = (settings.export.resolution_width, settings.export.resolution_height)
        if self.project_name == "fluxibri_montage":
             self.project_name = settings.export.project_name_template


class TimelineExporter:
    """
    Export Fluxibri montages to professional NLE formats.

    Workflow:
    1. Collect edit metadata during montage creation
    2. Generate proxies (optional, for large files)
    3. Export to OTIO + EDL
    4. Package as NLE project folder
    """

    def __init__(self, output_dir: Optional[str] = None):
        from .config import get_settings
        settings = get_settings()
        
        self.output_dir = output_dir or str(settings.paths.output_dir)
        self.proxy_dir = os.path.join(self.output_dir, "proxies")
        os.makedirs(self.proxy_dir, exist_ok=True)

    def _sanitize_metadata(self, data: Any) -> Any:
        """Recursively convert numpy types to python types for JSON/OTIO serialization."""
        if isinstance(data, dict):
            return {k: self._sanitize_metadata(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._sanitize_metadata(v) for v in data]
        elif hasattr(data, 'item'):  # numpy scalar
            try:
                return data.item()
            except (ValueError, TypeError):
                pass
        
        if hasattr(data, 'tolist'):  # numpy array
            return data.tolist()
            
        return data

    def export_timeline(
        self,
        timeline: Timeline,
        generate_proxies: bool = False,
        proxy_format: str = "h264",
        link_to_source: bool = False,
        export_otio: bool = True,
        export_edl: bool = True,
        export_xml: bool = True,
        export_csv: bool = True,
        export_recipe_card: bool = True
    ) -> Dict[str, str]:
        """
        Export timeline to NLE-compatible formats.

        Args:
            timeline: Timeline object with clips and metadata
            generate_proxies: Create proxies for editing
            proxy_format: Format for proxies ('h264', 'prores', 'dnxhr')
            link_to_source: If True, links XML/OTIO to original source files instead of proxies
            export_otio: Export OpenTimelineIO file
            export_edl: Export CMX 3600 EDL file
            export_xml: Export FCP XML v7 file (Resolve/Premiere/Kdenlive)
            export_csv: Export CSV spreadsheet
            export_recipe_card: Export human-readable recipe card for NLE recreation

        Returns:
            Dictionary of exported file paths
        """
        logger.info(f"\n📽️ Timeline Exporter v{VERSION}")
        logger.info(f"   Project: {timeline.project_name}")
        logger.info(f"   Clips: {len(timeline.clips)}")
        logger.info(f"   Duration: {timeline.total_duration:.1f}s")
        logger.info(f"   Link to Source: {link_to_source}")

        exported_files = {}

        # Generate proxies first (if requested)
        if generate_proxies:
            logger.info(f"\n🎞️ Generating proxies ({proxy_format})...")
            from .proxy_generator import ProxyGenerator
            pg = ProxyGenerator(self.proxy_dir)
            
            for clip in timeline.clips:
                try:
                    proxy_path = pg.get_proxy_path(clip.source_path, format=proxy_format)
                    if not proxy_path.exists():
                        pg.generate(clip.source_path, proxy_path, format=proxy_format)
                    
                    if proxy_path.exists():
                        clip.proxy_path = str(proxy_path)
                        logger.info(f"   ✅ {os.path.basename(clip.source_path)}")
                except Exception as e:
                    logger.error(f"   ❌ Proxy failed for {os.path.basename(clip.source_path)}: {e}")

        # Export OpenTimelineIO
        if export_otio and OTIO_AVAILABLE:
            otio_path = self._export_otio(timeline, link_to_source=link_to_source)
            if otio_path:
                exported_files['otio'] = otio_path
                logger.info(f"\n✅ OTIO exported: {otio_path}")

        # Export CMX EDL
        if export_edl:
            edl_path = self._export_edl(timeline)
            if edl_path:
                exported_files['edl'] = edl_path
                logger.info(f"✅ EDL exported: {edl_path}")

        # Export FCP XML
        if export_xml:
            xml_path = self._export_xml(timeline, link_to_source=link_to_source)
            if xml_path:
                exported_files['xml'] = xml_path
                logger.info(f"✅ XML exported: {xml_path}")

        # Export CSV
        if export_csv:
            csv_path = self._export_csv(timeline)
            if csv_path:
                exported_files['csv'] = csv_path
                logger.info(f"✅ CSV exported: {csv_path}")

        # Export metadata JSON
        metadata_path = self._export_metadata(timeline)
        exported_files['metadata'] = metadata_path
        logger.info(f"✅ Metadata exported: {metadata_path}")

        # Export recipe card (human-readable NLE recreation guide)
        if export_recipe_card:
            recipe_path = self._export_recipe_card(timeline)
            if recipe_path:
                exported_files['recipe_card'] = recipe_path
                logger.info(f"✅ Recipe card exported: {recipe_path}")

        # Create project package
        package_path = self._create_project_package(timeline, exported_files, link_to_source=link_to_source)
        exported_files['package'] = package_path
        logger.info(f"\n📦 Project package: {package_path}")

        return exported_files

    def _generate_proxy(self, source_path: str, format: str = "h264") -> Optional[str]:
        """
        Generate proxy using ProxyGenerator service.
        Delegate method to maintain API compatibility while using shared logic.
        """
        generator = ProxyGenerator(self.proxy_dir)
        path = generator.ensure_proxy(source_path, format=format)
        return str(path) if path else None

    def _export_otio(self, timeline: Timeline, link_to_source: bool = False) -> Optional[str]:
        """
        Export OpenTimelineIO file.

        OTIO is the industry standard (Academy Software Foundation).
        Supports: DaVinci Resolve, Premiere Pro, FCP, Avid, Nuke, etc.

        Args:
            timeline: Timeline object
            link_to_source: If True, link to original source files

        Returns:
            Path to .otio file
        """
        if not OTIO_AVAILABLE:
            return None

        # Create OTIO timeline
        otio_timeline = otio.schema.Timeline(name=timeline.project_name)
        video_track = otio.schema.Track(name="Video 1", kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name="Audio 1", kind=otio.schema.TrackKind.Audio)

        # Add video clips
        for clip_data in timeline.clips:
            # Use proxy if available, otherwise original
            if link_to_source:
                media_path = clip_data.source_path
            else:
                media_path = clip_data.proxy_path or clip_data.source_path

            # Create media reference (Robust URI handling)
            try:
                target_url = Path(media_path).absolute().as_uri()
            except Exception:
                # Fallback for weird paths
                target_url = f"file://{os.path.abspath(media_path)}"
            
            media_ref = otio.schema.ExternalReference(target_url=target_url)

            # Create source range (what part of the source file to use)
            source_range = otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(
                    clip_data.start_time * timeline.fps,
                    timeline.fps
                ),
                duration=otio.opentime.RationalTime(
                    clip_data.duration * timeline.fps,
                    timeline.fps
                )
            )

            # Create clip
            otio_clip = otio.schema.Clip(
                name=os.path.basename(clip_data.source_path),
                media_reference=media_ref,
                source_range=source_range
            )

            # Add metadata
            if clip_data.metadata:
                # Sanitize metadata (convert numpy types to python types)
                sanitized_metadata = self._sanitize_metadata(clip_data.metadata)
                otio_clip.metadata.update(sanitized_metadata)

            # Add enhancement metadata (NLE-compatible tracking)
            if clip_data.enhancement_decision:
                ed = clip_data.enhancement_decision
                otio_clip.metadata["montage_ai"] = {
                    "version": "1.0",
                    "enhancements": {
                        "stabilized": getattr(ed, 'stabilized', False),
                        "upscaled": getattr(ed, 'upscaled', False),
                        "denoised": getattr(ed, 'denoised', False),
                        "sharpened": getattr(ed, 'sharpened', False),
                        "color_graded": getattr(ed, 'color_graded', False),
                        "color_matched": getattr(ed, 'color_matched', False),
                        "film_grain_added": getattr(ed, 'film_grain_added', False),
                    },
                    "ai_reasoning": getattr(ed, 'ai_reasoning', None),
                }
                # Include full params if available
                if hasattr(ed, 'to_dict'):
                    otio_clip.metadata["montage_ai"]["params"] = self._sanitize_metadata(
                        ed.to_dict().get("params", {})
                    )

            video_track.append(otio_clip)

        # Add audio track
        try:
            audio_url = Path(timeline.audio_path).absolute().as_uri()
        except Exception:
            audio_url = f"file://{os.path.abspath(timeline.audio_path)}"

        audio_ref = otio.schema.ExternalReference(target_url=audio_url)
        audio_clip = otio.schema.Clip(
            name=os.path.basename(timeline.audio_path),
            media_reference=audio_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, timeline.fps),
                duration=otio.opentime.RationalTime(
                    timeline.total_duration * timeline.fps,
                    timeline.fps
                )
            )
        )
        audio_track.append(audio_clip)

        # Add tracks to timeline
        otio_timeline.tracks.append(video_track)
        otio_timeline.tracks.append(audio_track)

        # Write to file
        otio_path = os.path.join(
            self.output_dir,
            f"{timeline.project_name}.otio"
        )
        otio.adapters.write_to_file(otio_timeline, otio_path)

        return otio_path

    def _export_edl(self, timeline: Timeline) -> str:
        """
        Export CMX 3600 EDL file.

        EDL is the oldest standard (1970s) but works in every NLE.
        Format: Plain text with timecodes and cut points.

        Args:
            timeline: Timeline object

        Returns:
            Path to .edl file
        """
        edl_path = os.path.join(
            self.output_dir,
            f"{timeline.project_name}.edl"
        )
        temp_path = f"{edl_path}.tmp"

        try:
            with open(temp_path, 'w') as f:
                # EDL Header
                f.write(f"TITLE: {timeline.project_name}\n")
                f.write(f"FCM: NON-DROP FRAME\n\n")

                # EDL Events (one per clip)
                for i, clip in enumerate(timeline.clips, start=1):
                    # Timecode format: HH:MM:SS:FF (frames at 30fps)
                    src_in = self._seconds_to_timecode(clip.start_time, timeline.fps)
                    src_out = self._seconds_to_timecode(
                        clip.start_time + clip.duration,
                        timeline.fps
                    )
                    rec_in = self._seconds_to_timecode(clip.timeline_start, timeline.fps)
                    rec_out = self._seconds_to_timecode(
                        clip.timeline_start + clip.duration,
                        timeline.fps
                    )

                    # EDL line format:
                    # {event#} {reel} {track} {type} {src_in} {src_out} {rec_in} {rec_out}
                    reel_name = os.path.splitext(os.path.basename(clip.source_path))[0][:8]
                    f.write(f"{i:03d}  {reel_name:<8} V     C        ")
                    f.write(f"{src_in} {src_out} {rec_in} {rec_out}\n")

                    # Source file comment
                    f.write(f"* FROM CLIP NAME: {os.path.basename(clip.source_path)}\n")
                    f.write(f"* SOURCE FILE: {clip.source_path}\n")

                    # Enhancement tracking comments (NLE-compatible)
                    if clip.enhancement_decision and hasattr(clip.enhancement_decision, 'to_edl_comments'):
                        for comment in clip.enhancement_decision.to_edl_comments():
                            f.write(f"{comment}\n")

                    f.write("\n")
            
            # Atomic rename
            if os.path.exists(edl_path):
                os.remove(edl_path)
            os.rename(temp_path, edl_path)
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        return edl_path

    def _export_xml(self, timeline: Timeline, link_to_source: bool = False) -> str:
        """
        Export FCP XML v7 file.

        Compatible with:
        - DaVinci Resolve
        - Adobe Premiere Pro
        - Final Cut Pro 7 / X (via converter)
        - Kdenlive
        - Shotcut

        Args:
            timeline: Timeline object
            link_to_source: If True, link to original source files

        Returns:
            Path to .xml file
        """
        xml_path = os.path.join(
            self.output_dir,
            f"{timeline.project_name}.xml"
        )
        temp_path = f"{xml_path}.tmp"

        fps_int = int(timeline.fps)
        width, height = timeline.resolution

        try:
            with open(temp_path, 'w') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<!DOCTYPE xmeml>\n')
                f.write('<xmeml version="4">\n')
                f.write('  <project>\n')
                f.write(f'    <name>{timeline.project_name}</name>\n')
                f.write('    <children>\n')
                f.write(f'      <sequence id="sequence-1">\n')
                f.write(f'        <name>{timeline.project_name}</name>\n')
                f.write(f'        <duration>{int(timeline.total_duration * fps_int)}</duration>\n')
                f.write('        <rate>\n')
                f.write(f'          <timebase>{fps_int}</timebase>\n')
                f.write('          <ntsc>FALSE</ntsc>\n')
                f.write('        </rate>\n')
                f.write('        <media>\n')
                f.write('          <video>\n')
                f.write('            <format>\n')
                f.write('              <samplecharacteristics>\n')
                f.write('                <rate>\n')
                f.write(f'                  <timebase>{fps_int}</timebase>\n')
                f.write('                </rate>\n')
                f.write(f'                <width>{width}</width>\n')
                f.write(f'                <height>{height}</height>\n')
                f.write('                <pixelaspectratio>square</pixelaspectratio>\n')
                f.write('              </samplecharacteristics>\n')
                f.write('            </format>\n')
                f.write('            <track>\n')

                # Video Clips
                for i, clip in enumerate(timeline.clips, start=1):
                    clip_id = f"clipitem-{i}"
                    file_id = f"file-{i}"
                    
                    if link_to_source:
                        path = clip.source_path
                    else:
                        path = clip.proxy_path or clip.source_path
                        
                    filename = os.path.basename(path)
                    
                    # Calculate frames
                    start_frame = int(clip.timeline_start * fps_int)
                    end_frame = int((clip.timeline_start + clip.duration) * fps_int)
                    duration_frames = end_frame - start_frame
                    src_in_frame = int(clip.start_time * fps_int)
                    src_out_frame = src_in_frame + duration_frames
                    
                    # For file duration, we assume it's at least as long as the clip usage
                    # In a real scenario, we'd need the actual file duration. 
                    # We'll estimate it as src_out + 10 seconds buffer
                    file_duration = src_out_frame + (10 * fps_int)

                    f.write(f'              <clipitem id="{clip_id}">\n')
                    f.write(f'                <name>{filename}</name>\n')
                    f.write(f'                <duration>{duration_frames}</duration>\n')
                    f.write('                <rate>\n')
                    f.write(f'                  <timebase>{fps_int}</timebase>\n')
                    f.write('                  <ntsc>FALSE</ntsc>\n')
                    f.write('                </rate>\n')
                    f.write(f'                <start>{start_frame}</start>\n')
                    f.write(f'                <end>{end_frame}</end>\n')
                    f.write(f'                <in>{src_in_frame}</in>\n')
                    f.write(f'                <out>{src_out_frame}</out>\n')
                    f.write(f'                <file id="{file_id}">\n')
                    f.write(f'                  <name>{filename}</name>\n')
                    f.write(f'                  <pathurl>file://{path}</pathurl>\n')
                    f.write('                  <rate>\n')
                    f.write(f'                    <timebase>{fps_int}</timebase>\n')
                    f.write('                    <ntsc>FALSE</ntsc>\n')
                    f.write('                  </rate>\n')
                    f.write(f'                  <duration>{file_duration}</duration>\n')
                    f.write('                  <media>\n')
                    f.write('                    <video>\n')
                    f.write('                      <samplecharacteristics>\n')
                    f.write('                        <rate>\n')
                    f.write(f'                          <timebase>{fps_int}</timebase>\n')
                    f.write('                        </rate>\n')
                    f.write(f'                        <width>{width}</width>\n')
                    f.write(f'                        <height>{height}</height>\n')
                    f.write('                      </samplecharacteristics>\n')
                    f.write('                    </video>\n')
                    f.write('                  </media>\n')
                    f.write('                </file>\n')
                    
                    # Add metadata as labels/comments if possible
                    if clip.metadata:
                        f.write('                <labels>\n')
                        f.write(f'                  <label2>{clip.metadata.get("action", "")}</label2>\n')
                        f.write('                </labels>\n')
                        f.write(f'                <comments>\n')
                        f.write(f'                  <mastercomment1>{clip.metadata.get("energy", "")}</mastercomment1>\n')
                        f.write(f'                  <mastercomment2>{clip.metadata.get("shot", "")}</mastercomment2>\n')
                        f.write('                </comments>\n')

                    # Add enhancement metadata as comments
                    if clip.enhancement_decision:
                        ed = clip.enhancement_decision
                        enhancements = []
                        if getattr(ed, 'stabilized', False):
                            enhancements.append("STAB")
                        if getattr(ed, 'denoised', False):
                            enhancements.append("DENOISE")
                        if getattr(ed, 'sharpened', False):
                            enhancements.append("SHARP")
                        if getattr(ed, 'color_graded', False):
                            enhancements.append("COLOR")
                        if getattr(ed, 'upscaled', False):
                            enhancements.append("UPSCALE")
                        if enhancements:
                            if not clip.metadata:
                                f.write('                <comments>\n')
                            enhancement_str = "+".join(enhancements)
                            f.write(f'                  <mastercomment3>MONTAGE_AI: {enhancement_str}</mastercomment3>\n')
                            if hasattr(ed, 'ai_reasoning') and ed.ai_reasoning:
                                reason = ed.ai_reasoning[:80].replace("<", "&lt;").replace(">", "&gt;")
                                f.write(f'                  <mastercomment4>AI: {reason}</mastercomment4>\n')
                            if not clip.metadata:
                                f.write('                </comments>\n')

                    f.write('              </clipitem>\n')

                f.write('            </track>\n')
                f.write('          </video>\n')
                
                # Audio Track
                f.write('          <audio>\n')
                f.write('            <track>\n')
                f.write(f'              <clipitem id="clipitem-audio-1">\n')
                f.write(f'                <name>{os.path.basename(timeline.audio_path)}</name>\n')
                f.write(f'                <duration>{int(timeline.total_duration * fps_int)}</duration>\n')
                f.write('                <rate>\n')
                f.write(f'                  <timebase>{fps_int}</timebase>\n')
                f.write('                  <ntsc>FALSE</ntsc>\n')
                f.write('                </rate>\n')
                f.write('                <start>0</start>\n')
                f.write(f'                <end>{int(timeline.total_duration * fps_int)}</end>\n')
                f.write('                <in>0</in>\n')
                f.write(f'                <out>{int(timeline.total_duration * fps_int)}</out>\n')
                f.write(f'                <file id="file-audio-1">\n')
                f.write(f'                  <name>{os.path.basename(timeline.audio_path)}</name>\n')
                f.write(f'                  <pathurl>file://{timeline.audio_path}</pathurl>\n')
                f.write('                  <rate>\n')
                f.write(f'                    <timebase>{fps_int}</timebase>\n')
                f.write('                    <ntsc>FALSE</ntsc>\n')
                f.write('                  </rate>\n')
                f.write(f'                  <duration>{int(timeline.total_duration * fps_int)}</duration>\n')
                f.write('                </file>\n')
                f.write('              </clipitem>\n')
                f.write('            </track>\n')
                f.write('          </audio>\n')
                
                f.write('        </media>\n')
                f.write('      </sequence>\n')
                f.write('    </children>\n')
                f.write('  </project>\n')
                f.write('</xmeml>\n')

            # Atomic rename
            if os.path.exists(xml_path):
                os.remove(xml_path)
            os.rename(temp_path, xml_path)

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        return xml_path

    def _export_csv(self, timeline: Timeline) -> str:
        """
        Export timeline as CSV spreadsheet.

        CSV format is universal and can be opened in Excel, Google Sheets,
        or imported into custom tools. Useful for manual review and logging.

        Args:
            timeline: Timeline object

        Returns:
            Path to .csv file
        """
        import csv

        csv_path = os.path.join(
            self.output_dir,
            f"{timeline.project_name}.csv"
        )
        temp_path = f"{csv_path}.tmp"

        try:
            with open(temp_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # CSV Header
                writer.writerow([
                    "Clip #",
                    "Source File",
                    "Source In (sec)",
                    "Source Out (sec)",
                    "Timeline In (sec)",
                    "Timeline Out (sec)",
                    "Duration (sec)",
                    "Source In (TC)",
                    "Source Out (TC)",
                    "Timeline In (TC)",
                    "Timeline Out (TC)",
                    "Energy",
                    "Action",
                    "Shot Type",
                    "Notes"
                ])

                # CSV Rows (one per clip)
                for i, clip in enumerate(timeline.clips, start=1):
                    # Get metadata
                    meta = clip.metadata or {}
                    energy = meta.get('energy', 0.5)
                    action = meta.get('action', 'medium')
                    shot = meta.get('shot', 'medium')
                    notes = meta.get('notes', '')

                    # Timecodes
                    src_in_tc = self._seconds_to_timecode(clip.start_time, timeline.fps)
                    src_out_tc = self._seconds_to_timecode(clip.start_time + clip.duration, timeline.fps)
                    tl_in_tc = self._seconds_to_timecode(clip.timeline_start, timeline.fps)
                    tl_out_tc = self._seconds_to_timecode(clip.timeline_start + clip.duration, timeline.fps)

                    writer.writerow([
                        i,  # Clip #
                        clip.source_path,
                        f"{clip.start_time:.2f}",
                        f"{clip.start_time + clip.duration:.2f}",
                        f"{clip.timeline_start:.2f}",
                        f"{clip.timeline_start + clip.duration:.2f}",
                        f"{clip.duration:.2f}",
                        src_in_tc,
                        src_out_tc,
                        tl_in_tc,
                        tl_out_tc,
                        f"{energy:.2f}",
                        action,
                        shot,
                        notes
                    ])
            
            # Atomic rename
            if os.path.exists(csv_path):
                os.remove(csv_path)
            os.rename(temp_path, csv_path)
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        return csv_path

    def _export_metadata(self, timeline: Timeline) -> str:
        """
        Export timeline metadata as JSON.

        Includes:
        - Clip list with timecodes
        - Edit decisions (cut points, energy levels)
        - Fluxibri-specific metadata (motion scores, match cuts, etc.)

        Args:
            timeline: Timeline object

        Returns:
            Path to metadata JSON file
        """
        # Build clip data with enhancement tracking
        clips_data = []
        for i, clip in enumerate(timeline.clips, start=1):
            clip_dict = {
                "index": i,
                "source_file": clip.source_path,
                "proxy_file": clip.proxy_path,
                "source_in": clip.start_time,
                "source_out": clip.start_time + clip.duration,
                "timeline_in": clip.timeline_start,
                "timeline_out": clip.timeline_start + clip.duration,
                "duration": clip.duration,
                "metadata": self._sanitize_metadata(clip.metadata or {})
            }
            # Include enhancement decisions if available
            if clip.enhancement_decision and hasattr(clip.enhancement_decision, 'to_dict'):
                clip_dict["enhancement"] = self._sanitize_metadata(clip.enhancement_decision.to_dict())
            clips_data.append(clip_dict)

        metadata = {
            "project_name": timeline.project_name,
            "duration_sec": timeline.total_duration,
            "fps": timeline.fps,
            "resolution": timeline.resolution,
            "audio_file": timeline.audio_path,
            "clips": clips_data,
            "exporter_version": VERSION
        }

        json_path = os.path.join(
            self.output_dir,
            f"{timeline.project_name}_metadata.json"
        )
        temp_path = f"{json_path}.tmp"

        sanitized_metadata = self._sanitize_metadata(metadata)
        try:
            with open(temp_path, 'w') as f:
                json.dump(sanitized_metadata, f, indent=2)
            
            # Atomic rename
            if os.path.exists(json_path):
                os.remove(json_path)
            os.rename(temp_path, json_path)
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        return json_path

    def _create_project_package(
        self,
        timeline: Timeline,
        exported_files: Dict[str, str],
        link_to_source: bool = False
    ) -> str:
        """
        Create a complete project package for NLE import.

        Structure:
        {project_name}/
          ├── media/          # Original videos + audio
          ├── proxies/        # Proxy files (if generated)
          ├── {project}.otio  # OTIO timeline
          ├── {project}.edl   # CMX EDL
          ├── metadata.json   # Full edit metadata
          └── HOW_TO_CONFORM.md # Instructions

        Args:
            timeline: Timeline object
            exported_files: Paths to exported timeline files
            link_to_source: Whether XML/OTIO links to source or proxies

        Returns:
            Path to project package directory
        """
        package_dir = os.path.join(self.output_dir, f"{timeline.project_name}_PROJECT")
        os.makedirs(package_dir, exist_ok=True)

        # Create Conform Guide
        self._generate_conform_guide(package_dir, timeline, link_to_source)

        # Move exported files into package directory
        import shutil
        
        # Create media directory and copy raw footage
        media_dir = os.path.join(package_dir, "media")
        os.makedirs(media_dir, exist_ok=True)
        
        # Copy video clips
        copied_files = set()
        for clip in timeline.clips:
            if clip.source_path and os.path.exists(clip.source_path):
                if clip.source_path not in copied_files:
                    try:
                        shutil.copy2(clip.source_path, media_dir)
                        copied_files.add(clip.source_path)
                    except Exception as e:
                        logger.warning(f"Failed to copy media {clip.source_path}: {e}")

        # Copy audio
        if timeline.audio_path and os.path.exists(timeline.audio_path):
             if timeline.audio_path not in copied_files:
                try:
                    shutil.copy2(timeline.audio_path, media_dir)
                    copied_files.add(timeline.audio_path)
                except Exception as e:
                    logger.warning(f"Failed to copy audio {timeline.audio_path}: {e}")

        for key, file_path in exported_files.items():
            if key == 'package': continue
            if os.path.exists(file_path):
                dest_path = os.path.join(package_dir, os.path.basename(file_path))
                shutil.move(file_path, dest_path)
                # Update path in exported_files to point to new location
                exported_files[key] = dest_path

        return package_dir

    def _generate_conform_guide(self, package_dir: str, timeline: Timeline, link_to_source: bool) -> None:
        """Create detailed README with relinking instructions."""
        readme_path = os.path.join(package_dir, "HOW_TO_CONFORM.md")
        with open(readme_path, 'w') as f:
            f.write(f"# Fluxibri Timeline Export - {timeline.project_name}\n\n")
            f.write("This package contains everything needed to import this edit into professional NLE software.\n\n")
            
            f.write("## 📁 Folder Structure\n")
            f.write("- `/media`: Original high-res footage (Source)\n")
            f.write("- `/proxies`: Low-res proxies (if generated)\n")
            f.write(f"- `{timeline.project_name}.otio`: OpenTimelineIO (Best for Resolve/Premiere)\n")
            f.write(f"- `{timeline.project_name}.xml`: FCP XML v7 (Universal)\n")
            f.write(f"- `{timeline.project_name}.edl`: CMX EDL (Fallback)\n\n")

            f.write("## 🔗 Link Status\n")
            if link_to_source:
                f.write("**Current Link Target: SOURCE FILES (High Res)**\n")
                f.write("The XML/OTIO files are currently pointing to the original source files.\n")
                f.write("Use this for Color Grading or Final Finishing.\n\n")
            else:
                f.write("**Current Link Target: PROXIES (Low Res)**\n")
                f.write("The XML/OTIO files are currently pointing to the proxy files.\n")
                f.write("Use this for offline editing speed.\n\n")

            f.write("## 🎬 Import Instructions\n\n")
            
            f.write("### 1. DaVinci Resolve (Recommended)\n")
            f.write("1. Create a new project.\n")
            f.write("2. Go to **File > Import > Timeline...**\n")
            f.write(f"3. Select `{timeline.project_name}.xml` (or .otio).\n")
            f.write("4. In the dialog:\n")
            f.write("   - Uncheck 'Automatically import source clips into media pool'.\n")
            f.write("   - Check 'Use sizing information'.\n")
            f.write("5. **If media is offline:**\n")
            f.write("   - Select all clips in the Media Pool.\n")
            f.write("   - Right Click > **Relink Selected Clips**.\n")
            f.write("   - Point to the `media` folder (for Source) or `proxies` folder.\n\n")
            
            f.write("### 2. Adobe Premiere Pro\n")
            f.write("1. Go to **File > Import...**\n")
            f.write(f"2. Select `{timeline.project_name}.xml`.\n")
            f.write("3. If prompted for media location, point to the `media` folder.\n\n")
            
            f.write("### 3. Final Cut Pro X\n")
            f.write("1. Go to **File > Import > XML...**\n")
            f.write(f"2. Select `{timeline.project_name}.xml`.\n\n")
            
            f.write("## 💡 Workflow: Proxy to Source Relinking\n")
            f.write("If you started with proxies and want to switch to high-res source files:\n")
            f.write("1. **Resolve:** Select clips > Right Click > **Unlink Selected Clips**. Then Right Click > **Relink Selected Clips** > Point to `/media`.\n")
            f.write("2. **Premiere:** Select clips > Right Click > **Make Offline**. Then Right Click > **Link Media** > Point to `/media`.\n\n")
            
            f.write(f"Generated by Montage AI v{VERSION}\n")

    @staticmethod
    def _seconds_to_timecode(seconds: float, fps: float) -> str:
        """
        Convert seconds to SMPTE timecode (HH:MM:SS:FF).

        Args:
            seconds: Time in seconds
            fps: Frames per second

        Returns:
            Timecode string (e.g., "00:01:30:15")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * fps)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

    def _export_recipe_card(self, timeline: Timeline) -> Optional[str]:
        """
        Export human-readable recipe card for manual NLE recreation.

        This generates a Markdown file that professional editors can use to
        understand and recreate AI decisions in their preferred NLE.

        Args:
            timeline: Timeline object with enhancement decisions

        Returns:
            Path to recipe card Markdown file
        """
        recipe_path = os.path.join(
            self.output_dir,
            f"{timeline.project_name}_RECIPE_CARD.md"
        )
        temp_path = f"{recipe_path}.tmp"

        # Check if any clips have enhancement decisions
        has_enhancements = any(
            clip.enhancement_decision for clip in timeline.clips
        )

        if not has_enhancements:
            logger.debug("No enhancement decisions to export in recipe card")
            return None

        try:
            with open(temp_path, 'w') as f:
                # Header
                f.write(f"# Enhancement Recipe Card - {timeline.project_name}\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"**Total Clips:** {len(timeline.clips)}\n")

                enhanced_count = sum(
                    1 for clip in timeline.clips
                    if clip.enhancement_decision and any([
                        getattr(clip.enhancement_decision, 'stabilized', False),
                        getattr(clip.enhancement_decision, 'denoised', False),
                        getattr(clip.enhancement_decision, 'color_graded', False),
                        getattr(clip.enhancement_decision, 'upscaled', False),
                    ])
                )
                f.write(f"**Enhanced Clips:** {enhanced_count}\n\n")
                f.write("---\n\n")

                # Per-clip recipes
                for i, clip in enumerate(timeline.clips, start=1):
                    ed = clip.enhancement_decision
                    if not ed:
                        continue

                    # Check if any enhancement was applied
                    if not any([
                        getattr(ed, 'stabilized', False),
                        getattr(ed, 'denoised', False),
                        getattr(ed, 'sharpened', False),
                        getattr(ed, 'color_graded', False),
                        getattr(ed, 'upscaled', False),
                        getattr(ed, 'color_matched', False),
                        getattr(ed, 'film_grain_added', False),
                    ]):
                        continue

                    # Clip header
                    f.write(f"## Clip {i}: {os.path.basename(clip.source_path)}\n\n")
                    tc_in = self._seconds_to_timecode(clip.timeline_start, timeline.fps)
                    tc_out = self._seconds_to_timecode(clip.timeline_start + clip.duration, timeline.fps)
                    f.write(f"**Timeline:** {tc_in} - {tc_out}\n\n")

                    # Enhancement checklist
                    f.write("### Applied Enhancements:\n")
                    checks = [
                        ("Stabilization", getattr(ed, 'stabilized', False)),
                        ("Upscaling", getattr(ed, 'upscaled', False)),
                        ("Denoising", getattr(ed, 'denoised', False)),
                        ("Sharpening", getattr(ed, 'sharpened', False)),
                        ("Color Grading", getattr(ed, 'color_graded', False)),
                        ("Color Matching", getattr(ed, 'color_matched', False)),
                        ("Film Grain", getattr(ed, 'film_grain_added', False)),
                    ]
                    for name, applied in checks:
                        mark = "x" if applied else " "
                        f.write(f"- [{mark}] {name}\n")
                    f.write("\n")

                    # DaVinci Resolve instructions
                    f.write("### DaVinci Resolve Recreation:\n")
                    step = 1

                    if getattr(ed, 'stabilized', False):
                        f.write(f"{step}. **Stabilizer** (Color Page > Tracker)\n")
                        params = getattr(ed, 'stabilize_params', None)
                        if params and hasattr(params, 'to_resolve_params'):
                            for k, v in params.to_resolve_params().items():
                                if k != 'node':
                                    f.write(f"   - {k}: {v}\n")
                        else:
                            f.write("   - Mode: Perspective\n")
                            f.write("   - Smoothing: 0.30\n")
                        step += 1

                    if getattr(ed, 'denoised', False):
                        f.write(f"{step}. **Noise Reduction** (Color Page > Spatial NR)\n")
                        params = getattr(ed, 'denoise_params', None)
                        if params:
                            f.write(f"   - Luma Threshold: {getattr(params, 'spatial_strength', 0.3) * 10:.1f}\n")
                            f.write(f"   - Chroma Threshold: {getattr(params, 'chroma_strength', 0.5) * 10:.1f}\n")
                        step += 1

                    if getattr(ed, 'sharpened', False):
                        f.write(f"{step}. **Sharpening** (Color Page > Blur/Sharpen)\n")
                        params = getattr(ed, 'sharpen_params', None)
                        if params:
                            f.write(f"   - Amount: {getattr(params, 'amount', 0.4):.2f}\n")
                            f.write(f"   - Radius: {getattr(params, 'radius', 1.5):.1f}\n")
                        step += 1

                    if getattr(ed, 'color_graded', False):
                        f.write(f"{step}. **Color Grading** (Color Page > Primary Wheels)\n")
                        params = getattr(ed, 'color_grade_params', None)
                        if params:
                            f.write(f"   - Preset: {getattr(params, 'preset', 'custom')}\n")
                            f.write(f"   - Intensity: {getattr(params, 'intensity', 0.7):.0%}\n")
                            f.write(f"   - Saturation: {getattr(params, 'saturation', 1.0):.0%}\n")
                        step += 1

                    if getattr(ed, 'upscaled', False):
                        f.write(f"{step}. **Super Scale** (Edit Page > Inspector > Transform)\n")
                        params = getattr(ed, 'upscale_params', None)
                        if params:
                            f.write(f"   - Scale: {getattr(params, 'scale_factor', 2)}x\n")
                            f.write(f"   - Model: {getattr(params, 'model', 'Enhanced')}\n")
                        step += 1

                    f.write("\n")

                    # Premiere Pro instructions
                    f.write("### Premiere Pro Recreation:\n")
                    step = 1

                    if getattr(ed, 'stabilized', False):
                        f.write(f"{step}. **Warp Stabilizer** (Effects Panel)\n")
                        params = getattr(ed, 'stabilize_params', None)
                        if params and hasattr(params, 'to_premiere_params'):
                            for k, v in params.to_premiere_params().items():
                                if k != 'effect':
                                    f.write(f"   - {k}: {v}\n")
                        else:
                            f.write("   - Result: Smooth Motion\n")
                            f.write("   - Smoothness: 30%\n")
                        step += 1

                    if getattr(ed, 'color_graded', False):
                        f.write(f"{step}. **Lumetri Color** (Color Panel)\n")
                        params = getattr(ed, 'color_grade_params', None)
                        if params:
                            f.write(f"   - Creative Look: {getattr(params, 'preset', 'None')}\n")
                            f.write(f"   - Intensity: {getattr(params, 'intensity', 0.7) * 100:.0f}%\n")
                        step += 1

                    f.write("\n")

                    # AI Reasoning
                    if hasattr(ed, 'ai_reasoning') and ed.ai_reasoning:
                        f.write("### AI Reasoning:\n")
                        f.write(f"> {ed.ai_reasoning}\n\n")

                    f.write("---\n\n")

                # Footer
                f.write("## About This Export\n\n")
                f.write("This recipe card was generated by **Montage AI** to help you recreate\n")
                f.write("or adjust AI-driven enhancements in your professional NLE.\n\n")
                f.write("The parameters above can be applied manually in:\n")
                f.write("- **DaVinci Resolve**: Color Page nodes, Tracker, Super Scale\n")
                f.write("- **Premiere Pro**: Lumetri Color, Warp Stabilizer, Effects\n")
                f.write("- **Final Cut Pro**: Color Board, Stabilization, Effects\n\n")
                f.write(f"*Exporter Version: {VERSION}*\n")

            # Atomic rename
            if os.path.exists(recipe_path):
                os.remove(recipe_path)
            os.rename(temp_path, recipe_path)

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.warning(f"Failed to export recipe card: {e}")
            return None

        return recipe_path


from .config import get_settings

# Convenience function for use in smart_worker.py
def export_timeline_from_montage(
    clips_data: List[Dict],
    audio_path: str,
    total_duration: float,
    output_dir: Optional[str] = None,
    project_name: str = "montage_ai",
    generate_proxies: bool = False,
    link_to_source: bool = False,
    resolution: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None
) -> Dict[str, str]:
    """
    Convenience function to export timeline from montage data.

    Args:
        clips_data: List of clip dictionaries from create_montage()
        audio_path: Path to audio file
        total_duration: Total duration of montage
        output_dir: Where to save exports
        project_name: Name of the project
        generate_proxies: Whether to generate proxy files
        link_to_source: Whether to link to source files instead of proxies
        resolution: Optional (width, height) for the target project
        fps: Optional frame rate for the target project

    Returns:
        Dictionary of exported file paths

    Example usage in smart_worker.py:
        >>> clips_metadata = []  # Collect during montage creation
        >>> export_timeline_from_montage(
        ...     clips_metadata,
        ...     music_path,
        ...     final_video.duration,
        ...     project_name="gallery_montage_v1"
        ... )
    """
    # Convert dictionaries to Clip objects
    clips = [
        Clip(
            source_path=c['source_path'],
            start_time=c['start_time'],
            duration=c['duration'],
            timeline_start=c['timeline_start'],
            metadata=c.get('metadata', {}),
            enhancement_decision=c.get('enhancement_decision'),  # Pass enhancement tracking
        )
        for c in clips_data
    ]

    # Create Timeline object
    timeline = Timeline(
        clips=clips,
        audio_path=audio_path,
        total_duration=total_duration,
        project_name=project_name,
        fps=fps if fps else 30.0,
        resolution=resolution if resolution else (1080, 1920)
    )

    # Export
    exporter = TimelineExporter(output_dir=output_dir)
    return exporter.export_timeline(
        timeline,
        generate_proxies=generate_proxies,
        link_to_source=link_to_source,
        export_otio=True,
        export_edl=True,
        export_xml=True
    )


if __name__ == "__main__":
    # Test/demo
    print(f"Timeline Exporter v{VERSION}")
    print("OpenTimelineIO available:", OTIO_AVAILABLE)
