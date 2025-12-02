"""
Timeline Exporter: Export Montage AI edits to professional NLE formats

# STATUS: Work in Progress - Not yet fully tested

Supports:
- OpenTimelineIO (.otio) - Industry standard for DaVinci, Premiere, FCP, Avid
- CMX 3600 EDL (.edl) - Universal fallback for all NLEs
- Proxy generation (H.264 low-res) for smooth editing workflow

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
from typing import List, Dict, Tuple, Optional
from pathlib import Path

VERSION = "0.1.0"

# Try importing OpenTimelineIO (optional dependency)
try:
    import opentimelineio as otio
    OTIO_AVAILABLE = True
except ImportError:
    print("⚠️ OpenTimelineIO not installed. OTIO export disabled.")
    print("   Install: pip install OpenTimelineIO")
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


@dataclass
class Timeline:
    """Represents a complete edited timeline."""
    clips: List[Clip]
    audio_path: str
    total_duration: float
    fps: float = 30.0
    resolution: Tuple[int, int] = (1080, 1920)  # 9:16 vertical
    project_name: str = "fluxibri_montage"


class TimelineExporter:
    """
    Export Fluxibri montages to professional NLE formats.

    Workflow:
    1. Collect edit metadata during montage creation
    2. Generate proxies (optional, for large files)
    3. Export to OTIO + EDL
    4. Package as NLE project folder
    """

    def __init__(self, output_dir: str = "/data/output"):
        self.output_dir = output_dir
        self.proxy_dir = os.path.join(output_dir, "proxies")
        os.makedirs(self.proxy_dir, exist_ok=True)

    def export_timeline(
        self,
        timeline: Timeline,
        generate_proxies: bool = False,
        export_otio: bool = True,
        export_edl: bool = True,
        export_csv: bool = True
    ) -> Dict[str, str]:
        """
        Export timeline to NLE-compatible formats.

        Args:
            timeline: Timeline object with clips and metadata
            generate_proxies: Create H.264 proxies for editing
            export_otio: Export OpenTimelineIO file
            export_edl: Export CMX 3600 EDL file
            export_csv: Export CSV spreadsheet

        Returns:
            Dictionary of exported file paths
        """
        print(f"\n📽️ Timeline Exporter v{VERSION}")
        print(f"   Project: {timeline.project_name}")
        print(f"   Clips: {len(timeline.clips)}")
        print(f"   Duration: {timeline.total_duration:.1f}s")

        exported_files = {}

        # Generate proxies first (if requested)
        if generate_proxies:
            print("\n🎞️ Generating proxies...")
            for clip in timeline.clips:
                proxy_path = self._generate_proxy(clip.source_path)
                if proxy_path:
                    clip.proxy_path = proxy_path
                    print(f"   ✅ {os.path.basename(clip.source_path)}")

        # Export OpenTimelineIO
        if export_otio and OTIO_AVAILABLE:
            otio_path = self._export_otio(timeline)
            if otio_path:
                exported_files['otio'] = otio_path
                print(f"\n✅ OTIO exported: {otio_path}")

        # Export CMX EDL
        if export_edl:
            edl_path = self._export_edl(timeline)
            if edl_path:
                exported_files['edl'] = edl_path
                print(f"✅ EDL exported: {edl_path}")

        # Export CSV
        if export_csv:
            csv_path = self._export_csv(timeline)
            if csv_path:
                exported_files['csv'] = csv_path
                print(f"✅ CSV exported: {csv_path}")

        # Export metadata JSON
        metadata_path = self._export_metadata(timeline)
        exported_files['metadata'] = metadata_path
        print(f"✅ Metadata exported: {metadata_path}")

        # Create project package
        package_path = self._create_project_package(timeline, exported_files)
        exported_files['package'] = package_path
        print(f"\n📦 Project package: {package_path}")

        return exported_files

    def _generate_proxy(self, source_path: str) -> Optional[str]:
        """
        Generate H.264 proxy file for smooth editing.

        Proxy settings:
        - Resolution: 960x540 (half-res for 9:16 1080x1920)
        - Codec: H.264 (libx264)
        - Bitrate: 5Mbps (balance of quality/size)
        - Preset: fast (quick encoding)

        Args:
            source_path: Path to original video file

        Returns:
            Path to proxy file, or None if failed
        """
        source_name = os.path.basename(source_path)
        proxy_name = f"proxy_{source_name}"
        proxy_path = os.path.join(self.proxy_dir, proxy_name)

        # Skip if proxy already exists
        if os.path.exists(proxy_path):
            return proxy_path

        cmd = [
            "ffmpeg", "-y",
            "-i", source_path,
            "-vf", "scale=960:540",  # Half-res
            "-c:v", "libx264",
            "-preset", "fast",
            "-b:v", "5M",
            "-c:a", "aac",
            "-b:a", "128k",
            proxy_path
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return proxy_path
        except subprocess.CalledProcessError:
            print(f"   ⚠️ Proxy generation failed: {source_name}")
            return None

    def _export_otio(self, timeline: Timeline) -> Optional[str]:
        """
        Export OpenTimelineIO file.

        OTIO is the industry standard (Academy Software Foundation).
        Supports: DaVinci Resolve, Premiere Pro, FCP, Avid, Nuke, etc.

        Args:
            timeline: Timeline object

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
            media_path = clip_data.proxy_path or clip_data.source_path

            # Create media reference
            media_ref = otio.schema.ExternalReference(target_url=f"file://{media_path}")

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
                otio_clip.metadata.update(clip_data.metadata)

            video_track.append(otio_clip)

        # Add audio track
        audio_ref = otio.schema.ExternalReference(target_url=f"file://{timeline.audio_path}")
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

        with open(edl_path, 'w') as f:
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
                f.write(f"* SOURCE FILE: {clip.source_path}\n\n")

        return edl_path

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

        with open(csv_path, 'w', newline='') as f:
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
        metadata = {
            "project_name": timeline.project_name,
            "duration_sec": timeline.total_duration,
            "fps": timeline.fps,
            "resolution": timeline.resolution,
            "audio_file": timeline.audio_path,
            "clips": [
                {
                    "index": i,
                    "source_file": clip.source_path,
                    "proxy_file": clip.proxy_path,
                    "source_in": clip.start_time,
                    "source_out": clip.start_time + clip.duration,
                    "timeline_in": clip.timeline_start,
                    "timeline_out": clip.timeline_start + clip.duration,
                    "duration": clip.duration,
                    "metadata": clip.metadata or {}
                }
                for i, clip in enumerate(timeline.clips, start=1)
            ],
            "exporter_version": VERSION
        }

        json_path = os.path.join(
            self.output_dir,
            f"{timeline.project_name}_metadata.json"
        )

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return json_path

    def _create_project_package(
        self,
        timeline: Timeline,
        exported_files: Dict[str, str]
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
          └── README.txt      # Instructions

        Args:
            timeline: Timeline object
            exported_files: Paths to exported timeline files

        Returns:
            Path to project package directory
        """
        package_dir = os.path.join(self.output_dir, f"{timeline.project_name}_PROJECT")
        os.makedirs(package_dir, exist_ok=True)

        # Create README
        readme_path = os.path.join(package_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write(f"Fluxibri Timeline Export - {timeline.project_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write("This package contains everything needed to import this edit\n")
            f.write("into professional NLE software (DaVinci, Premiere, FCP, etc.)\n\n")
            f.write("Files:\n")
            f.write(f"- {timeline.project_name}.otio: OpenTimelineIO (recommended)\n")
            f.write(f"- {timeline.project_name}.edl: CMX EDL (universal fallback)\n")
            f.write(f"- metadata.json: Full edit metadata from Fluxibri\n\n")
            f.write("Import Instructions:\n")
            f.write("1. DaVinci Resolve: File > Import > Timeline > .otio\n")
            f.write("2. Adobe Premiere: File > Import > .edl\n")
            f.write("3. Final Cut Pro: File > Import > XML (use .otio)\n\n")
            f.write(f"Generated by Montage AI v{VERSION}\n")

        return package_dir

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


# Convenience function for use in smart_worker.py
def export_timeline_from_montage(
    clips_data: List[Dict],
    audio_path: str,
    total_duration: float,
    output_dir: str = "/data/output",
    project_name: str = "montage_ai",
    generate_proxies: bool = False
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
            metadata=c.get('metadata', {})
        )
        for c in clips_data
    ]

    # Create Timeline object
    timeline = Timeline(
        clips=clips,
        audio_path=audio_path,
        total_duration=total_duration,
        project_name=project_name
    )

    # Export
    exporter = TimelineExporter(output_dir=output_dir)
    return exporter.export_timeline(
        timeline,
        generate_proxies=generate_proxies,
        export_otio=True,
        export_edl=True
    )


if __name__ == "__main__":
    # Test/demo
    print(f"Timeline Exporter v{VERSION}")
    print("OpenTimelineIO available:", OTIO_AVAILABLE)
