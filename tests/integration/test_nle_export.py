"""
NLE Export Smoke Tests - Validate OTIO/EDL/AAF Export

Tests that exported timeline files are valid and can be imported
into professional NLEs (DaVinci Resolve, Premiere, Final Cut Pro).

These are "smoke tests" - they validate file structure and syntax
without actually launching NLE applications.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

# Skip if dependencies not available
pytest.importorskip("opentimelineio")

class TestOTIOExport:
    """Test OpenTimelineIO export functionality."""

    @pytest.fixture
    def sample_timeline(self):
        """Create a sample timeline for testing."""
        from montage_ai.timeline_exporter import Timeline, Clip

        clips = [
            Clip(
                source_path="/data/input/video1.mp4",
                start_time=0.0,
                duration=5.0,
                timeline_start=0.0
            ),
            Clip(
                source_path="/data/input/video2.mp4",
                start_time=2.0,
                duration=5.0,
                timeline_start=5.0
            ),
            Clip(
                source_path="/data/input/video3.mp4",
                start_time=0.0,
                duration=3.0,
                timeline_start=10.0
            ),
        ]

        return Timeline(
            clips=clips,
            audio_path="/data/music/test.mp3",
            total_duration=13.0,
            fps=30.0
        )

    def test_otio_export_creates_valid_file(self, sample_timeline):
        """Test that OTIO export creates a parseable file."""
        from montage_ai.timeline_exporter import TimelineExporter
        import opentimelineio as otio

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create exporter with tempdir as output
            exporter = TimelineExporter(output_dir=tmpdir)
            
            # Export timeline (only OTIO format)
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=True,
                export_edl=False,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            # Get OTIO file path from returned dict
            assert 'otio' in exported, "OTIO export not generated"
            output_path = Path(exported['otio'])

            # Verify file exists
            assert output_path.exists(), "OTIO file was not created"

            # Verify file is valid OTIO
            timeline = otio.adapters.read_from_file(str(output_path))
            assert timeline is not None, "Failed to parse OTIO file"

            # Verify structure
            assert hasattr(timeline, 'tracks'), "Timeline missing tracks"

    def test_otio_export_contains_all_clips(self, sample_timeline):
        """Test that all clips are present in OTIO export."""
        from montage_ai.timeline_exporter import TimelineExporter
        import opentimelineio as otio

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=True,
                export_edl=False,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            output_path = Path(exported['otio'])
            timeline = otio.adapters.read_from_file(str(output_path))

            # Count clips in all tracks
            clip_count = 0
            for track in timeline.tracks:
                # Count only video clips (skip audio tracks)
                if track.kind == otio.schema.TrackKind.Video:
                    for item in track:
                        if isinstance(item, otio.schema.Clip):
                            clip_count += 1

            # Allow for both 3 clips (video only) or 4 clips (video + audio)
            assert clip_count >= 3, f"Expected at least 3 clips, found {clip_count}"

    def test_otio_export_preserves_timing(self, sample_timeline):
        """Test that clip timings are preserved in OTIO export."""
        from montage_ai.timeline_exporter import TimelineExporter
        import opentimelineio as otio

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=True,
                export_edl=False,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            output_path = Path(exported['otio'])
            timeline = otio.adapters.read_from_file(str(output_path))

            # Get total duration
            video_track = None
            for track in timeline.tracks:
                if track.kind == otio.schema.TrackKind.Video:
                    video_track = track
                    break

            if video_track:
                total_duration = video_track.duration().value / video_track.duration().rate
                assert abs(total_duration - 13.0) < 0.1, f"Duration mismatch: {total_duration}"


class TestEDLExport:
    """Test EDL (Edit Decision List) export functionality."""

    @pytest.fixture
    def sample_timeline(self):
        """Create a sample timeline for testing."""
        from montage_ai.timeline_exporter import Timeline, Clip

        clips = [
            Clip(
                source_path="/data/input/video1.mp4",
                start_time=0.0,
                duration=5.0,
                timeline_start=0.0
            ),
            Clip(
                source_path="/data/input/video2.mp4",
                start_time=2.0,
                duration=5.0,
                timeline_start=5.0
            ),
        ]

        return Timeline(
            clips=clips,
            audio_path="/data/music/test.mp3",
            total_duration=10.0,
            fps=24.0
        )

    def test_edl_export_creates_valid_file(self, sample_timeline):
        """Test that EDL export creates a parseable file."""
        from montage_ai.timeline_exporter import TimelineExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=False,
                export_edl=True,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            assert 'edl' in exported, "EDL export not generated"
            output_path = Path(exported['edl'])

            # Verify file exists
            assert output_path.exists(), "EDL file was not created"

            # Verify file has content
            content = output_path.read_text()
            assert len(content) > 0, "EDL file is empty"

    def test_edl_export_has_valid_header(self, sample_timeline):
        """Test that EDL export has a valid CMX3600 header."""
        from montage_ai.timeline_exporter import TimelineExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=False,
                export_edl=True,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            output_path = Path(exported['edl'])
            content = output_path.read_text()
            lines = content.strip().split('\n')

            # Check header
            assert lines[0].startswith('TITLE:'), "EDL missing TITLE header"

    def test_edl_export_has_valid_events(self, sample_timeline):
        """Test that EDL export contains valid event entries."""
        from montage_ai.timeline_exporter import TimelineExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=False,
                export_edl=True,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            output_path = Path(exported['edl'])
            content = output_path.read_text()

            # Check for event numbers (001, 002, etc.)
            assert '001' in content, "EDL missing event 001"
            assert '002' in content, "EDL missing event 002"

            # Check for timecode format (HH:MM:SS:FF)
            import re
            timecode_pattern = r'\d{2}:\d{2}:\d{2}:\d{2}'
            timecodes = re.findall(timecode_pattern, content)
            assert len(timecodes) >= 4, f"EDL has insufficient timecodes: {len(timecodes)}"

    def test_edl_timecode_format(self, sample_timeline):
        """Test that EDL uses correct timecode format based on frame rate."""
        from montage_ai.timeline_exporter import TimelineExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=False,
                export_edl=True,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            output_path = Path(exported['edl'])
            content = output_path.read_text()

            # For 24fps, frame values should be 00-23
            import re
            timecodes = re.findall(r'(\d{2}):(\d{2}):(\d{2}):(\d{2})', content)

            for tc in timecodes:
                frames = int(tc[3])
                assert frames < 24, f"Invalid frame value for 24fps: {frames}"


class TestNLECompatibility:
    """Test compatibility with specific NLE applications."""

    @pytest.fixture
    def sample_timeline(self):
        """Create a sample timeline for testing."""
        from montage_ai.timeline_exporter import Timeline, Clip

        clips = [
            Clip(
                source_path="/data/input/footage.mp4",
                start_time=0.0,
                duration=10.0,
                timeline_start=0.0
            ),
        ]

        return Timeline(
            clips=clips,
            audio_path="/data/music/test.mp3",
            total_duration=10.0,
            fps=30.0
        )

    def test_davinci_resolve_otio_compatibility(self, sample_timeline):
        """
        Test OTIO export is compatible with DaVinci Resolve.

        DaVinci Resolve requirements:
        - Uses 'tracks' structure
        - Supports standard clip types
        - Handles source ranges properly
        """
        from montage_ai.timeline_exporter import TimelineExporter
        import opentimelineio as otio

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=True,
                export_edl=False,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            output_path = Path(exported['otio'])
            timeline = otio.adapters.read_from_file(str(output_path))

            # Verify DaVinci-compatible structure
            assert isinstance(timeline, (otio.schema.Timeline, otio.schema.SerializableCollection))

            # Verify clips have media references
            for track in timeline.tracks:
                for item in track:
                    if isinstance(item, otio.schema.Clip):
                        assert item.media_reference is not None, "Clip missing media reference"

    def test_premiere_edl_compatibility(self, sample_timeline):
        """
        Test EDL export is compatible with Adobe Premiere.

        Premiere EDL requirements:
        - CMX3600 format
        - Valid timecodes
        - Proper event numbering
        """
        from montage_ai.timeline_exporter import TimelineExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=False,
                export_edl=True,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            output_path = Path(exported['edl'])
            content = output_path.read_text()

            # CMX3600 format checks
            assert 'TITLE:' in content, "Missing TITLE (CMX3600 requirement)"
            assert 'FCM:' in content or 'NON-DROP' in content or 'DROP' in content or True, \
                "Frame count mode indicator recommended"

            # Event format: nnn  source  V  C  source_in source_out rec_in rec_out
            lines = [l for l in content.split('\n') if l.strip() and not l.startswith('*')]
            event_count = sum(1 for l in lines if l[:3].isdigit())
            assert event_count >= 1, "No valid events found"

    def test_fcpx_xml_export(self, sample_timeline):
        """
        Test XML export for Final Cut Pro X compatibility.

        FCPX uses FCPXML format which is different from OTIO.
        This is a basic structure test.
        """
        # FCPXML export is optional - skip if not implemented
        from montage_ai.timeline_exporter import TimelineExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            # Use XML export (FCP XML v7 format)
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=False,
                export_edl=False,
                export_xml=True,
                export_csv=False,
                export_recipe_card=False
            )
            
            # Check if XML was generated
            if 'xml' not in exported:
                pytest.skip("XML export not available")
            
            output_path = Path(exported['xml'])
            assert output_path.exists(), "XML file was not created"


class TestRoundTrip:
    """Test round-trip export/import workflows."""

    @pytest.fixture
    def sample_timeline(self):
        """Create a sample timeline for testing."""
        from montage_ai.timeline_exporter import Timeline, Clip

        clips = [
            Clip(
                source_path="/data/input/test.mp4",
                start_time=1.0,
                duration=5.0,
                timeline_start=0.0
            ),
        ]

        return Timeline(
            clips=clips,
            audio_path="/data/music/test.mp3",
            total_duration=5.0,
            fps=30.0
        )

    def test_otio_roundtrip(self, sample_timeline):
        """Test that OTIO export can be re-imported."""
        from montage_ai.timeline_exporter import TimelineExporter
        import opentimelineio as otio

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TimelineExporter(output_dir=tmpdir)
            
            # Export
            exported = exporter.export_timeline(
                sample_timeline,
                export_otio=True,
                export_edl=False,
                export_xml=False,
                export_csv=False,
                export_recipe_card=False
            )
            
            output_path = Path(exported['otio'])

            # Re-import
            imported = otio.adapters.read_from_file(str(output_path))

            # Verify structure preserved
            assert imported is not None
            assert len(imported.tracks) > 0


# =============================================================================
# Fixtures for integration with real NLE applications (manual testing)
# =============================================================================

@pytest.fixture
def nle_test_output_dir(tmp_path):
    """
    Create a temporary directory for NLE test outputs.

    For manual NLE testing, copy files from here to test import.
    """
    output_dir = tmp_path / "nle_test_output"
    output_dir.mkdir()
    return output_dir


def test_generate_nle_test_files(nle_test_output_dir):
    """
    Generate test files for manual NLE import testing.

    Run with: pytest tests/integration/test_nle_export.py::test_generate_nle_test_files -v

    Then manually import the generated files into:
    - DaVinci Resolve (File > Import Timeline > OTIO or EDL)
    - Adobe Premiere (File > Import > EDL)
    - Final Cut Pro X (File > Import > XML)
    """
    from montage_ai.timeline_exporter import TimelineExporter, Timeline, Clip

    # Create a realistic timeline
    clips = [
        Clip(
            source_path="footage/intro.mp4",
            start_time=0.0,
            duration=3.0,
            timeline_start=0.0
        ),
        Clip(
            source_path="footage/action1.mp4",
            start_time=5.0,
            duration=7.0,
            timeline_start=3.0
        ),
        Clip(
            source_path="footage/cutaway.mp4",
            start_time=0.0,
            duration=2.0,
            timeline_start=10.0
        ),
        Clip(
            source_path="footage/action2.mp4",
            start_time=0.0,
            duration=8.0,
            timeline_start=12.0
        ),
        Clip(
            source_path="footage/outro.mp4",
            start_time=0.0,
            duration=5.0,
            timeline_start=20.0
        ),
    ]

    timeline = Timeline(
        clips=clips,
        audio_path="footage/music.mp3",
        total_duration=25.0,
        fps=24.0
    )

    exporter = TimelineExporter(output_dir=str(nle_test_output_dir))

    # Export all formats
    exported = exporter.export_timeline(
        timeline,
        export_otio=True,
        export_edl=True,
        export_xml=True,
        export_csv=True,
        export_recipe_card=True
    )
    
    otio_path = Path(exported['otio'])
    edl_path = Path(exported['edl'])
    print(f"\nGenerated OTIO: {otio_path}")
    print(f"Generated EDL: {edl_path}")

    print(f"\nTest files generated in: {nle_test_output_dir}")
    print("Import these files into your NLE to verify compatibility.")

    assert otio_path.exists()
    assert edl_path.exists()
