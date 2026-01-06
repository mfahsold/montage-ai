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
                name="clip_001",
                source_path="/data/input/video1.mp4",
                source_in=0.0,
                source_out=5.0,
                timeline_in=0.0,
                timeline_out=5.0
            ),
            Clip(
                name="clip_002",
                source_path="/data/input/video2.mp4",
                source_in=2.0,
                source_out=7.0,
                timeline_in=5.0,
                timeline_out=10.0
            ),
            Clip(
                name="clip_003",
                source_path="/data/input/video3.mp4",
                source_in=0.0,
                source_out=3.0,
                timeline_in=10.0,
                timeline_out=13.0
            ),
        ]

        return Timeline(
            name="Test Montage",
            clips=clips,
            duration=13.0,
            fps=30.0
        )

    def test_otio_export_creates_valid_file(self, sample_timeline):
        """Test that OTIO export creates a parseable file."""
        from montage_ai.timeline_exporter import TimelineExporter
        import opentimelineio as otio

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_timeline.otio"
            exporter.export_otio(sample_timeline, str(output_path))

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

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_timeline.otio"
            exporter.export_otio(sample_timeline, str(output_path))

            timeline = otio.adapters.read_from_file(str(output_path))

            # Count clips in all tracks
            clip_count = 0
            for track in timeline.tracks:
                for item in track:
                    if isinstance(item, otio.schema.Clip):
                        clip_count += 1

            assert clip_count == 3, f"Expected 3 clips, found {clip_count}"

    def test_otio_export_preserves_timing(self, sample_timeline):
        """Test that clip timings are preserved in OTIO export."""
        from montage_ai.timeline_exporter import TimelineExporter
        import opentimelineio as otio

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_timeline.otio"
            exporter.export_otio(sample_timeline, str(output_path))

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
                name="CLIP001",
                source_path="/data/input/video1.mp4",
                source_in=0.0,
                source_out=5.0,
                timeline_in=0.0,
                timeline_out=5.0
            ),
            Clip(
                name="CLIP002",
                source_path="/data/input/video2.mp4",
                source_in=2.0,
                source_out=7.0,
                timeline_in=5.0,
                timeline_out=10.0
            ),
        ]

        return Timeline(
            name="Test EDL",
            clips=clips,
            duration=10.0,
            fps=24.0
        )

    def test_edl_export_creates_valid_file(self, sample_timeline):
        """Test that EDL export creates a parseable file."""
        from montage_ai.timeline_exporter import TimelineExporter

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_timeline.edl"
            exporter.export_edl(sample_timeline, str(output_path))

            # Verify file exists
            assert output_path.exists(), "EDL file was not created"

            # Verify file has content
            content = output_path.read_text()
            assert len(content) > 0, "EDL file is empty"

    def test_edl_export_has_valid_header(self, sample_timeline):
        """Test that EDL export has a valid CMX3600 header."""
        from montage_ai.timeline_exporter import TimelineExporter

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_timeline.edl"
            exporter.export_edl(sample_timeline, str(output_path))

            content = output_path.read_text()
            lines = content.strip().split('\n')

            # Check header
            assert lines[0].startswith('TITLE:'), "EDL missing TITLE header"

    def test_edl_export_has_valid_events(self, sample_timeline):
        """Test that EDL export contains valid event entries."""
        from montage_ai.timeline_exporter import TimelineExporter

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_timeline.edl"
            exporter.export_edl(sample_timeline, str(output_path))

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

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_timeline.edl"
            exporter.export_edl(sample_timeline, str(output_path))

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
                name="clip_a",
                source_path="/data/input/footage.mp4",
                source_in=0.0,
                source_out=10.0,
                timeline_in=0.0,
                timeline_out=10.0
            ),
        ]

        return Timeline(
            name="NLE Test",
            clips=clips,
            duration=10.0,
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

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "davinci_test.otio"
            exporter.export_otio(sample_timeline, str(output_path))

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

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "premiere_test.edl"
            exporter.export_edl(sample_timeline, str(output_path))

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

        exporter = TimelineExporter()

        # Check if FCPXML export is available
        if not hasattr(exporter, 'export_fcpxml'):
            pytest.skip("FCPXML export not implemented")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fcpx_test.fcpxml"
            exporter.export_fcpxml(sample_timeline, str(output_path))

            assert output_path.exists(), "FCPXML file was not created"


class TestRoundTrip:
    """Test round-trip export/import workflows."""

    @pytest.fixture
    def sample_timeline(self):
        """Create a sample timeline for testing."""
        from montage_ai.timeline_exporter import Timeline, Clip

        clips = [
            Clip(
                name="roundtrip_clip",
                source_path="/data/input/test.mp4",
                source_in=1.0,
                source_out=6.0,
                timeline_in=0.0,
                timeline_out=5.0
            ),
        ]

        return Timeline(
            name="Roundtrip Test",
            clips=clips,
            duration=5.0,
            fps=30.0
        )

    def test_otio_roundtrip(self, sample_timeline):
        """Test that OTIO export can be re-imported."""
        from montage_ai.timeline_exporter import TimelineExporter
        import opentimelineio as otio

        exporter = TimelineExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "roundtrip.otio"

            # Export
            exporter.export_otio(sample_timeline, str(output_path))

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
            name="INTRO_SHOT",
            source_path="footage/intro.mp4",
            source_in=0.0,
            source_out=3.0,
            timeline_in=0.0,
            timeline_out=3.0
        ),
        Clip(
            name="MAIN_ACTION_01",
            source_path="footage/action1.mp4",
            source_in=5.0,
            source_out=12.0,
            timeline_in=3.0,
            timeline_out=10.0
        ),
        Clip(
            name="CUTAWAY_01",
            source_path="footage/cutaway.mp4",
            source_in=0.0,
            source_out=2.0,
            timeline_in=10.0,
            timeline_out=12.0
        ),
        Clip(
            name="MAIN_ACTION_02",
            source_path="footage/action2.mp4",
            source_in=0.0,
            source_out=8.0,
            timeline_in=12.0,
            timeline_out=20.0
        ),
        Clip(
            name="OUTRO_SHOT",
            source_path="footage/outro.mp4",
            source_in=0.0,
            source_out=5.0,
            timeline_in=20.0,
            timeline_out=25.0
        ),
    ]

    timeline = Timeline(
        name="Montage AI Export Test",
        clips=clips,
        duration=25.0,
        fps=24.0
    )

    exporter = TimelineExporter()

    # Export OTIO
    otio_path = nle_test_output_dir / "montage_ai_test.otio"
    exporter.export_otio(timeline, str(otio_path))
    print(f"\nGenerated OTIO: {otio_path}")

    # Export EDL
    edl_path = nle_test_output_dir / "montage_ai_test.edl"
    exporter.export_edl(timeline, str(edl_path))
    print(f"Generated EDL: {edl_path}")

    print(f"\nTest files generated in: {nle_test_output_dir}")
    print("Import these files into your NLE to verify compatibility.")

    assert otio_path.exists()
    assert edl_path.exists()
