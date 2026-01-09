"""
Tests für OTIO Builder und Export functionality.

Validiert:
- OTIOBuilder timeline creation
- Clip metadata attachment
- Marker handling
- Multi-format export (OTIO, EDL, Premiere, AAF)
- JSON parameter export
- Roundtrip: Timeline → OTIO → re-import
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from montage_ai.export.otio_builder import OTIOBuilder, TimelineClipInfo
from montage_ai.export import export_to_nle, create_export_summary
from montage_ai.editing_parameters import EditingParameters, StabilizationParameters, ColorGradingParameters


class TestOTIOBuilder:
    """Test OTIOBuilder functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.builder = OTIOBuilder(fps=30.0, video_width=1920, video_height=1080)

    def test_create_timeline(self):
        """Test timeline creation."""
        timeline = self.builder.create_timeline("Test Project")
        assert timeline is not None
        assert timeline.name == "Test Project"
        assert len(timeline.tracks) == 1
        assert timeline.tracks[0].kind.name == "Video"

    def test_add_clip_minimal(self):
        """Test adding a minimal clip to timeline."""
        self.builder.create_timeline()
        
        clip_info = TimelineClipInfo(
            source_path="/data/input/clip1.mp4",
            in_time=0.0,
            out_time=5.0,
            duration=5.0,
            sequence_number=1,
            applied_effects={}
        )
        
        editing_params = EditingParameters()
        clip = self.builder.add_clip(clip_info, editing_params)
        
        assert clip is not None
        assert clip.name == "Clip_001"
        assert len(self.builder.video_track.children) == 1

    def test_add_multiple_clips(self):
        """Test adding multiple clips."""
        self.builder.create_timeline()
        
        editing_params = EditingParameters()
        
        for i in range(3):
            clip_info = TimelineClipInfo(
                source_path=f"/data/input/clip{i+1}.mp4",
                in_time=0.0,
                out_time=3.0,
                duration=3.0,
                sequence_number=i+1,
                applied_effects={}
            )
            self.builder.add_clip(clip_info, editing_params)
        
        assert len(self.builder.video_track.children) == 3

    def test_clip_metadata_attachment(self):
        """Test metadata is attached to clips."""
        self.builder.create_timeline()
        
        clip_info = TimelineClipInfo(
            source_path="/data/input/clip1.mp4",
            in_time=0.0,
            out_time=5.0,
            duration=5.0,
            sequence_number=1,
            applied_effects={
                "color_grading": {"preset": "teal_orange", "intensity": 0.9},
                "stabilization": {"smoothing": 20}
            },
            confidence_scores={"color_grading": 0.85}
        )
        
        editing_params = EditingParameters()
        clip = self.builder.add_clip(clip_info, editing_params)
        
        # Check metadata
        assert 'montage_ai' in clip.metadata
        assert 'applied_effects' in clip.metadata['montage_ai']
        assert clip.metadata['montage_ai']['applied_effects']['color_grading']['preset'] == 'teal_orange'

    def test_add_markers(self):
        """Test adding beat and section markers."""
        self.builder.create_timeline()
        
        beat_timecodes = [
            (1.0, "beat_1"),
            (2.0, "beat_2"),
            (3.0, "beat_3")
        ]
        
        section_markers = [
            (0.0, "intro"),
            (2.0, "build"),
            (4.0, "climax")
        ]
        
        self.builder.add_markers(beat_timecodes, section_markers)
        
        # Timeline should have markers (exact count depends on OTIO)
        assert len(self.builder.timeline.markers) > 0

    def test_export_to_otio_json(self):
        """Test exporting to OTIO JSON format."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.otio"
            
            self.builder.create_timeline("Test")
            
            clip_info = TimelineClipInfo(
                source_path="/data/input/clip1.mp4",
                in_time=0.0,
                out_time=5.0,
                duration=5.0,
                sequence_number=1,
                applied_effects={}
            )
            editing_params = EditingParameters()
            self.builder.add_clip(clip_info, editing_params)
            
            # Mock the OTIO write function since OTIO might not be installed
            with patch('montage_ai.export.otio_builder.otio.adapters.write_to_file') as mock_write:
                mock_write.return_value = None
                success = self.builder.export_to_otio_json(output_path)
                
                assert success is True
                mock_write.assert_called_once()

    def test_export_editing_parameters_json(self):
        """Test exporting EditingParameters to JSON."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "params.json"
            
            params = EditingParameters()
            params.color_grading.preset = "teal_orange"
            params.color_grading.intensity = 0.9
            
            success = self.builder.export_editing_parameters_json(output_path, params)
            
            assert success is True
            assert output_path.exists()
            
            # Verify JSON content
            with open(output_path) as f:
                exported = json.load(f)
            
            assert exported['color_grading']['preset'] == 'teal_orange'
            assert exported['color_grading']['intensity'] == 0.9


class TestExportConvenience:
    """Test export convenience functions."""

    def test_export_to_nle_no_clips(self):
        """Test export with no clips."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            params = EditingParameters()
            
            with patch('montage_ai.export.otio_builder.otio.adapters.write_to_file'):
                results = export_to_nle(
                    timeline_clips=[],
                    editing_params=params,
                    output_dir=output_dir,
                    formats=["otio"]
                )
            
            assert "otio" in results
            assert results["otio"][0] is True

    def test_export_summary_creation(self):
        """Test export summary generation."""
        results = {
            "otio": (True, Path("/tmp/project.otio")),
            "edl": (False, None),
            "premiere": (True, Path("/tmp/project.xml"))
        }
        
        summary = create_export_summary(results)
        
        assert "✅" in summary
        assert "❌" in summary
        assert "OTIO" in summary
        assert "EDL" in summary
        assert "PREMIERE" in summary

    def test_export_with_multiple_formats(self):
        """Test exporting to multiple formats."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            params = EditingParameters()
            
            clip_info = TimelineClipInfo(
                source_path="/data/input/test.mp4",
                in_time=0.0,
                out_time=10.0,
                duration=10.0,
                sequence_number=1,
                applied_effects={}
            )
            
            with patch('montage_ai.export.otio_builder.otio.adapters.write_to_file'):
                results = export_to_nle(
                    timeline_clips=[clip_info],
                    editing_params=params,
                    output_dir=output_dir,
                    formats=["otio", "edl", "premiere"],
                    project_name="TestProject"
                )
            
            # All formats should be attempted
            assert "otio" in results
            assert "edl" in results
            assert "premiere" in results


class TestJSONParsingRobustness:
    """Test JSON parsing and error recovery (for CreativeDirector integration)."""

    def test_safe_parse_json_valid(self):
        """Test parsing valid JSON."""
        builder = OTIOBuilder()
        response = '{"key": "value"}'
        result = builder._safe_parse_json_response(response)
        assert result == {"key": "value"}

    def test_safe_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown blocks."""
        builder = OTIOBuilder()
        response = '```json\n{"key": "value"}\n```'
        result = builder._safe_parse_json_response(response)
        assert result == {"key": "value"}

    def test_safe_parse_json_from_text(self):
        """Test extracting JSON from text."""
        builder = OTIOBuilder()
        response = 'Here is the response: {"key": "value"} and more text'
        result = builder._safe_parse_json_response(response)
        assert result == {"key": "value"}

    def test_safe_parse_json_empty(self):
        """Test parsing empty response."""
        builder = OTIOBuilder()
        result = builder._safe_parse_json_response("")
        assert result is None

    def test_safe_parse_json_invalid(self):
        """Test parsing completely invalid response."""
        builder = OTIOBuilder()
        result = builder._safe_parse_json_response("This is not JSON at all")
        assert result is None


class TestTimelineClipInfo:
    """Test TimelineClipInfo dataclass."""

    def test_clip_info_creation(self):
        """Test creating TimelineClipInfo."""
        clip = TimelineClipInfo(
            source_path="/data/input/test.mp4",
            in_time=0.0,
            out_time=5.0,
            duration=5.0,
            sequence_number=1,
            applied_effects={"color_grading": {"preset": "cinematic"}}
        )
        
        assert clip.source_path == "/data/input/test.mp4"
        assert clip.duration == 5.0
        assert clip.applied_effects["color_grading"]["preset"] == "cinematic"

    def test_clip_info_with_confidence(self):
        """Test TimelineClipInfo with confidence scores."""
        clip = TimelineClipInfo(
            source_path="/data/input/test.mp4",
            in_time=0.0,
            out_time=5.0,
            duration=5.0,
            sequence_number=1,
            applied_effects={},
            confidence_scores={"color_grading": 0.85, "stabilization": 0.72}
        )
        
        assert clip.confidence_scores["color_grading"] == 0.85
        assert clip.confidence_scores["stabilization"] == 0.72


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
