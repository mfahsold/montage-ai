"""
Export convenience module - unified interface for montage → NLE.

Usage:
    from montage_ai.export import export_to_nle
    
    export_to_nle(
        timeline_clips=[clip1, clip2, clip3],
        editing_params=params,
        output_dir=Path("/data/output"),
        formats=["edl", "otio", "premiere"]  # desired export formats
    )
"""

from pathlib import Path
from typing import List, Literal, Optional
import logging

from montage_ai.export.otio_builder import OTIOBuilder, TimelineClipInfo
from montage_ai.editing_parameters import EditingParameters

logger = logging.getLogger(__name__)


def export_to_nle(
    timeline_clips: List[TimelineClipInfo],
    editing_params: EditingParameters,
    output_dir: Path,
    formats: List[Literal["edl", "otio", "premiere", "aaf", "params_json"]] = ["otio", "edl"],
    project_name: str = "Montage AI Export",
    fps: float = 30.0,
    beat_timecodes: Optional[List[tuple]] = None,
    section_markers: Optional[List[tuple]] = None,
) -> dict:
    """Unified export interface for montage to NLE formats.
    
    Args:
        timeline_clips: List of TimelineClipInfo objects
        editing_params: Global EditingParameters
        output_dir: Output directory for export files
        formats: Export formats (edl, otio, premiere, aaf, params_json)
        project_name: Project name for timeline
        fps: Frames per second
        beat_timecodes: Optional beat markers [(seconds, label), ...]
        section_markers: Optional section markers [(seconds, "intro"/...)]
        
    Returns:
        dict with export results: {format: (success, filepath)}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create OTIO builder and timeline
    builder = OTIOBuilder(fps=fps)
    builder.create_timeline(project_name)
    
    # Add clips to timeline
    logger.info(f"Adding {len(timeline_clips)} clips to timeline...")
    for clip_info in timeline_clips:
        builder.add_clip(clip_info, editing_params)
    
    # Add markers
    if beat_timecodes:
        logger.info(f"Adding {len(beat_timecodes)} beat markers...")
        builder.add_markers(beat_timecodes, section_markers)
    
    # Export to requested formats
    results = {}
    
    if "otio" in formats:
        otio_path = output_dir / f"{project_name.replace(' ', '_')}.otio"
        success = builder.export_to_otio_json(otio_path)
        results["otio"] = (success, otio_path if success else None)
    
    if "edl" in formats:
        edl_path = output_dir / f"{project_name.replace(' ', '_')}.edl"
        success = builder.export_to_edl(edl_path)
        results["edl"] = (success, edl_path if success else None)
    
    if "premiere" in formats:
        xml_path = output_dir / f"{project_name.replace(' ', '_')}_premiere.xml"
        success = builder.export_to_premiere_xml(xml_path)
        results["premiere"] = (success, xml_path if success else None)
    
    if "aaf" in formats:
        aaf_path = output_dir / f"{project_name.replace(' ', '_')}.aaf"
        success = builder.export_to_aaf(aaf_path)
        results["aaf"] = (success, aaf_path if success else None)
    
    if "params_json" in formats:
        params_path = output_dir / f"{project_name.replace(' ', '_')}_parameters.json"
        success = builder.export_editing_parameters_json(params_path, editing_params)
        results["params_json"] = (success, params_path if success else None)
    
    # Summary
    success_count = sum(1 for success, _ in results.values() if success)
    logger.info(f"Export complete: {success_count}/{len(results)} formats successful")
    
    return results


def create_export_summary(export_results: dict) -> str:
    """Create human-readable export summary."""
    lines = ["Export Summary", "=" * 50]
    for fmt, (success, path) in export_results.items():
        status = "✅ OK" if success else "❌ FAILED"
        lines.append(f"{fmt.upper():15} {status:10} {path or '(n/a)'}")
    return "\n".join(lines)
