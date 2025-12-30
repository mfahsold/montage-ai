"""
FFmpeg Tools - LLM-Callable Video Processing Interface

Replaces external FFmpeg-MCP with direct tool functions that can be
called by the Creative Director LLM for video manipulation tasks.

Architecture:
    Creative Director (LLM) → FFmpeg Tools → subprocess/FFmpeg → Video Output

Each tool is defined as a dataclass with:
- name: Function identifier for LLM
- description: Natural language description for LLM understanding  
- parameters: Expected inputs with types and descriptions
- execute(): Actual implementation

Usage:
    from .ffmpeg_tools import FFmpegToolkit
    
    toolkit = FFmpegToolkit()
    result = toolkit.execute("extract_frames", {
        "input": "/data/input/video.mp4",
        "timestamps": [1.5, 3.0, 5.5],
        "output_dir": "/tmp/frames"
    })
"""

import os
import subprocess
import json
import tempfile
from .config import get_settings
import shutil
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "array", "boolean"
    description: str
    required: bool = True
    default: Any = None


@dataclass 
class FFmpegTool:
    """
    Tool definition for LLM calling.
    
    Each tool wraps an FFmpeg operation that the Creative Director
    can invoke via natural language interpretation.
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    execute: Callable[[Dict[str, Any]], Dict[str, Any]] = field(default=None, repr=False)
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.required:
                required.append(param.name)
                
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class FFmpegToolkit:
    """
    Toolkit of FFmpeg operations callable by LLM.
    
    Provides video processing capabilities:
    - Frame extraction
    - Segment cutting
    - Transitions
    - Color grading
    - Audio mixing
    - Format conversion
    - Stabilization
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        if temp_dir is None:
            temp_dir = str(get_settings().paths.temp_dir / "ffmpeg_tools")
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.tools: Dict[str, FFmpegTool] = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register all available FFmpeg tools."""
        
        # Tool 1: Extract Frames
        self.tools["extract_frames"] = FFmpegTool(
            name="extract_frames",
            description="Extract frames from video at specific timestamps. Useful for thumbnail generation, keyframe analysis, or creating image sequences.",
            parameters=[
                ToolParameter("input", "string", "Path to input video file"),
                ToolParameter("timestamps", "array", "List of timestamps in seconds to extract frames at"),
                ToolParameter("output_dir", "string", "Directory to save extracted frames"),
                ToolParameter("format", "string", "Output format: jpg, png", required=False, default="jpg")
            ],
            execute=self._extract_frames
        )
        
        # Tool 2: Create Segment
        self.tools["create_segment"] = FFmpegTool(
            name="create_segment",
            description="Extract a video segment between two timestamps. Creates a new video file with the specified portion.",
            parameters=[
                ToolParameter("input", "string", "Path to input video file"),
                ToolParameter("start", "number", "Start timestamp in seconds"),
                ToolParameter("end", "number", "End timestamp in seconds"),
                ToolParameter("output", "string", "Path to output video file"),
                ToolParameter("codec", "string", "Video codec: copy, h264, h265", required=False, default="copy")
            ],
            execute=self._create_segment
        )
        
        # Tool 3: Apply Transition
        self.tools["apply_transition"] = FFmpegTool(
            name="apply_transition",
            description="Apply a transition effect between two video clips. Supports crossfade, wipe, slide, and other effects.",
            parameters=[
                ToolParameter("clip_a", "string", "Path to first video clip"),
                ToolParameter("clip_b", "string", "Path to second video clip"),
                ToolParameter("transition", "string", "Transition type: crossfade, wipe, slide, fade"),
                ToolParameter("duration", "number", "Transition duration in seconds"),
                ToolParameter("output", "string", "Path to output video file")
            ],
            execute=self._apply_transition
        )
        
        # Tool 4: Color Grade
        self.tools["color_grade"] = FFmpegTool(
            name="color_grade",
            description="Apply color grading to video using LUT files or built-in presets. Adjusts colors, contrast, and mood.",
            parameters=[
                ToolParameter("input", "string", "Path to input video file"),
                ToolParameter("preset", "string", "Preset name: cinematic, vintage, cold, warm, noir, vivid"),
                ToolParameter("intensity", "number", "Effect intensity from 0.0 to 1.0"),
                ToolParameter("output", "string", "Path to output video file"),
                ToolParameter("lut_path", "string", "Optional custom LUT file path", required=False)
            ],
            execute=self._color_grade
        )
        
        # Tool 5: Mix Audio
        self.tools["mix_audio"] = FFmpegTool(
            name="mix_audio",
            description="Mix audio tracks together or replace video audio. Can adjust volumes and apply crossfades.",
            parameters=[
                ToolParameter("video", "string", "Path to video file"),
                ToolParameter("audio", "string", "Path to audio file to mix/replace"),
                ToolParameter("mode", "string", "Mode: replace, mix, ducking"),
                ToolParameter("video_volume", "number", "Video audio volume 0.0-1.0", required=False, default=1.0),
                ToolParameter("audio_volume", "number", "New audio volume 0.0-1.0", required=False, default=1.0),
                ToolParameter("output", "string", "Path to output video file")
            ],
            execute=self._mix_audio
        )
        
        # Tool 6: Resize/Scale
        self.tools["resize"] = FFmpegTool(
            name="resize",
            description="Resize video to specific dimensions or scale factor. Maintains aspect ratio by default.",
            parameters=[
                ToolParameter("input", "string", "Path to input video file"),
                ToolParameter("width", "number", "Target width in pixels (-1 for auto)"),
                ToolParameter("height", "number", "Target height in pixels (-1 for auto)"),
                ToolParameter("output", "string", "Path to output video file"),
                ToolParameter("algorithm", "string", "Scaling algorithm: lanczos, bicubic, bilinear", required=False, default="lanczos")
            ],
            execute=self._resize
        )
        
        # Tool 7: Concatenate
        self.tools["concatenate"] = FFmpegTool(
            name="concatenate",
            description="Concatenate multiple video clips into a single video file in the order specified.",
            parameters=[
                ToolParameter("clips", "array", "List of video file paths to concatenate"),
                ToolParameter("output", "string", "Path to output video file"),
                ToolParameter("reencode", "boolean", "Force re-encoding for compatibility", required=False, default=False)
            ],
            execute=self._concatenate
        )
        
        # Tool 8: Get Video Info
        self.tools["get_video_info"] = FFmpegTool(
            name="get_video_info",
            description="Get detailed information about a video file including duration, resolution, codec, framerate, and audio properties.",
            parameters=[
                ToolParameter("input", "string", "Path to video file")
            ],
            execute=self._get_video_info
        )
        
        # Tool 9: Generate Thumbnail Grid
        self.tools["thumbnail_grid"] = FFmpegTool(
            name="thumbnail_grid",
            description="Generate a grid of thumbnails from video for quick visual overview. Useful for video preview.",
            parameters=[
                ToolParameter("input", "string", "Path to input video file"),
                ToolParameter("columns", "number", "Number of columns in grid"),
                ToolParameter("rows", "number", "Number of rows in grid"),
                ToolParameter("output", "string", "Path to output image file")
            ],
            execute=self._thumbnail_grid
        )
        
        # Tool 10: Speed Change
        self.tools["speed_change"] = FFmpegTool(
            name="speed_change",
            description="Change video playback speed. Can create slow-motion or time-lapse effects.",
            parameters=[
                ToolParameter("input", "string", "Path to input video file"),
                ToolParameter("speed", "number", "Speed multiplier: 0.5 = half speed, 2.0 = double speed"),
                ToolParameter("output", "string", "Path to output video file"),
                ToolParameter("preserve_audio_pitch", "boolean", "Keep audio pitch when changing speed", required=False, default=True)
            ],
            execute=self._speed_change
        )
    
    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Dictionary of parameters
            
        Returns:
            Dict with 'success', 'output', and optionally 'error'
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys())
            }
        
        tool = self.tools[tool_name]
        
        # Validate required parameters
        for param in tool.parameters:
            if param.required and param.name not in params:
                return {
                    "success": False,
                    "error": f"Missing required parameter: {param.name}"
                }
        
        # Apply defaults
        for param in tool.parameters:
            if param.name not in params and param.default is not None:
                params[param.name] = param.default
        
        try:
            return tool.execute(params)
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling schema for all tools."""
        return [tool.to_openai_schema() for tool in self.tools.values()]
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List available tools with descriptions."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]
    
    # ========================================================================
    # Tool Implementations
    # ========================================================================
    
    def _run_ffmpeg(self, args: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run FFmpeg command with standard options."""
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
        return subprocess.run(cmd, capture_output=capture_output, text=True)
    
    def _run_ffprobe(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run FFprobe command."""
        cmd = ["ffprobe", "-v", "quiet"] + args
        return subprocess.run(cmd, capture_output=True, text=True)
    
    def _extract_frames(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract frames at specific timestamps."""
        input_path = params["input"]
        timestamps = params["timestamps"]
        output_dir = Path(params["output_dir"])
        fmt = params.get("format", "jpg")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted = []
        for i, ts in enumerate(timestamps):
            output_path = output_dir / f"frame_{i:04d}_{ts:.2f}.{fmt}"
            
            result = self._run_ffmpeg([
                "-ss", str(ts),
                "-i", input_path,
                "-vframes", "1",
                "-q:v", "2",
                str(output_path)
            ])
            
            if result.returncode == 0:
                extracted.append(str(output_path))
        
        return {
            "success": True,
            "output": extracted,
            "count": len(extracted)
        }
    
    def _create_segment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract video segment."""
        input_path = params["input"]
        start = params["start"]
        end = params["end"]
        output_path = params["output"]
        codec = params.get("codec", "copy")
        
        duration = end - start
        
        codec_args = ["-c", "copy"] if codec == "copy" else ["-c:v", codec, "-c:a", "aac"]
        
        result = self._run_ffmpeg([
            "-ss", str(start),
            "-i", input_path,
            "-t", str(duration),
            *codec_args,
            output_path
        ])
        
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        
        return {
            "success": True,
            "output": output_path,
            "duration": duration
        }
    
    def _apply_transition(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transition between clips."""
        clip_a = params["clip_a"]
        clip_b = params["clip_b"]
        transition = params["transition"]
        duration = params["duration"]
        output = params["output"]
        
        # Get durations
        info_a = self._get_video_info({"input": clip_a})
        info_b = self._get_video_info({"input": clip_b})
        
        if not info_a["success"] or not info_b["success"]:
            return {"success": False, "error": "Could not get video info"}
        
        dur_a = info_a["output"]["duration"]
        dur_b = info_b["output"]["duration"]
        
        # Build filter based on transition type
        if transition == "crossfade":
            filter_complex = (
                f"[0:v][1:v]xfade=transition=fade:duration={duration}:"
                f"offset={dur_a - duration}[v];"
                f"[0:a][1:a]acrossfade=d={duration}[a]"
            )
        elif transition == "wipe":
            filter_complex = (
                f"[0:v][1:v]xfade=transition=wipeleft:duration={duration}:"
                f"offset={dur_a - duration}[v];"
                f"[0:a][1:a]acrossfade=d={duration}[a]"
            )
        elif transition == "slide":
            filter_complex = (
                f"[0:v][1:v]xfade=transition=slideleft:duration={duration}:"
                f"offset={dur_a - duration}[v];"
                f"[0:a][1:a]acrossfade=d={duration}[a]"
            )
        else:  # fade (default)
            filter_complex = (
                f"[0:v][1:v]xfade=transition=fade:duration={duration}:"
                f"offset={dur_a - duration}[v];"
                f"[0:a][1:a]acrossfade=d={duration}[a]"
            )
        
        result = self._run_ffmpeg([
            "-i", clip_a,
            "-i", clip_b,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-c:a", "aac",
            output
        ])
        
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        
        return {
            "success": True,
            "output": output,
            "transition": transition,
            "duration": dur_a + dur_b - duration
        }
    
    def _color_grade(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply color grading using LUTs or built-in presets.
        
        Supports:
        - Custom 3D LUT files (.cube, .3dl, .dat)
        - Built-in presets with professional color science
        - Intensity blending for subtle adjustments
        
        LUT files can be provided via:
        - lut_path parameter (absolute path)
        - LUT_DIR environment variable + preset name
        """
        input_path = params["input"]
        preset = params["preset"]
        intensity = params["intensity"]
        output = params["output"]
        lut_path = params.get("lut_path")
        
        # Check for LUT directory (mounted volume or local)
        lut_dir = str(get_settings().paths.lut_dir)
        
        # Extended presets with professional color grading
        # Each preset has: filter chain OR lut file name
        presets = {
            # === CLASSIC FILM LOOKS ===
            "cinematic": "colorbalance=rs=0.1:gs=-0.05:bs=0.1:rm=0.1:bm=0.05,curves=preset=cross_process",
            "teal_orange": "colorbalance=rs=-0.15:gs=-0.05:bs=0.2:rm=0.1:bm=-0.1:rh=0.15:bh=-0.15,eq=saturation=1.1:contrast=1.05",
            "blockbuster": "colorbalance=rs=-0.1:bs=0.15:rh=0.12:bh=-0.1,curves=m='0/0 0.25/0.22 0.5/0.5 0.75/0.78 1/1',eq=contrast=1.08",
            
            # === VINTAGE / RETRO ===
            "vintage": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,curves=preset=vintage",
            "film_fade": "curves=m='0/0.05 0.5/0.5 1/0.95',colorbalance=rs=0.1:gs=0.05:bs=-0.05,eq=saturation=0.85",
            "70s": "colortemperature=temperature=5500,colorbalance=rs=0.15:gs=0.1:rh=0.1,eq=saturation=1.2:contrast=0.95",
            "polaroid": "curves=r='0/0.1 0.5/0.55 1/0.9':g='0/0.05 0.5/0.5 1/0.95':b='0/0 0.5/0.45 1/0.85',eq=saturation=0.9",
            
            # === TEMPERATURE ===
            "cold": "colortemperature=temperature=7500,colorbalance=bs=0.1:bm=0.05,eq=saturation=0.9",
            "warm": "colortemperature=temperature=4000,colorbalance=rs=0.08:rh=0.05,eq=saturation=1.1",
            "golden_hour": "colortemperature=temperature=3500,colorbalance=rs=0.12:gs=0.05:rh=0.1:gh=0.05,eq=saturation=1.15:brightness=0.03",
            "blue_hour": "colortemperature=temperature=8000,colorbalance=bs=0.15:bm=0.08,eq=saturation=0.95:contrast=1.05",
            
            # === MOOD / GENRE ===
            "noir": "hue=s=0,curves=preset=darker,eq=contrast=1.2",
            "horror": "colorbalance=gs=-0.1:bs=0.05:bm=0.1,curves=m='0/0 0.4/0.35 0.6/0.65 1/1',eq=saturation=0.7:contrast=1.15",
            "sci_fi": "colorbalance=bs=0.2:bm=0.1:gs=-0.05,eq=saturation=0.85:contrast=1.1,curves=m='0/0 0.3/0.25 1/1'",
            "dreamy": "gblur=sigma=0.5,colorbalance=rs=0.05:bs=0.05,eq=saturation=0.9:brightness=0.05,curves=m='0/0.05 0.5/0.55 1/1'",
            
            # === PROFESSIONAL ===
            "vivid": "eq=saturation=1.4:contrast=1.1,unsharp=5:5:1.0",
            "muted": "eq=saturation=0.7:contrast=0.95,curves=m='0/0.05 0.5/0.5 1/0.95'",
            "high_contrast": "curves=m='0/0 0.25/0.15 0.5/0.5 0.75/0.85 1/1',eq=contrast=1.15",
            "low_contrast": "curves=m='0/0.1 0.5/0.5 1/0.9',eq=contrast=0.9",
            "desaturated": "eq=saturation=0.5",
            "punch": "eq=saturation=1.25:contrast=1.1,unsharp=3:3:0.8,curves=m='0/0 0.2/0.15 0.8/0.85 1/1'"
        }
        
        # LUT file mappings (if LUT files are available in LUT_DIR)
        # These map preset names to common LUT filenames
        lut_files = {
            "cinematic_lut": "cinematic.cube",
            "teal_orange_lut": "teal_orange.cube",
            "film_emulation": "kodak_2383.cube",
            "log_to_rec709": "log_to_rec709.cube",
            "bleach_bypass": "bleach_bypass.cube"
        }
        
        filter_str = None
        
        # Priority 1: Explicit LUT path
        if lut_path and os.path.exists(lut_path):
            filter_str = f"lut3d={lut_path}"
        
        # Priority 2: Check if preset maps to a LUT file
        elif preset in lut_files:
            lut_file = os.path.join(lut_dir, lut_files[preset])
            if os.path.exists(lut_file):
                filter_str = f"lut3d={lut_file}"
            else:
                return {"success": False, "error": f"LUT file not found: {lut_file}. Mount LUTs to {lut_dir}"}
        
        # Priority 3: Built-in filter presets
        elif preset in presets:
            filter_str = presets[preset]
        
        # Priority 4: Check if preset is a direct LUT filename in LUT_DIR
        elif os.path.exists(os.path.join(lut_dir, f"{preset}.cube")):
            filter_str = f"lut3d={os.path.join(lut_dir, preset)}.cube"
        
        else:
            available = list(presets.keys()) + list(lut_files.keys())
            return {"success": False, "error": f"Unknown preset: {preset}. Available: {available}"}
        
        # Apply intensity by mixing with original
        if intensity < 1.0:
            # Blend graded with original based on intensity
            filter_str = f"split[a][b];[a]{filter_str}[graded];[b][graded]blend=all_expr='A*(1-{intensity})+B*{intensity}'"
        
        result = self._run_ffmpeg([
            "-i", input_path,
            "-vf", filter_str,
            "-c:a", "copy",
            output
        ])
        
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        
        return {
            "success": True,
            "output": output,
            "preset": preset,
            "intensity": intensity,
            "used_lut": lut_path is not None or preset in lut_files
        }
    
    def _mix_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mix or replace audio."""
        video = params["video"]
        audio = params["audio"]
        mode = params["mode"]
        video_vol = params.get("video_volume", 1.0)
        audio_vol = params.get("audio_volume", 1.0)
        output = params["output"]
        
        if mode == "replace":
            # Replace video audio entirely
            result = self._run_ffmpeg([
                "-i", video,
                "-i", audio,
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output
            ])
        elif mode == "mix":
            # Mix both audio tracks
            filter_complex = (
                f"[0:a]volume={video_vol}[va];"
                f"[1:a]volume={audio_vol}[aa];"
                f"[va][aa]amix=inputs=2:duration=first[aout]"
            )
            result = self._run_ffmpeg([
                "-i", video,
                "-i", audio,
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                output
            ])
        elif mode == "ducking":
            # Duck video audio when new audio is present
            filter_complex = (
                f"[1:a]asplit=2[sc][mix];"
                f"[0:a][sc]sidechaincompress=threshold=0.03:ratio=4:attack=200:release=1000[ducked];"
                f"[ducked]volume={video_vol}[va];"
                f"[mix]volume={audio_vol}[aa];"
                f"[va][aa]amix=inputs=2:duration=first[aout]"
            )
            result = self._run_ffmpeg([
                "-i", video,
                "-i", audio,
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                output
            ])
        else:
            return {"success": False, "error": f"Unknown mode: {mode}"}
        
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        
        return {
            "success": True,
            "output": output,
            "mode": mode
        }
    
    def _resize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resize video."""
        input_path = params["input"]
        width = params["width"]
        height = params["height"]
        output = params["output"]
        algorithm = params.get("algorithm", "lanczos")
        
        scale_filter = f"scale={width}:{height}:flags={algorithm}"
        
        result = self._run_ffmpeg([
            "-i", input_path,
            "-vf", scale_filter,
            "-c:a", "copy",
            output
        ])
        
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        
        return {
            "success": True,
            "output": output,
            "width": width,
            "height": height
        }
    
    def _concatenate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Concatenate video clips."""
        clips = params["clips"]
        output = params["output"]
        reencode = params.get("reencode", False)
        
        # Create concat file
        concat_file = self.temp_dir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")
        
        if reencode:
            # Re-encode for compatibility
            result = self._run_ffmpeg([
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264",
                "-c:a", "aac",
                output
            ])
        else:
            # Stream copy (fast)
            result = self._run_ffmpeg([
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                output
            ])
        
        # Cleanup
        concat_file.unlink()
        
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        
        return {
            "success": True,
            "output": output,
            "clips_count": len(clips)
        }
    
    def _get_video_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get video information using ffprobe."""
        input_path = params["input"]
        
        result = self._run_ffprobe([
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            input_path
        ])
        
        if result.returncode != 0:
            return {"success": False, "error": "Could not probe video"}
        
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"success": False, "error": "Invalid probe output"}
        
        # Extract key info
        video_stream = None
        audio_stream = None
        
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and not video_stream:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and not audio_stream:
                audio_stream = stream
        
        info = {
            "duration": float(data.get("format", {}).get("duration", 0)),
            "size_bytes": int(data.get("format", {}).get("size", 0)),
            "format": data.get("format", {}).get("format_name", "unknown")
        }
        
        if video_stream:
            info.update({
                "width": video_stream.get("width"),
                "height": video_stream.get("height"),
                "video_codec": video_stream.get("codec_name"),
                "framerate": eval(video_stream.get("r_frame_rate", "0/1"))
            })
        
        if audio_stream:
            info.update({
                "audio_codec": audio_stream.get("codec_name"),
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "channels": audio_stream.get("channels")
            })
        
        return {
            "success": True,
            "output": info
        }
    
    def _thumbnail_grid(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate thumbnail grid."""
        input_path = params["input"]
        columns = params["columns"]
        rows = params["rows"]
        output = params["output"]
        
        total_thumbs = columns * rows
        
        # Get video duration
        info = self._get_video_info({"input": input_path})
        if not info["success"]:
            return info
        
        duration = info["output"]["duration"]
        interval = duration / (total_thumbs + 1)
        
        # Generate thumbnail grid using tile filter
        result = self._run_ffmpeg([
            "-i", input_path,
            "-vf", f"fps=1/{interval},scale=320:-1,tile={columns}x{rows}",
            "-frames:v", "1",
            "-q:v", "2",
            output
        ])
        
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        
        return {
            "success": True,
            "output": output,
            "grid": f"{columns}x{rows}",
            "thumbnails": total_thumbs
        }
    
    def _speed_change(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Change video speed."""
        input_path = params["input"]
        speed = params["speed"]
        output = params["output"]
        preserve_pitch = params.get("preserve_audio_pitch", True)
        
        # Video speed: setpts filter (inverse of speed)
        video_filter = f"setpts={1/speed}*PTS"
        
        # Audio speed: atempo filter (must be between 0.5 and 2.0)
        # Chain multiple atempo filters for extreme speeds
        if preserve_pitch:
            if speed > 2.0:
                # Chain atempo filters
                atempo_chain = []
                remaining = speed
                while remaining > 2.0:
                    atempo_chain.append("atempo=2.0")
                    remaining /= 2.0
                atempo_chain.append(f"atempo={remaining}")
                audio_filter = ",".join(atempo_chain)
            elif speed < 0.5:
                atempo_chain = []
                remaining = speed
                while remaining < 0.5:
                    atempo_chain.append("atempo=0.5")
                    remaining /= 0.5
                atempo_chain.append(f"atempo={remaining}")
                audio_filter = ",".join(atempo_chain)
            else:
                audio_filter = f"atempo={speed}"
        else:
            # Just resample (pitch will change)
            audio_filter = f"asetrate=44100*{speed},aresample=44100"
        
        result = self._run_ffmpeg([
            "-i", input_path,
            "-vf", video_filter,
            "-af", audio_filter,
            "-c:v", "libx264",
            "-c:a", "aac",
            output
        ])
        
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        
        return {
            "success": True,
            "output": output,
            "speed": speed
        }


# Convenience function for direct tool execution
def execute_ffmpeg_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an FFmpeg tool by name."""
    toolkit = FFmpegToolkit()
    return toolkit.execute(tool_name, params)


# Export tools schema for LLM integration
def get_ffmpeg_tools_schema() -> List[Dict[str, Any]]:
    """Get OpenAI-compatible function schema for all FFmpeg tools."""
    toolkit = FFmpegToolkit()
    return toolkit.get_tools_schema()


# ---------------------------------------------------------------------------
# Compatibility helpers for legacy tests
# ---------------------------------------------------------------------------
def _color_grade(preset: str):
    """
    Lightweight helper for tests: return the filter chain for a preset.

    The production pipeline uses FFmpegToolkit._color_grade with real files.
    This shim avoids filesystem/FFmpeg calls during unit tests by exposing the
    preset mapping only.
    """
    # Mirror preset keys used inside FFmpegToolkit._color_grade
    presets = {
        # === CLASSIC FILM LOOKS ===
        "cinematic": "colorbalance=rs=0.1:gs=-0.05:bs=0.1:rm=0.1:bm=0.05,curves=preset=cross_process",
        "teal_orange": "colorbalance=rs=-0.15:gs=-0.05:bs=0.2:rm=0.1:bm=-0.1:rh=0.15:bh=-0.15,eq=saturation=1.1:contrast=1.05",
        "blockbuster": "colorbalance=rs=-0.1:bs=0.15:rh=0.12:bh=-0.1,curves=m='0/0 0.25/0.22 0.5/0.5 0.75/0.78 1/1',eq=contrast=1.08",

        # === VINTAGE / RETRO ===
        "vintage": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,curves=preset=vintage",
        "film_fade": "curves=m='0/0.05 0.5/0.5 1/0.95',colorbalance=rs=0.1:gs=0.05:bs=-0.05,eq=saturation=0.85",
        "70s": "colortemperature=temperature=5500,colorbalance=rs=0.15:gs=0.1:rh=0.1,eq=saturation=1.2:contrast=0.95",
        "polaroid": "curves=r='0/0.1 0.5/0.55 1/0.9':g='0/0.05 0.5/0.5 1/0.95':b='0/0 0.5/0.45 1/0.85',eq=saturation=0.9",

        # === TEMPERATURE ===
        "cold": "colortemperature=temperature=7500,colorbalance=bs=0.1:bm=0.05,eq=saturation=0.9",
        "warm": "colortemperature=temperature=4000,colorbalance=rs=0.08:rh=0.05,eq=saturation=1.1",
        "golden_hour": "colortemperature=temperature=3500,colorbalance=rs=0.12:gs=0.05:rh=0.1:gh=0.05,eq=saturation=1.15:brightness=0.03",
        "blue_hour": "colortemperature=temperature=8000,colorbalance=bs=0.15:bm=0.08,eq=saturation=0.95:contrast=1.05",

        # === MOOD / GENRE ===
        "noir": "hue=s=0,curves=preset=darker,eq=contrast=1.2",
        "horror": "colorbalance=gs=-0.1:bs=0.05:bm=0.1,curves=m='0/0 0.4/0.35 0.6/0.65 1/1',eq=saturation=0.7:contrast=1.15",
        "sci_fi": "colorbalance=bs=0.2:bm=0.1:gs=-0.05,eq=saturation=0.85:contrast=1.1,curves=m='0/0 0.3/0.25 1/1'",
        "dreamy": "gblur=sigma=0.5,colorbalance=rs=0.05:bs=0.05,eq=saturation=0.9:brightness=0.05,curves=m='0/0.05 0.5/0.55 1/1'",

        # === PROFESSIONAL ===
        "vivid": "eq=saturation=1.4:contrast=1.1,unsharp=5:5:1.0",
        "muted": "eq=saturation=0.7:contrast=0.95,curves=m='0/0.05 0.5/0.5 1/0.95'",
        "high_contrast": "curves=m='0/0 0.25/0.15 0.5/0.5 0.75/0.85 1/1',eq=contrast=1.15",
        "low_contrast": "curves=m='0/0.1 0.5/0.5 1/0.9',eq=contrast=0.9",
        "desaturated": "eq=saturation=0.5",
        "punch": "eq=saturation=1.25:contrast=1.1,unsharp=3:3:0.8,curves=m='0/0 0.2/0.15 0.8/0.85 1/1'"
    }

    return presets.get(preset)
