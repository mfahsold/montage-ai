"""
Montage AI - AI-Powered Video Montage Creation

Core:
    from montage_ai import create_montage
    create_montage(variant_id=1)

Styles:
    from montage_ai.style_templates import (
        get_style_template,
        list_available_styles,
        load_style_templates,
        reload_style_templates,
    )

FFmpeg Tools (LLM-callable):
    from montage_ai.ffmpeg_tools import FFmpegToolkit, execute_ffmpeg_tool
    
    toolkit = FFmpegToolkit()
    result = toolkit.execute("create_segment", {...})

VideoAgent (Memory-Augmented Analysis):
    from montage_ai.video_agent import VideoAgentAdapter, create_video_agent
    
    agent = create_video_agent()
    agent.analyze_video("/path/to/video.mp4")
    clips = agent.caption_retrieval("energetic action scene")

Video Generation (cgpu):
    from montage_ai.wan_vace import WanVACEService, WanBRollGenerator
    from montage_ai.open_sora import OpenSoraGenerator
    
    # B-Roll generation
    generator = WanBRollGenerator()
    video = generator.generate_transition("city", "nature", duration=2.0)

Cloud GPU Upscaling:
    from montage_ai.cgpu_upscaler import upscale_with_cgpu
    
    upscale_with_cgpu("/input.mp4", "/output.mp4", scale=4)

Experimental:
    from montage_ai.timeline_exporter import TimelineExporter  # WIP
    from montage_ai.footage_analyzer import DeepFootageAnalyzer  # WIP
"""

__version__ = "0.4.0"

from .editor import create_montage
from .style_templates import (
    get_style_template,
    list_available_styles,
    load_style_templates,
    reload_style_templates,
)

# FFmpeg Tools
try:
    from .ffmpeg_tools import FFmpegToolkit, execute_ffmpeg_tool, get_ffmpeg_tools_schema
    FFMPEG_TOOLS_AVAILABLE = True
except ImportError:
    FFMPEG_TOOLS_AVAILABLE = False

# VideoAgent
try:
    from .video_agent import VideoAgentAdapter, create_video_agent
    VIDEO_AGENT_AVAILABLE = True
except ImportError:
    VIDEO_AGENT_AVAILABLE = False

# Video Generation (cgpu)
try:
    from .wan_vace import WanVACEService, WanBRollGenerator
    WAN_VACE_AVAILABLE = True
except ImportError:
    WAN_VACE_AVAILABLE = False

try:
    from .open_sora import OpenSoraGenerator, create_open_sora
    OPEN_SORA_AVAILABLE = True
except ImportError:
    OPEN_SORA_AVAILABLE = False

# cgpu Upscaler
try:
    from .cgpu_upscaler import upscale_with_cgpu, upscale_image_with_cgpu
    CGPU_UPSCALER_AVAILABLE = True
except ImportError:
    CGPU_UPSCALER_AVAILABLE = False

__all__ = [
    # Core
    "create_montage",
    # Styles
    "get_style_template",
    "list_available_styles",
    "load_style_templates",
    "reload_style_templates",
    # FFmpeg Tools
    "FFmpegToolkit",
    "execute_ffmpeg_tool",
    "get_ffmpeg_tools_schema",
    # VideoAgent
    "VideoAgentAdapter",
    "create_video_agent",
    # Video Generation
    "WanVACEService",
    "WanBRollGenerator",
    "OpenSoraGenerator",
    "create_open_sora",
    # Upscaling
    "upscale_with_cgpu",
    "upscale_image_with_cgpu",
    # Feature flags
    "FFMPEG_TOOLS_AVAILABLE",
    "VIDEO_AGENT_AVAILABLE",
    "WAN_VACE_AVAILABLE",
    "OPEN_SORA_AVAILABLE",
    "CGPU_UPSCALER_AVAILABLE",
]
