"""
Montage AI - AI Post-Production Assistant

Core:
    from montage_ai import create_montage
    create_montage(variant_id=1)

B-Roll Planning:
    from montage_ai.broll_planner import plan_broll
    suggestions = plan_broll("The athlete trains. Victory moment.")

VideoAgent (Memory-Augmented Analysis):
    from montage_ai.video_agent import create_video_agent
    agent = create_video_agent()
    agent.analyze_video("/path/to/video.mp4")
    clips = agent.caption_retrieval("energetic action scene")

Cloud GPU (Upscaling, Transcription):
    from montage_ai.cgpu_upscaler import upscale_with_cgpu
    from montage_ai.transcriber import Transcriber

Cloud GPU Jobs (Unified Pipeline):
    from montage_ai.cgpu_jobs import CGPUJobManager, TranscribeJob, UpscaleJob, StabilizeJob
    manager = CGPUJobManager()
    manager.submit(StabilizeJob("video.mp4"))
    manager.process_queue()

Timeline Export:
    from montage_ai.timeline_exporter import TimelineExporter
"""

__version__ = "0.4.0"

def create_montage(*args, **kwargs):
    """
    Lazy-loaded entry point for creating a montage.
    Reduces package import time by ~6s.
    """
    from .editor import create_montage as _create_montage
    return _create_montage(*args, **kwargs)

from .exceptions import (
    MontageError,
    VideoAnalysisError,
    RenderError,
    FFmpegError,
    LLMError,
    CGPUError,
    TimelineError,
    ConfigurationError,
)
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

# cgpu Utilities (shared module)
try:
    from .cgpu_utils import (
        CGPUConfig,
        is_cgpu_available,
        check_cgpu_gpu,
        run_cgpu_command,
        cgpu_copy_to_remote,
        get_cgpu_llm_client,
    )
    CGPU_UTILS_AVAILABLE = True
except ImportError:
    CGPU_UTILS_AVAILABLE = False

# cgpu Upscaler
def upscale_with_cgpu(*args, **kwargs):
    from .cgpu_upscaler import upscale_with_cgpu as _upscale
    return _upscale(*args, **kwargs)

def upscale_image_with_cgpu(*args, **kwargs):
    from .cgpu_upscaler import upscale_image_with_cgpu as _upscale
    return _upscale(*args, **kwargs)

CGPU_UPSCALER_AVAILABLE = True

# Transcriber
def Transcriber(*args, **kwargs):
    from .transcriber import Transcriber as _Transcriber
    return _Transcriber(*args, **kwargs)

TRANSCRIBER_AVAILABLE = True

# B-Roll Planner
def plan_broll(*args, **kwargs):
    from .broll_planner import plan_broll as _plan_broll
    return _plan_broll(*args, **kwargs)

def format_plan(*args, **kwargs):
    from .broll_planner import format_plan as _format_plan
    return _format_plan(*args, **kwargs)

BROLL_PLANNER_AVAILABLE = True # Assume available, will error on call if missing

# StateFlow Director (Deterministic Multi-State Pipeline)
# ... keep available as is for now since it's just classes mostly
try:
    from .stateflow_director import (
        StateFlowDirector,
        DirectorState,
        StateContext,
        IntentAnalysis,
        create_director_output,
    )
    STATEFLOW_AVAILABLE = True
except ImportError:
    STATEFLOW_AVAILABLE = False

# Story Arc CSP Solver (Constraint Satisfaction)
try:
    from .story_arc_csp import (
        StoryArcCSPSolver,
        StoryPhase,
        PhaseConstraints,
        solve_story_arc,
    )
    STORY_ARC_CSP_AVAILABLE = True
except ImportError:
    STORY_ARC_CSP_AVAILABLE = False

__all__ = [
    # Core
    "create_montage",
    # Exceptions
    "MontageError",
    "VideoAnalysisError",
    "RenderError",
    "FFmpegError",
    "LLMError",
    "CGPUError",
    "TimelineError",
    "ConfigurationError",
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
    # cgpu Utilities
    "CGPUConfig",
    "is_cgpu_available",
    "check_cgpu_gpu",
    "run_cgpu_command",
    "cgpu_copy_to_remote",
    "get_cgpu_llm_client",
    # Upscaling
    "upscale_with_cgpu",
    "upscale_image_with_cgpu",
    # Transcriber
    "Transcriber",
    # B-Roll Planner
    "plan_broll",
    "format_plan",
    # StateFlow Director
    "StateFlowDirector",
    "DirectorState",
    "StateContext",
    "IntentAnalysis",
    "create_director_output",
    # Story Arc CSP
    "StoryArcCSPSolver",
    "StoryPhase",
    "PhaseConstraints",
    "solve_story_arc",
    # Feature flags
    "FFMPEG_TOOLS_AVAILABLE",
    "VIDEO_AGENT_AVAILABLE",
    "CGPU_UTILS_AVAILABLE",
    "CGPU_UPSCALER_AVAILABLE",
    "TRANSCRIBER_AVAILABLE",
    "BROLL_PLANNER_AVAILABLE",
    "STATEFLOW_AVAILABLE",
    "STORY_ARC_CSP_AVAILABLE",
]
