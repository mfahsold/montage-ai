"""
Montage AI Web Interface

Simple Flask web UI for creating video montages.
DRY + KISS: Minimal dependencies, no over-engineering.
"""

import os
import re
import json
import shutil
import subprocess
import threading
import psutil
import queue
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional
from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

from ..cgpu_utils import is_cgpu_available, check_cgpu_gpu
from ..core.hardware import get_best_hwaccel
from ..auto_reframe import AutoReframeEngine
from ..ffmpeg_utils import build_ffmpeg_cmd, build_ffprobe_cmd
from ..audio_analysis import remove_filler_words
from ..transcriber import Transcriber

# Centralized Configuration (Single Source of Truth)
from ..config import get_settings, reload_settings
from ..logger import logger
from .job_options import normalize_options, apply_preview_preset, apply_finalize_overrides

# Job phase tracking models
from .models import JobPhase, PIPELINE_PHASES

# Session Management (Centralized State)
from ..core.session import get_session_manager

# Timeline Export
from ..timeline_exporter import TimelineExporter, Timeline, Clip

# SSE Helper (Refactored to separate module)
from .sse import MessageAnnouncer, format_sse

announcer = MessageAnnouncer()

# B-Roll Planning (semantic clip search via video_agent)
try:
    from ..video_agent import create_video_agent
    VIDEO_AGENT_AVAILABLE = True
except ImportError:
    VIDEO_AGENT_AVAILABLE = False


def get_version() -> str:
    """Get version from git commit hash (short) or fallback to env/default."""
    git_commit = os.environ.get("GIT_COMMIT", "").strip()
    if git_commit:
        return git_commit[:8]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=2,
            cwd=Path(__file__).parent.parent.parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return "dev"


VERSION = get_version()

# =============================================================================
# CONFIGURATION (from centralized config module)
# =============================================================================
_settings = get_settings()

# Path aliases for backward compatibility
INPUT_DIR = _settings.paths.input_dir
MUSIC_DIR = _settings.paths.music_dir
OUTPUT_DIR = _settings.paths.output_dir
ASSETS_DIR = _settings.paths.assets_dir

# Note: Directories are created lazily on first job to avoid issues in test environments

# Default options derived from settings (Single Source of Truth)
DEFAULT_OPTIONS = {
    "enhance": _settings.features.enhance,
    "stabilize": _settings.features.stabilize,
    "upscale": _settings.features.upscale,
    "cgpu": _settings.llm.cgpu_gpu_enabled,
    "story_engine": _settings.features.story_engine,
    "llm_clip_selection": _settings.features.llm_clip_selection,
    "creative_loop": _settings.features.creative_loop,
    "export_timeline": _settings.features.export_timeline,
    "generate_proxies": _settings.features.generate_proxies,
    "preserve_aspect": _settings.features.preserve_aspect,
}

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload
app.config['UPLOAD_FOLDER'] = INPUT_DIR
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year cache for static files (Hardware Nah: minimize I/O)





# Job queue and management
from rq import Queue
from redis import Redis
from ..core.job_store import JobStore
from ..tasks import run_montage, run_transcript_render

# Redis connection
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_conn = Redis(host=redis_host, port=redis_port)
q = Queue(connection=redis_conn)
job_store = JobStore()

# jobs = {} # Removed
# job_lock = threading.Lock() # Removed
# job_queue = deque() # Removed
# active_jobs = 0 # Removed
MAX_CONCURRENT_JOBS = _settings.processing.max_concurrent_jobs
MIN_MEMORY_GB = 2  # Minimum memory required to start a job


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def bool_to_env(value: bool) -> str:
    """Convert boolean to env var string. DRY helper for options -> env vars."""
    return "true" if value else "false"





def check_memory_available(required_gb: float = MIN_MEMORY_GB) -> tuple[bool, float]:
    """Check if enough memory is available to start a job.

    Returns:
        (is_available, available_gb)
    """
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        return available_gb >= required_gb, available_gb
    except Exception as e:
        print(f"⚠️ Error checking memory: {e}")
        return True, 0.0  # Fail-open to avoid blocking


def get_job_status(job_id: str) -> dict:
    """Get status of a job."""
    job = job_store.get_job(job_id)
    return job if job else {"status": "not_found"}











# Register Blueprints
from .routes.session import session_bp
from .routes.analysis import analysis_bp

app.register_blueprint(session_bp)
app.register_blueprint(analysis_bp)

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Main landing page with workflow selection (modern voxel design)."""
    return render_template('index.html', version=VERSION)


@app.route('/montage')
def montage_creator():
    """Montage Creator - advanced editing interface."""
    return render_template('montage.html', version=VERSION)


@app.route('/shorts')
def shorts_studio():
    """Shorts Studio - vertical video creation."""
    return render_template('shorts.html', version=VERSION)


@app.route('/transcript')
def transcript_editor():
    """Transcript Editor - text-based video editing."""
    return render_template('transcript.html', version=VERSION)


@app.route('/api/status')
def api_status():
    """API health check with system stats."""
    mem = psutil.virtual_memory()
    memory_ok, available_gb = check_memory_available()

    # GPU/CGPU stats
    hw_config = get_best_hwaccel()
    encoder_status = hw_config.encoder
    if hw_config.is_gpu:
        encoder_status = f"{hw_config.type} ({encoder_status})"
    cgpu_ok = is_cgpu_available()

    # Get queue stats (RQ)
    try:
        active_jobs = len(q.started_job_registry)
        queued_jobs = len(q)
    except:
        active_jobs = 0
        queued_jobs = 0
    
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "system": {
            "memory_available_gb": round(available_gb, 2),
            "memory_total_gb": round(mem.total / (1024**3), 2),
            "memory_percent": mem.percent,
            "active_jobs": active_jobs,
            "queued_jobs": queued_jobs,
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            # New fields for UI
            "gpu_encoder": encoder_status,
            "encoder": hw_config.encoder,
            "cgpu_available": cgpu_ok,
            "version": VERSION
        },
        "defaults": {
            **DEFAULT_OPTIONS,
            "cgpu": DEFAULT_OPTIONS.get("cgpu", False) or cgpu_ok,  # Enable if available
        }
    })


@app.route('/api/analyze-crops', methods=['POST'])
def analyze_crops():
    """Analyze a video for smart reframing crops."""
    data = request.json
    filename = data.get('filename')
    keyframes_data = data.get('keyframes', [])
    
    if not filename:
        return jsonify({"error": "Filename required"}), 400
        
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
        
    try:
        # Use a smaller smoothing window for UI responsiveness if needed, 
        # but default is fine.
        reframer = AutoReframeEngine()
        
        # Convert keyframes dicts to Keyframe objects
        from ..auto_reframe import Keyframe
        keyframes = []
        if keyframes_data:
            for kf in keyframes_data:
                keyframes.append(Keyframe(
                    time=float(kf['time']),
                    center_x_norm=float(kf['x'])
                ))
        
        crops = reframer.analyze(filepath, keyframes=keyframes)
        
        # Convert dataclasses to dicts
        result = [
            {
                "time": c.time,
                "x": c.x,
                "y": c.y,
                "width": c.width,
                "height": c.height,
                "score": c.score
            }
            for c in crops
        ]
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing crops: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/transparency')
def api_transparency():
    """Return Responsible AI and transparency metadata for the UI."""
    settings = get_settings()
    llm = settings.llm

    return jsonify({
        "policy": {
            "data_handling": "Local by default; optional cgpu offload when enabled.",
            "training": "No model training on user footage.",
            "control": "Users choose features and can export editable timelines (OTIO/EDL).",
        },
        "explainability": {
            "decision_logs": "Available when EXPORT_DECISIONS=true (see /api/jobs/<id>/decisions).",
        },
        "llm_backends": {
            "openai_compatible": llm.has_openai_backend,
            "google_ai": llm.has_google_backend,
            "cgpu": llm.cgpu_enabled,
            "ollama": True,
        },
        "oss_stack": [
            {"name": "FFmpeg", "purpose": "Encoding/decoding"},
            {"name": "OpenCV", "purpose": "Visual analysis"},
            {"name": "librosa", "purpose": "Audio analysis"},
            {"name": "OpenTimelineIO", "purpose": "NLE export"},
            {"name": "Whisper", "purpose": "Transcription"},
            {"name": "Demucs", "purpose": "Voice isolation"},
            {"name": "Real-ESRGAN", "purpose": "Upscaling"},
        ],
        "scope": [
            "AI-assisted rough cuts from existing footage",
            "Beat sync and story arc pacing",
            "Professional NLE handoff via OTIO/EDL/XML",
        ],
        "out_of_scope": [
            "Generative text-to-video",
            "Full NLE timeline editing",
            "Social hosting platform",
        ],
    })


@app.route('/api/files', methods=['GET'])
def api_list_files():
    """List uploaded files."""
    videos = [f.name for f in INPUT_DIR.glob('*.mp4')] + [f.name for f in INPUT_DIR.glob('*.mov')]
    music = [f.name for f in MUSIC_DIR.glob('*.mp3')] + [f.name for f in MUSIC_DIR.glob('*.wav')]

    return jsonify({
        "videos": sorted(videos),
        "music": sorted(music),
        "video_count": len(videos),
        "music_count": len(music)
    })


@app.route('/api/transcript/<filename>', methods=['GET'])
def api_get_transcript(filename):
    """Get transcript JSON for a video file."""
    # Sanitize filename
    filename = secure_filename(filename)
    
    # Look for JSON file (assuming same name as video but .json)
    base_name = os.path.splitext(filename)[0]
    json_path = INPUT_DIR / f"{base_name}.json"
    
    if not json_path.exists():
        return jsonify({"error": "Transcript not found"}), 404
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/transcript/render', methods=['POST'])
def api_render_transcript_edit():
    """
    Render a preview based on transcript edits.
    
    Expects JSON:
    {
        "filename": "video.mp4",
        "edits": [
            {"index": 0, "removed": false},
            {"index": 1, "removed": true},
            ...
        ]
    }
    """
    data = request.json
    filename = secure_filename(data.get('filename'))
    edits = data.get('edits', [])
    
    if not filename:
        return jsonify({"error": "Missing filename"}), 400
        
    video_path = INPUT_DIR / filename
    transcript_path = INPUT_DIR / f"{os.path.splitext(filename)[0]}.json"
    
    if not video_path.exists() or not transcript_path.exists():
        return jsonify({"error": "File not found"}), 404

    # Create job ID
    job_id = f"transcript_edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Queue the job
    job_data = {
        "id": job_id,
        "type": "transcript_render",
        "video_path": str(video_path),
        "transcript_path": str(transcript_path),
        "edits": edits,
        "status": "queued",
        "created_at": datetime.now().isoformat()
    }
    
    # Store in Redis
    job_store.create_job(job_id, job_data)
    
    # Enqueue with RQ
    q.enqueue(run_transcript_render, job_id, str(video_path), str(transcript_path), edits)
        
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route('/api/transcript/detect-fillers', methods=['POST'])
def api_detect_fillers():
    """
    Detect filler words in a transcript.
    
    Returns indices of filler words to be removed.
    """
    data = request.json
    filename = secure_filename(data.get('filename'))
    
    if not filename:
        return jsonify({"error": "Missing filename"}), 400
        
    transcript_path = INPUT_DIR / f"{os.path.splitext(filename)[0]}.json"
    
    if not transcript_path.exists():
        return jsonify({"error": "Transcript not found"}), 404
        
    try:
        # Load transcript
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
            
        # Flatten words
        words = []
        if 'segments' in transcript_data:
            for seg in transcript_data['segments']:
                words.extend(seg.get('words', []))
        elif 'words' in transcript_data:
            words = transcript_data['words']
            
        # Detect fillers
        # Use the same list as TextEditor
        from ..text_editor import FILLER_WORDS
        
        filler_indices = []
        for i, word in enumerate(words):
            text = word.get('word', '').lower().strip(".,!?;:")
            if text in FILLER_WORDS:
                filler_indices.append(i)
                
        return jsonify({
            "success": True,
            "filler_indices": filler_indices,
            "count": len(filler_indices)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/video/<filename>', methods=['GET'])
def api_serve_video(filename):
    """Serve video file for preview."""
    filename = secure_filename(filename)
    file_path = INPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path)


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload video or music files."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_type = request.form.get('type', 'video')  # 'video' or 'music'

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Validate file type (from centralized config)
    video_extensions = _settings.file_types.video_extensions
    music_extensions = _settings.file_types.audio_extensions

    if file_type == 'video':
        if not allowed_file(file.filename, video_extensions):
            return jsonify({"error": f"Invalid video format. Allowed: {video_extensions}"}), 400
        target_dir = INPUT_DIR
    elif file_type == 'music':
        if not allowed_file(file.filename, music_extensions):
            return jsonify({"error": f"Invalid music format. Allowed: {music_extensions}"}), 400
        target_dir = MUSIC_DIR
    else:
        return jsonify({"error": "Invalid file type"}), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = target_dir / filename
    file.save(filepath)

    return jsonify({
        "success": True,
        "filename": filename,
        "size": filepath.stat().st_size
    })




@app.route('/api/jobs', methods=['POST'])
def api_create_job():
    """Create new montage job with queue management."""
    data = request.json

    # Validate required fields
    if 'style' not in data:
        return jsonify({"error": "Missing required field: style"}), 400

    # Generate job ID
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Handle Quick Preview override
    data = apply_preview_preset(data)

    # Normalize options (single source of truth for parsing/defaults/derivation)
    normalized_options = normalize_options(data, DEFAULT_OPTIONS, _settings)

    # Create job with normalized options and structured phase tracking
    job = {
        "id": job_id,
        "style": data['style'],
        "options": normalized_options,
        "status": "queued",
        "phase": JobPhase.initial().to_dict(),
        "created_at": datetime.now().isoformat()
    }

    # Store in Redis
    job_store.create_job(job_id, job)

    # Enqueue
    q.enqueue(run_montage, job_id, data['style'], job['options'])

    return jsonify(job)


@app.route('/api/stream')
def stream():
    """Server-Sent Events for real-time updates (Hardware Nah: Zero polling overhead)."""
    def event_stream():
        messages = announcer.listen()
        while True:
            msg = messages.get()  # blocks until a new message arrives
            yield msg
    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/api/jobs/<job_id>', methods=['GET'])
def api_get_job(job_id):
    """Get job status."""
    job = get_job_status(job_id)
    if job.get("status") == "not_found":
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route('/api/jobs/<job_id>/finalize', methods=['POST'])
def api_finalize_job(job_id):
    """Create a high-quality render from a preview job."""
    # 1. Get original job
    original_job = get_job_status(job_id)
    if original_job.get("status") == "not_found":
        return jsonify({"error": "Job not found"}), 404
        
    # 2. Create new job options based on original
    new_options = apply_finalize_overrides(original_job['options'])
    
    # 3. Create new job
    new_job_id = f"{job_id}_hq"
    style = original_job['style']
    
    job = {
        "id": new_job_id,
        "style": style,
        "options": new_options,
        "status": "queued",
        "phase": JobPhase.initial().to_dict(),
        "created_at": datetime.now().isoformat(),
        "parent_job_id": job_id
    }

    # Store in Redis
    job_store.create_job(new_job_id, job)

    # Enqueue
    q.enqueue(run_montage, new_job_id, style, new_options)

    return jsonify(job)









@app.route('/api/jobs', methods=['GET'])
def api_list_jobs():
    """List all jobs."""
    jobs_dict = job_store.list_jobs()
    job_list = list(jobs_dict.values())
    return jsonify({"jobs": sorted(job_list, key=lambda x: x['created_at'], reverse=True)})


@app.route('/api/download/<filename>', methods=['GET'])
def api_download(filename):
    """Download output file with proper MIME type."""
    # secure_filename sanitizes but preserves the base name
    safe_filename = secure_filename(filename)
    filepath = OUTPUT_DIR / safe_filename

    if not filepath.exists():
        # Debug: log what we're looking for
        print(f"⚠️ Download requested but file not found: {filepath}")
        print(f"   Available files: {list(OUTPUT_DIR.glob('*.mp4'))[:5]}")
        return jsonify({"error": f"File not found: {safe_filename}"}), 404

    # Explicit MIME type for video files
    mimetype = 'video/mp4' if safe_filename.endswith('.mp4') else None
    
    return send_file(
        filepath, 
        as_attachment=True,
        download_name=safe_filename,  # Explicit filename for download
        mimetype=mimetype
    )


@app.route('/api/styles', methods=['GET'])
def api_list_styles():
    """List available styles."""
    styles = [
        {"id": "dynamic", "name": "Dynamic", "description": "Position-aware pacing (default)"},
        {"id": "hitchcock", "name": "Hitchcock", "description": "Suspense - slow build, fast climax"},
        {"id": "mtv", "name": "MTV", "description": "Rapid 1-2 beat cuts"},
        {"id": "action", "name": "Action", "description": "Michael Bay fast cuts"},
        {"id": "documentary", "name": "Documentary", "description": "Natural, observational"},
        {"id": "minimalist", "name": "Minimalist", "description": "Contemplative long takes"},
        {"id": "wes_anderson", "name": "Wes Anderson", "description": "Symmetric, stylized"}
    ]
    return jsonify({"styles": styles})


@app.route('/api/jobs/<job_id>/logs', methods=['GET'])
def api_get_job_logs(job_id):
    """Get logs for a specific job."""
    # Look for log file
    log_file = OUTPUT_DIR / f"render_{job_id}.log"

    # Also try without job_id prefix (legacy format)
    if not log_file.exists():
        log_file = OUTPUT_DIR / "render.log"

    if not log_file.exists():
        return jsonify({"error": "Log file not found"}), 404

    # Read last N lines (default 500, max 2000)
    try:
        lines = int(request.args.get('lines', 500))
        lines = min(lines, 2000)  # Cap at 2000 lines

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return jsonify({
            "job_id": job_id,
            "log_file": str(log_file.name),
            "total_lines": len(all_lines),
            "returned_lines": len(last_lines),
            "logs": ''.join(last_lines)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to read log: {str(e)}"}), 500


@app.route('/api/jobs/<job_id>/decisions', methods=['GET'])
def api_get_job_decisions(job_id):
    """Get AI decisions/analysis for a specific job."""
    # Look for decisions JSON file (exported by monitoring.py)
    decisions_file = OUTPUT_DIR / f"decisions_{job_id}.json"

    if not decisions_file.exists():
        # Check for alternative naming
        decisions_file = OUTPUT_DIR / f"montage_{job_id}_decisions.json"

    if not decisions_file.exists():
        return jsonify({
            "job_id": job_id,
            "available": False,
            "message": "No decisions file found. Set EXPORT_DECISIONS=true in environment."
        })

    try:
        with open(decisions_file, 'r') as f:
            decisions_data = json.load(f)

        return jsonify({
            "job_id": job_id,
            "available": True,
            **decisions_data
        })
    except Exception as e:
        return jsonify({"error": f"Failed to read decisions: {str(e)}"}), 500


@app.route('/api/broll/suggest', methods=['POST'])
def api_suggest_broll():
    """Suggest B-roll clips based on script/keywords (semantic search).

    DRY: Reuses video_agent.caption_retrieval() for semantic matching.
    KISS: Simple query → results, no complex state.
    """
    if not VIDEO_AGENT_AVAILABLE:
        return jsonify({"error": "Video Agent not available"}), 500

    data = request.json
    query = data.get('query', '').strip()
    top_k = min(int(data.get('top_k', 5)), 20)  # Cap at 20

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        agent = create_video_agent()

        # Search for matching clips
        results = agent.caption_retrieval(query, top_k=top_k)

        return jsonify({
            "query": query,
            "suggestions": results,
            "count": len(results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/broll/analyze', methods=['POST'])
def api_analyze_footage():
    """Analyze footage and build searchable memory (run once per session).

    Processes all clips in INPUT_DIR and stores in video_agent memory.
    """
    if not VIDEO_AGENT_AVAILABLE:
        return jsonify({"error": "Video Agent not available"}), 500

    try:
        agent = create_video_agent()

        # Find all video files
        video_files = list(INPUT_DIR.glob('*.mp4')) + list(INPUT_DIR.glob('*.mov'))

        results = []
        for video_path in video_files:
            result = agent.analyze_video(str(video_path))
            results.append({
                "file": video_path.name,
                "success": result.get("success", False),
                "segments": result.get("segments_created", 0)
            })

        return jsonify({
            "analyzed": len(results),
            "results": results,
            "memory_stats": agent.get_memory_stats()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/jobs/<job_id>/creative-instructions', methods=['GET'])
def api_get_creative_instructions(job_id):
    """Get Creative Director instructions for a job."""
    job = get_job_status(job_id)

    if job.get("status") == "not_found":
        return jsonify({"error": "Job not found"}), 404

    # Extract creative prompt and style from job options
    options = job.get("options", {})

    return jsonify({
        "job_id": job_id,
        "creative_prompt": options.get("prompt", ""),
        "style": job.get("style", "dynamic"),
        "options": options,
        "status": job.get("status")
    })


# =============================================================================
# CGPU JOB ENDPOINTS (Cloud GPU Operations)
# =============================================================================

# Lazy import to avoid circular imports and startup delays
_cgpu_manager = None

def get_cgpu_manager():
    """Get CGPUJobManager singleton (lazy init)."""
    global _cgpu_manager
    if _cgpu_manager is None:
        try:
            from ..cgpu_jobs import CGPUJobManager
            _cgpu_manager = CGPUJobManager()
        except ImportError as e:
            print(f"⚠️ CGPUJobManager not available: {e}")
            return None
    return _cgpu_manager


@app.route('/api/cgpu/status', methods=['GET'])
def api_cgpu_status():
    """Check CGPU availability and status."""
    try:
        from ..cgpu_utils import is_cgpu_available, check_cgpu_gpu

        available = is_cgpu_available()
        gpu_ok, gpu_info = check_cgpu_gpu() if available else (False, "Not available")

        manager = get_cgpu_manager()
        stats = manager.stats() if manager else {}

        return jsonify({
            "available": available,
            "gpu": {"ok": gpu_ok, "info": gpu_info},
            "queue_size": stats.get("queue_size", 0),
            "completed": stats.get("completed_count", 0),
        })
    except Exception as e:
        return jsonify({"available": False, "error": str(e)}), 500


@app.route('/api/cgpu/transcribe', methods=['POST'])
def api_cgpu_transcribe():
    """Submit transcription job to CGPU.

    Request JSON:
        { "file": "audio.wav", "model": "medium", "format": "srt" }
    """
    data = request.json or {}

    filename = data.get('file')
    if not filename:
        return jsonify({"error": "Missing 'file' parameter"}), 400

    # Resolve file path
    filepath = INPUT_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        from ..cgpu_jobs import TranscribeJob

        job = TranscribeJob(
            audio_path=str(filepath),
            model=data.get('model', 'medium'),
            output_format=data.get('format', 'srt'),
            language=data.get('language'),
        )

        # Run in background thread
        def run_job():
            result = job.execute()
            print(f"   Transcription {'✅' if result.success else '❌'}: {result.output_path or result.error}")

        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()

        return jsonify({
            "job_id": job.job_id,
            "status": "submitted",
            "input": filename,
            "model": job.model,
            "format": job.output_format,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cgpu/upscale', methods=['POST'])
def api_cgpu_upscale():
    """Submit upscaling job to CGPU.

    Request JSON:
        { "file": "video.mp4", "scale": 4, "model": "realesr-animevideov3" }
    """
    data = request.json or {}

    filename = data.get('file')
    if not filename:
        return jsonify({"error": "Missing 'file' parameter"}), 400

    filepath = INPUT_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        from ..cgpu_jobs import UpscaleJob

        # Output to OUTPUT_DIR
        output_name = f"{filepath.stem}_upscaled{filepath.suffix}"
        output_path = OUTPUT_DIR / output_name

        job = UpscaleJob(
            input_path=str(filepath),
            output_path=str(output_path),
            scale=int(data.get('scale', 4)),
            model=data.get('model', 'realesr-animevideov3'),
        )

        def run_job():
            result = job.execute()
            print(f"   Upscale {'✅' if result.success else '❌'}: {result.output_path or result.error}")

        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()

        return jsonify({
            "job_id": job.job_id,
            "status": "submitted",
            "input": filename,
            "output": output_name,
            "scale": job.scale,
            "model": job.model,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cgpu/stabilize', methods=['POST'])
def api_cgpu_stabilize():
    """Submit stabilization job to CGPU.

    Request JSON:
        { "file": "video.mp4", "smoothing": 10, "shakiness": 5 }
    """
    data = request.json or {}

    filename = data.get('file')
    if not filename:
        return jsonify({"error": "Missing 'file' parameter"}), 400

    filepath = INPUT_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        from ..cgpu_jobs import StabilizeJob

        output_name = f"{filepath.stem}_stabilized{filepath.suffix}"
        output_path = OUTPUT_DIR / output_name

        job = StabilizeJob(
            video_path=str(filepath),
            output_path=str(output_path),
            smoothing=int(data.get('smoothing', 10)),
            shakiness=int(data.get('shakiness', 5)),
        )

        def run_job():
            result = job.execute()
            print(f"   Stabilize {'✅' if result.success else '❌'}: {result.output_path or result.error}")

        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()

        return jsonify({
            "job_id": job.job_id,
            "status": "submitted",
            "input": filename,
            "output": output_name,
            "smoothing": job.smoothing,
            "shakiness": job.shakiness,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cgpu/jobs', methods=['GET'])
def api_cgpu_jobs():
    """List CGPU job queue and history."""
    manager = get_cgpu_manager()
    if not manager:
        return jsonify({"error": "CGPUJobManager not available"}), 503

    return jsonify(manager.stats())


# =============================================================================
# SESSION API (Centralized State Management)
# =============================================================================

@app.route('/api/session/create', methods=['POST'])
def api_session_create():
    """Create a new editing session (Transcript or Shorts)."""
    data = request.json or {}
    session_type = data.get('type', 'generic')
    
    Session = get_session_manager()
    session = Session.create(session_type=session_type)
    
    return jsonify(session.__dict__)

@app.route('/api/session/<session_id>', methods=['GET'])
def api_session_get(session_id):
    """Get session state."""
    Session = get_session_manager()
    session = Session.load(session_id)
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
        
    return jsonify(session.__dict__)

@app.route('/api/session/<session_id>/asset', methods=['POST'])
def api_session_add_asset(session_id):
    """Add an asset to a session (Unified Upload)."""
    Session = get_session_manager()
    session = Session.load(session_id)
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
        
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    asset_type = request.form.get('type', 'video')
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    # Determine target directory based on asset type
    if asset_type == 'audio':
        target_dir = MUSIC_DIR
    else:
        target_dir = INPUT_DIR
        
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    filepath = target_dir / unique_filename
    
    file.save(filepath)
    
    # Register asset in session
    asset = session.add_asset(str(filepath), asset_type)
    
    return jsonify({
        "success": True,
        "asset": asset.__dict__,
        "session": session.__dict__
    })

@app.route('/api/session/<session_id>/analyze', methods=['POST'])
def api_session_analyze(session_id):
    """Run analysis on session assets (Crops, Transcription, etc.)."""
    Session = get_session_manager()
    session = Session.load(session_id)
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
        
    data = request.json or {}
    analysis_type = data.get('type', 'crops')
    
    try:
        main_video = session.get_main_video()
        if not main_video:
            return jsonify({"error": "No video asset in session"}), 400
            
        if analysis_type == 'crops':
            # Run AutoReframeEngine
            from ..auto_reframe import AutoReframeEngine, Keyframe
            
            # Check if we have cached results (skip if keyframes provided, as that implies re-calc)
            if 'crops_auto' in session.state and not data.get('force', False) and not data.get('keyframes'):
                return jsonify({"success": True, "crops": session.state['crops_auto']})
            
            # Initialize engine
            reframer = AutoReframeEngine(target_aspect=9/16)
            
            # Parse keyframes if present
            keyframes = []
            if 'keyframes' in data:
                for kf in data['keyframes']:
                    keyframes.append(Keyframe(time=float(kf['time']), center_x_norm=float(kf['x'])))
            
            # Run analysis (Note: This is blocking. For large files, use background job)
            # For the prototype, we'll allow it to block for short clips or assume async client handling.
            # Ideally, we should submit a job and return job_id.
            # But to keep it simple as requested:
            
            # We need to pass the absolute path
            crop_data = reframer.analyze(main_video.path, keyframes=keyframes)
            
            # Convert CropWindow objects to dicts for JSON serialization
            crops_json = [
                {
                    "time": c.time,
                    "x": c.x,
                    "y": c.y,
                    "width": c.width,
                    "height": c.height,
                    "score": c.score
                }
                for c in crop_data
            ]
            
            # Update session state
            session.update_state('crops_auto', crops_json)
            
            return jsonify({"success": True, "crops": crops_json})
            
        elif analysis_type == 'transcription':
            # Check cache
            if 'transcript' in session.state and not data.get('force', False):
                 return jsonify({"success": True, "transcript": session.state['transcript']})

            # Run transcription
            transcriber = Transcriber()
            if not transcriber.is_available():
                 return jsonify({"error": "Transcription service unavailable"}), 503

            # We want JSON output to parse it
            output_path = transcriber.transcribe(main_video.path, output_format="json", word_timestamps=True)
            
            if not output_path or not os.path.exists(output_path):
                return jsonify({"error": "Transcription failed"}), 500
                
            with open(output_path, 'r') as f:
                transcript_data = json.load(f)
                
            session.update_state('transcript', transcript_data)
            return jsonify({"success": True, "transcript": transcript_data})
            
        return jsonify({"error": "Unknown analysis type"}), 400
        
    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({"error": str(e)}), 500

@app.route('/api/session/<session_id>/remove_fillers', methods=['POST'])
def api_session_remove_fillers(session_id):
    """Remove filler words from the transcript."""
    Session = get_session_manager()
    session = Session.load(session_id)
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
        
    if 'transcript' not in session.state:
        return jsonify({"error": "No transcript found. Run analysis first."}), 400
        
    transcript = session.state['transcript']
    
    # Identify fillers
    indices_to_remove = remove_filler_words(transcript)
    
    if not indices_to_remove:
        return jsonify({"success": True, "count": 0, "message": "No filler words found"})
        
    # Update edits in session state
    current_edits = session.state.get('edits', [])
    
    # Add new edits (avoid duplicates)
    existing_indices = {e['index'] for e in current_edits if e.get('removed')}
    new_edits = []
    
    for idx in indices_to_remove:
        if idx not in existing_indices:
            new_edits.append({"index": idx, "removed": True})
            
    updated_edits = current_edits + new_edits
    session.update_state('edits', updated_edits)
    
    return jsonify({
        "success": True, 
        "count": len(new_edits), 
        "indices": indices_to_remove,
        "edits": updated_edits
    })

@app.route('/api/session/<session_id>/state', methods=['POST'])
def api_session_update_state(session_id):
    """Update session state (e.g. edits, cuts)."""
    Session = get_session_manager()
    session = Session.load(session_id)
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
        
    data = request.json or {}
    for key, value in data.items():
        session.update_state(key, value)
        
    return jsonify({"success": True, "session": session.__dict__})

# Preview Generator
from ..preview_generator import PreviewGenerator

@app.route('/api/session/<session_id>/render_preview', methods=['POST'])
def api_session_render_preview(session_id):
    """
    Generate a preview for the session (Phone Rig or Transcript Loop).
    
    Centralized preview logic that delegates to specific engines based on session type.
    """
    Session = get_session_manager()
    session = Session.load(session_id)
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
        
    data = request.json or {}
    preview_type = data.get('type', 'frame') # 'frame' or 'clip'
    timestamp = data.get('timestamp', 0)
    
    try:
        main_video = session.get_main_video()
        if not main_video:
            return jsonify({"error": "No video asset in session"}), 400
            
        generator = PreviewGenerator(output_dir=str(OUTPUT_DIR))
        output_filename = f"preview_{session.id}_{int(time.time())}.mp4"
        
        if session.type == 'shorts':
            # Shorts Preview (Phone Rig)
            # Apply crop at timestamp
            
            # Get crop data for this time
            crops = session.state.get('crops_auto', [])
            # Find closest crop
            crop = next((c for c in crops if c['time'] >= timestamp), crops[-1] if crops else None)
            
            if not crop:
                # Default center crop
                crop = {"x": 0.5, "y": 0.5, "width": 9/16, "height": 1.0} # Normalized
            
            if preview_type == 'clip':
                # Check if we have dynamic crops (keyframes)
                keyframes = session.state.get('crops_auto', [])
                
                # If no keyframes, run analysis on the fly
                if not keyframes:
                    logger.info(f"No cached crops found for session {session.id}. Running auto-reframe analysis...")
                    try:
                        from ..auto_reframe import AutoReframeEngine
                        reframer = AutoReframeEngine(target_aspect=9/16)
                        # Run analysis (blocking)
                        crop_data = reframer.analyze(main_video.path)
                        
                        # Convert to JSON-serializable format
                        crops_json = [
                            {
                                "time": c.time,
                                "x": c.x,
                                "y": c.y,
                                "width": c.width,
                                "height": c.height,
                                "score": c.score
                            }
                            for c in crop_data
                        ]
                        
                        # Update session state
                        session.update_state('crops_auto', crops_json)
                        keyframes = crops_json
                        
                        # Update the single frame crop as well since we now have data
                        crop = next((c for c in keyframes if c['time'] >= timestamp), keyframes[-1] if keyframes else None)
                        
                    except Exception as e:
                        logger.error(f"Auto-reframe analysis failed during preview: {e}")
                        # Fallback to center crop is handled by generate_shorts_preview if keyframes is None/Empty
                        pass

                output_path = generator.generate_shorts_preview(
                    main_video.path, 
                    crop, 
                    output_filename,
                    keyframes=keyframes if keyframes else None
                )
                return jsonify({
                    "success": True,
                    "url": f"/downloads/{output_filename}",
                    "crop": crop
                })
            else:
                # Frame preview (just return crop data for frontend to render overlay)
                return jsonify({
                    "success": True,
                    "url": f"/api/video/{main_video.filename}", 
                    "crop": crop
                })

        elif session.type == 'transcript':
            # Transcript Preview (Cut List)
            timeline = _build_timeline_from_session(session, main_video)
            if not timeline:
                 return jsonify({"error": "Could not build timeline"}), 400
            
            # Convert timeline clips to segments (start, end)
            segments = [(clip.start_time, clip.start_time + clip.duration) for clip in timeline.clips]
            
            output_path = generator.generate_transcript_preview(
                main_video.path,
                segments,
                output_filename
            )
            
            return jsonify({
                "success": True,
                "url": f"/downloads/{output_filename}"
            })
            
        else:
            return jsonify({"error": "Unknown session type"}), 400

    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# TRANSCRIPT API ENDPOINTS (Text-Based Editing)
# =============================================================================

# Create upload directory for transcript videos
TRANSCRIPT_UPLOAD_DIR = Path(os.environ.get("TRANSCRIPT_DIR", "/tmp/montage_transcript"))
TRANSCRIPT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)





@app.route('/api/transcript/upload', methods=['POST'])
def api_transcript_upload():
    """Upload video file for transcript editing."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename, {'mp4', 'mov', 'avi', 'mkv', 'webm', 'mp3', 'wav'}):
        return jsonify({"error": "Invalid file type"}), 400
    
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    filepath = TRANSCRIPT_UPLOAD_DIR / unique_filename
    
    file.save(filepath)
    
    return jsonify({
        "success": True,
        "path": str(filepath),
        "filename": unique_filename
    })





@app.route('/api/transcript/video/<filename>')
def api_transcript_video(filename):
    """Serve uploaded video for preview."""
    filepath = TRANSCRIPT_UPLOAD_DIR / secure_filename(filename)
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath)


@app.route('/api/transcript/transcribe', methods=['POST'])
def api_transcript_transcribe():
    """Transcribe video using Whisper (local or CGPU)."""
    data = request.json or {}
    video_path_str = data.get('video_path')
    session_id = data.get('session_id')
    
    # Session-based workflow
    session = None
    if session_id:
        Session = get_session_manager()
        session = Session.load(session_id)
        if session:
            # Check if already transcribed
            if 'transcript' in session.state:
                return jsonify({"success": True, "transcript": session.state['transcript'], "cached": True})
            
            # If no video path provided, try to find main video in session
            if not video_path_str:
                main_video = session.get_main_video()
                if main_video:
                    video_path_str = main_video.path

    if not video_path_str:
        return jsonify({"error": "No video path provided"}), 400

    # Resolve path
    video_path = Path(video_path_str)
    if not video_path.exists():
        # Try relative to INPUT_DIR
        video_path = INPUT_DIR / video_path_str
    
    if not video_path.exists():
        return jsonify({"error": f"Video file not found: {video_path_str}"}), 404
    
    # Define output JSON path (same name as video)
    json_output_path = video_path.with_suffix('.json')
    
    try:
        transcript = None
        
        # Try CGPU first, fall back to local
        cgpu_available = is_cgpu_available()
        
        if cgpu_available:
            # Use CGPU transcription
            from ..cgpu_jobs import TranscribeJob
            job = TranscribeJob(
                audio_path=str(video_path),
                model=data.get('model', 'medium'),
                output_format='json',  # Word-level timestamps
                language=data.get('language'),
                word_timestamps=True,
            )
            result = job.execute()
            
            if result.success and result.output_path:
                with open(result.output_path, 'r') as f:
                    transcript = json.load(f)
            else:
                raise Exception(result.error or "Transcription failed")
        else:
            # Local transcription via whisper
            from ..transcriber import transcribe_audio
            
            # Extract audio first
            audio_path = video_path.with_suffix('.wav')
            cmd = build_ffmpeg_cmd([
                '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                str(audio_path)
            ])
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Transcribe
            transcript = transcribe_audio(str(audio_path), model='base', word_timestamps=True)
            
            # Clean up temp audio
            audio_path.unlink(missing_ok=True)
            
        # Save to local JSON path for persistence
        if transcript:
            with open(json_output_path, 'w') as f:
                json.dump(transcript, f, indent=2)
                
            # Update session state if active
            if session:
                session.update_state('transcript', transcript)

            return jsonify({"success": True, "transcript": transcript})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/transcript/export', methods=['POST'])
def api_transcript_export():
    """Export edited transcript as video, EDL, or OTIO."""
    data = request.json or {}
    
    # Support both workflows: 
    # 1. filename + edits (preferred, loads from disk)
    # 2. video_path + transcript (legacy/direct)
    
    filename = data.get('filename')
    edits = data.get('edits')
    
    video_path = None
    editor = None
    
    try:
        from ..text_editor import TextEditor, Segment, Word
        
        if filename and edits is not None:
            # Workflow 1: Load from disk and apply edits
            filename = secure_filename(filename)
            video_path = INPUT_DIR / filename
            transcript_path = INPUT_DIR / f"{os.path.splitext(filename)[0]}.json"
            
            if not video_path.exists() or not transcript_path.exists():
                return jsonify({"error": "File not found"}), 404
                
            editor = TextEditor(str(video_path))
            editor.load_transcript(str(transcript_path))
            
            # Apply edits (indices)
            removed_indices = [e['index'] for e in edits if e.get('removed')]
            
            current_idx = 0
            for seg in editor.segments:
                for word in seg.words:
                    if current_idx in removed_indices:
                        word.removed = True
                    current_idx += 1
                    
        else:
            # Workflow 2: Full transcript payload
            video_path_str = data.get('video_path')
            transcript = data.get('transcript')
            
            if not video_path_str or not transcript:
                return jsonify({"error": "Missing filename+edits OR video_path+transcript"}), 400
            
            video_path = Path(video_path_str)
            if not video_path.exists():
                return jsonify({"error": "Video file not found"}), 404
                
            editor = TextEditor(str(video_path))
            
            # Convert frontend transcript format to TextEditor segments
            segments = []
            for idx, seg_data in enumerate(transcript.get('segments', [])):
                words = []
                for w in seg_data.get('words', []):
                    word = Word(
                        text=w.get('word', w.get('text', '')).strip(),
                        start=w.get('start', 0),
                        end=w.get('end', 0),
                        confidence=w.get('probability', w.get('confidence', 1.0)),
                        removed=w.get('removed', False)
                    )
                    if word.text:
                        words.append(word)
                
                segment = Segment(
                    id=seg_data.get('id', idx),
                    start=seg_data.get('start', 0),
                    end=seg_data.get('end', 0),
                    text=seg_data.get('text', ''),
                    words=words
                )
                segments.append(segment)
            
            editor.segments = segments
        
        # Perform Export
        export_format = data.get('format', 'video')
        output_dir = OUTPUT_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == 'video':
            # Use TextEditor's export method with smooth audio crossfades
            output_path = output_dir / f"transcript_edit_{timestamp}.mp4"
            editor.export(str(output_path))
            
            return jsonify({
                "success": True,
                "filename": output_path.name,
                "path": str(output_path),
                "stats": editor.get_stats()
            })
            
        elif export_format == 'edl':
            output_path = output_dir / f"transcript_edit_{timestamp}.edl"
            editor.export_edl(str(output_path))
            return jsonify({
                "success": True,
                "filename": output_path.name,
                "path": str(output_path)
            })
            
        elif export_format == 'otio':
            output_path = output_dir / f"transcript_edit_{timestamp}.otio"
            try:
                editor.export_otio(str(output_path))
                return jsonify({
                    "success": True,
                    "filename": output_path.name,
                    "path": str(output_path)
                })
            except RuntimeError as e:
                return jsonify({"success": False, "error": f"OTIO export failed: {str(e)}"}), 500
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
            
        else:
            return jsonify({"error": f"Unknown format: {export_format}"}), 400

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# SHORTS STUDIO API ENDPOINTS
# =============================================================================

# Create upload directory for shorts videos
SHORTS_UPLOAD_DIR = Path(os.environ.get("SHORTS_DIR", "/tmp/montage_shorts"))
SHORTS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.route('/api/shorts/upload', methods=['POST'])
def api_shorts_upload():
    """Upload video file for shorts processing."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename, {'mp4', 'mov', 'avi', 'mkv', 'webm'}):
        return jsonify({"error": "Invalid file type"}), 400
    
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    filepath = SHORTS_UPLOAD_DIR / unique_filename
    
    file.save(filepath)
    
    return jsonify({
        "success": True,
        "path": str(filepath),
        "filename": unique_filename
    })


@app.route('/api/shorts/analyze', methods=['POST'])
def api_shorts_analyze():
    """Analyze video for smart reframing (face detection, subject tracking)."""
    data = request.json or {}
    video_path = data.get('video_path')
    reframe_mode = data.get('reframe_mode', 'auto')  # auto, speaker, center, custom
    session_id = data.get('session_id')
    
    # Session-based workflow
    session = None
    if session_id:
        Session = get_session_manager()
        session = Session.load(session_id)
        if session:
            # Check cache
            cache_key = f"crops_{reframe_mode}"
            if cache_key in session.state:
                return jsonify({
                    "success": True,
                    "mode": reframe_mode,
                    "crop_data": session.state[cache_key],
                    "cached": True
                })
            
            if not video_path:
                main_video = session.get_main_video()
                if main_video:
                    video_path = main_video.path

    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video file not found"}), 404
    
    try:
        from ..auto_reframe import AutoReframeEngine as SmartReframer
        
        reframer = SmartReframer(target_aspect=9/16)
        
        if reframe_mode == 'center':
            # Simple center crop, no analysis needed
            return jsonify({
                "success": True,
                "mode": "center",
                "crop_data": None,
                "message": "Center crop mode - no analysis needed"
            })
        
        # Analyze video for face/subject tracking
        crop_data = reframer.analyze(video_path)
        
        # Convert to JSON-serializable format
        crops_json = [
            {
                "time": c.time,
                "x": c.x,
                "y": c.y,
                "width": c.width,
                "height": c.height,
                "score": c.score
            }
            for c in crop_data
        ]
        
        # Save to session
        if session:
            session.update_state(f"crops_{reframe_mode}", crops_json)
        
        return jsonify({
            "success": True,
            "mode": reframe_mode,
            "crop_data": crops_json,
            "frame_count": len(crops_json)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/shorts/visualize', methods=['POST'])
def shorts_visualize():
    """Analyze video for smart reframing and return crop data."""
    try:
        data = request.json
        video_path = data.get('video_path')
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video not found'}), 404
            
        from ..auto_reframe import AutoReframeEngine as SmartReframer
        from dataclasses import asdict
        import cv2
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             return jsonify({'error': 'Could not open video'}), 500
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Initialize reframer
        reframer = SmartReframer(target_aspect=9/16)
        
        # Analyze
        crop_windows = reframer.analyze(video_path)
        
        # Convert to JSON-serializable format
        results = [asdict(cw) for cw in crop_windows]
        
        # Downsample for frontend performance if too many frames
        # Return max 500 points to keep payload light
        step = max(1, len(results) // 500)
        results = results[::step]
            
        return jsonify({
            'success': True,
            'crops': results,
            'original_width': width,
            'original_height': height
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _apply_clean_audio(video_path: str) -> str:
    """
    Apply clean audio processing: voice isolation + noise reduction.

    Uses SNR check to determine appropriate processing:
    - SNR >= 40dB: Excellent audio, skip processing
    - SNR 25-40dB: Good audio, light noise reduction only
    - SNR 15-25dB: Fair audio, noise reduction
    - SNR < 15dB: Poor audio, voice isolation + noise reduction

    Returns the path to a video with cleaned audio, or original path on failure.
    """
    try:
        from ..cgpu_jobs import VoiceIsolationJob, NoiseReductionJob
        from ..cgpu_utils import is_cgpu_available
        from ..audio_analysis import estimate_audio_snr

        # Step 0: Check SNR to determine processing needs
        try:
            quality = estimate_audio_snr(video_path)
            snr_db = quality.snr_db
            logger.info(f"Clean audio: SNR = {snr_db:.1f}dB ({quality.quality_tier})")

            # Excellent audio - skip processing
            if snr_db >= 40:
                logger.info("Clean audio: Excellent quality, skipping processing")
                return video_path
        except Exception as e:
            logger.warning(f"SNR estimation failed: {e}, proceeding with full cleanup")
            snr_db = 0  # Assume needs cleanup

        if not is_cgpu_available():
            logger.warning("Clean audio: CGPU not available, using FFmpeg fallback")
            return _apply_ffmpeg_noise_reduction(video_path)

        input_path = Path(video_path)
        timestamp_polish = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Step 1: Voice isolation (only for poor quality audio < 25dB SNR)
        vocals_path = None
        if snr_db < 25:
            try:
                job = VoiceIsolationJob(
                    audio_path=str(input_path),
                    model="htdemucs",
                    two_stems=True,
                    keep_all_stems=True
                )
                result = job.execute()
                if result.success and result.metadata.get("stems", {}).get("vocals"):
                    vocals_path = result.metadata["stems"]["vocals"]
                    logger.info("Clean audio: Voice isolation successful")
            except Exception as e:
                logger.warning(f"Voice isolation failed: {e}")
        else:
            logger.info("Clean audio: Good SNR, skipping voice isolation")

        # Step 2: Noise reduction (adaptive strength based on SNR)
        audio_to_clean = vocals_path or str(input_path)
        cleaned_audio = None

        # Determine noise reduction strength based on SNR
        if snr_db >= 35:
            attenuation = 50  # Light cleanup
        elif snr_db >= 25:
            attenuation = 75  # Moderate cleanup
        else:
            attenuation = 100  # Full cleanup

        try:
            noise_job = NoiseReductionJob(audio_path=audio_to_clean, attenuation_limit=attenuation)
            noise_result = noise_job.execute()
            if noise_result.success and noise_result.output_path:
                cleaned_audio = noise_result.output_path
                logger.info(f"Clean audio: Noise reduction successful (strength: {attenuation}%)")
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")

        # Use best available cleaned audio
        final_audio = cleaned_audio or vocals_path
        if not final_audio:
            logger.warning("Clean audio: all processing failed, using original")
            return video_path

        # Step 3: Replace audio in video
        polished_video_path = OUTPUT_DIR / f"shorts_cleaned_{timestamp_polish}.mp4"
        cmd = build_ffmpeg_cmd([
            "-i", video_path,
            "-i", final_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            str(polished_video_path)
        ])
        subprocess.run(cmd, check=True, capture_output=True)
        return str(polished_video_path)

    except Exception as e:
        logger.error(f"Clean audio pipeline failed: {e}")
        return video_path


def _apply_ffmpeg_noise_reduction(video_path: str) -> str:
    """FFmpeg fallback for noise reduction when CGPU is not available."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"shorts_denoised_{timestamp}.mp4"

        # Adaptive FFT-based denoiser + EQ + light compression
        noise_filter = (
            "afftdn=nf=-25:nr=10:nt=w"
            ",highpass=f=80"
            ",lowpass=f=14000"
            ",compand=attacks=0.3:decays=0.8:points=-80/-900|-45/-15|-27/-9|0/-7:soft-knee=6:gain=3"
        )

        cmd = build_ffmpeg_cmd([
            "-i", video_path,
            "-c:v", "copy",
            "-af", noise_filter,
            "-c:a", "aac",
            str(output_path)
        ])
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)

    except Exception as e:
        logger.warning(f"FFmpeg noise reduction failed: {e}")
        return video_path


@app.route('/api/shorts/render', methods=['POST'])
def api_shorts_render():
    """Render vertical video with smart reframing and captions."""
    data = request.json or {}
    video_path = data.get('video_path')
    reframe_mode = data.get('reframeMode', data.get('reframe_mode', 'auto'))
    caption_style = data.get('captionStyle', data.get('caption_style', 'tiktok'))
    add_captions = data.get('autoCaptions', data.get('add_captions', True))
    platform = data.get('platform', 'tiktok')
    duration = int(data.get('duration', 60))
    # Support both old (audioPolish) and new (cleanAudio) parameter names
    clean_audio = data.get('cleanAudio', data.get('audioPolish', False))

    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video file not found"}), 404

    try:
        # Apply Clean Audio if requested (voice isolation + noise reduction)
        if clean_audio:
            video_path = _apply_clean_audio(video_path)

        from ..auto_reframe import AutoReframeEngine as SmartReframer
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reframed_path = OUTPUT_DIR / f"shorts_reframed_{timestamp}.mp4"
        output_path = OUTPUT_DIR / f"shorts_{timestamp}.mp4"
        
        reframer = SmartReframer(target_aspect=9/16)
        
        # Analyze if needed
        if reframe_mode != 'center':
            crop_data = reframer.analyze(video_path)
        else:
            crop_data = None
        
        # Apply reframing
        reframer.apply(crop_data, video_path, str(reframed_path))
        
        # Add captions if requested
        if add_captions:
            from ..transcriber import transcribe_audio
            from ..caption_burner import CaptionBurner, CaptionStyle
            
            # Map frontend caption style to CaptionStyle enum
            style_map = {
                'default': CaptionStyle.TIKTOK,
                'tiktok': CaptionStyle.TIKTOK,
                'bold': CaptionStyle.BOLD,
                'minimal': CaptionStyle.MINIMAL,
                'gradient': CaptionStyle.KARAOKE,  # Gradient uses karaoke highlighting
                'karaoke': CaptionStyle.KARAOKE,
            }
            burner_style = style_map.get(caption_style, CaptionStyle.TIKTOK)
            
            # Transcribe video
            transcript = transcribe_audio(str(reframed_path), model='base', word_timestamps=True)
            
            # Generate SRT file
            srt_path = reframed_path.with_suffix('.srt')
            _generate_srt_from_transcript(transcript, str(srt_path))
            
            # Burn captions into video
            burner = CaptionBurner(style=burner_style)
            burner.burn(str(reframed_path), str(srt_path), str(output_path))
            
            # Clean up intermediate file
            reframed_path.unlink(missing_ok=True)
            srt_path.unlink(missing_ok=True)
        else:
            # No captions - just rename
            reframed_path.rename(output_path)
        
        return jsonify({
            "success": True,
            "filename": output_path.name,
            "path": str(output_path),
            "mode": reframe_mode,
            "platform": platform,
            "caption_style": caption_style if add_captions else None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def _generate_srt_from_transcript(transcript: dict, srt_path: str):
    """Generate SRT file from Whisper transcript."""
    segments = transcript.get('segments', [])
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()
            
            # Format timecodes: HH:MM:SS,mmm
            def tc(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                ms = int((seconds - int(seconds)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
            f.write(f"{i}\n")
            f.write(f"{tc(start)} --> {tc(end)}\n")
            f.write(f"{text}\n\n")


@app.route('/api/shorts/highlights', methods=['POST'])
def api_shorts_highlights():
    """
    Detect highlight moments for clip extraction.
    
    Multi-signal highlight detection:
    1. Audio Energy: High-energy regions (music drops, loud moments)
    2. Beat Alignment: Key beats in music
    3. Speech Phrases: Important speech segments (hooks)
    """
    data = request.json or {}
    video_path = data.get('video_path')
    max_clips = int(data.get('max_clips', 5))
    min_duration = float(data.get('min_duration', 5.0))
    max_duration = float(data.get('max_duration', 60.0))
    include_speech = data.get('include_speech', True)
    
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video file not found"}), 404
    
    try:
        import numpy as np
        from ..audio_analysis import AudioAnalyzer
        
        analyzer = AudioAnalyzer(video_path)
        analyzer.analyze()
        
        highlights = []
        beats = analyzer.beat_times if hasattr(analyzer, 'beat_times') else []
        energy = analyzer.energy_curve if hasattr(analyzer, 'energy_curve') else []
        
        # 1. Energy-based highlights
        if len(energy) > 0:
            threshold_high = np.percentile(energy, 85)  # Top 15% energy
            threshold_low = np.percentile(energy, 70)   # Top 30% for context
            
            hop_time = getattr(analyzer, 'hop_length', 512) / getattr(analyzer, 'sr', 22050)
            
            in_highlight = False
            start_time = 0
            peak_energy = 0
            
            for i, e in enumerate(energy):
                time = i * hop_time
                
                if e > threshold_high and not in_highlight:
                    in_highlight = True
                    start_time = max(0, time - 0.5)  # Capture lead-in
                    peak_energy = e
                elif in_highlight:
                    peak_energy = max(peak_energy, e)
                    if e <= threshold_low:
                        # End of highlight
                        in_highlight = False
                        duration = time - start_time
                        if min_duration <= duration <= max_duration:
                            # Normalize score to 0-1
                            score = min(1.0, peak_energy / (np.max(energy) + 0.001))
                            highlights.append({
                                "time": start_time,
                                "start": start_time,
                                "end": time,
                                "duration": round(duration, 2),
                                "score": round(score, 3),
                                "type": "Energy",
                                "label": f"🔥 High Energy ({int(score*100)}%)"
                            })
        
        # 2. Beat-drop detection (sudden energy increases)
        if len(energy) > 20 and len(beats) > 0:
            energy_diff = np.diff(energy)
            beat_indices = [int(b * getattr(analyzer, 'sr', 22050) / getattr(analyzer, 'hop_length', 512)) for b in beats]
            
            for beat_idx, beat_time in zip(beat_indices[:20], beats[:20]):
                if 0 < beat_idx < len(energy_diff):
                    # Check if this beat has a significant energy increase
                    if energy_diff[beat_idx] > np.percentile(energy_diff, 90):
                        # This is a "drop" moment
                        if not any(abs(h['time'] - beat_time) < 3 for h in highlights):  # Not too close to existing
                            highlights.append({
                                "time": beat_time,
                                "start": max(0, beat_time - 1),
                                "end": beat_time + min_duration,
                                "duration": min_duration + 1,
                                "score": 0.85,
                                "type": "Drop",
                                "label": "💥 Beat Drop"
                            })
        
        # 3. Speech hook detection (first 30 seconds often contain hooks)
        if include_speech and len(highlights) < max_clips:
            try:
                # Quick transcription to find speech density
                # Hook = high word density in first 30s
                from ..transcriber import transcribe_audio
                transcript = transcribe_audio(video_path, model='tiny', word_timestamps=True)
                
                segments = transcript.get('segments', [])
                for seg in segments[:5]:  # First 5 segments only
                    start = seg.get('start', 0)
                    end = seg.get('end', start + 3)
                    duration = end - start
                    
                    # Look for segments with punchy delivery (short, dense)
                    words = seg.get('words', [])
                    word_density = len(words) / (duration + 0.001) if duration > 0 else 0
                    
                    if word_density > 2.0 and duration >= 2:  # >2 words/sec = energetic delivery
                        if not any(abs(h['time'] - start) < 3 for h in highlights):
                            highlights.append({
                                "time": start,
                                "start": start,
                                "end": end,
                                "duration": round(duration, 2),
                                "score": min(0.9, 0.5 + word_density * 0.15),
                                "type": "Speech",
                                "label": "🎤 Hook"
                            })
            except Exception:
                pass  # Speech analysis optional, don't fail the whole endpoint
        
        # 4. Fallback: evenly distributed beat-aligned moments
        if len(highlights) < max_clips and len(beats) > 4:
            video_duration = getattr(analyzer, 'duration', 60)
            interval = video_duration / (max_clips + 1)
            
            for i in range(1, max_clips - len(highlights) + 1):
                target_time = i * interval
                # Find nearest beat
                nearest_beat = min(beats, key=lambda b: abs(b - target_time)) if beats else target_time
                
                if not any(abs(h['time'] - nearest_beat) < 5 for h in highlights):
                    highlights.append({
                        "time": nearest_beat,
                        "start": nearest_beat,
                        "end": nearest_beat + min_duration,
                        "duration": min_duration,
                        "score": 0.6,
                        "type": "Beat",
                        "label": "🎵 Beat"
                    })
        
        # Sort by time for timeline display, then limit
        highlights.sort(key=lambda x: x['time'])
        
        # Take best clips by score if too many
        if len(highlights) > max_clips:
            highlights.sort(key=lambda x: x['score'], reverse=True)
            highlights = highlights[:max_clips]
            highlights.sort(key=lambda x: x['time'])  # Resort by time
        
        return jsonify({
            "success": True,
            "highlights": highlights,
            "total_found": len(highlights)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# API: Quality Profiles & Cloud Status
# =============================================================================

@app.route('/api/quality-profiles')
def api_quality_profiles():
    """Return available quality profiles and their settings."""
    from ..env_mapper import QUALITY_PROFILES
    
    return jsonify({
        "profiles": QUALITY_PROFILES,
        "default": "standard"
    })


@app.route('/api/cloud/status')
def api_cloud_status():
    """Check cloud acceleration availability."""
    cgpu_available = is_cgpu_available()
    cgpu_gpu = check_cgpu_gpu() if cgpu_available else False
    
    return jsonify({
        "available": cgpu_available,
        "gpu_available": cgpu_gpu,
        "features": {
            "transcription": cgpu_available,
            "upscaling": cgpu_gpu,
            "llm": cgpu_available
        },
        "fallback": "local"  # Always fall back to local if cloud unavailable
    })


# =============================================================================
# AUDIO POLISH API ENDPOINTS
# =============================================================================

@app.route('/api/audio/clean', methods=['POST'])
def api_audio_clean():
    """
    Clean Audio - one-click audio polish.
    
    Bundles voice isolation + noise reduction into a single "Clean Audio" toggle.
    Returns enhanced audio with background noise removed.
    
    Uses CGPU if available, falls back to local FFmpeg noise reduction.
    """
    data = request.json or {}
    audio_path = data.get('audio_path') or data.get('video_path')
    isolate_voice = data.get('isolate_voice', True)
    reduce_noise = data.get('reduce_noise', True)
    
    if not audio_path or not Path(audio_path).exists():
        return jsonify({"error": "Audio file not found"}), 404
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = Path(audio_path)
        output_path = OUTPUT_DIR / f"clean_audio_{timestamp}.wav"
        
        # Try CGPU voice isolation first
        cgpu_available = is_cgpu_available()
        voice_isolated = False
        
        if isolate_voice and cgpu_available:
            try:
                from ..cgpu_jobs import VoiceIsolationJob
                job = VoiceIsolationJob(
                    audio_path=str(input_path),
                    model="htdemucs",
                    two_stems=True  # Faster: just vocals vs accompaniment
                )
                result = job.execute()
                
                if result.success and result.output_path:
                    input_path = Path(result.output_path)
                    voice_isolated = True
            except Exception as e:
                logger.warning(f"Voice isolation failed, continuing with noise reduction: {e}")
        
        # Apply noise reduction via FFmpeg
        if reduce_noise or not voice_isolated:
            # FFmpeg noise reduction using afftdn (adaptive FFT-based denoiser)
            # This is a solid local fallback when CGPU isn't available
            noise_reduction_filter = (
                "afftdn=nf=-25:nr=10:nt=w"  # Adaptive denoiser
                ",highpass=f=80"  # Remove low rumble
                ",lowpass=f=14000"  # Remove high hiss
                ",compand=attacks=0.3:decays=0.8:points=-80/-900|-45/-15|-27/-9|0/-7:soft-knee=6:gain=3"  # Light compression
            )
            
            cmd = build_ffmpeg_cmd(
                [
                    '-i', str(input_path),
                    '-af', noise_reduction_filter,
                    '-acodec', 'pcm_s16le',  # High quality WAV
                    str(output_path)
                ],
                hide_banner=True,
                loglevel="error"
            )
            
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # Just copy the voice-isolated file
            import shutil
            shutil.copy(input_path, output_path)
        
        # Calculate actual SNR improvement
        from ..audio_analysis import estimate_audio_snr
        
        snr_before = estimate_audio_snr(str(Path(audio_path)))
        snr_after = estimate_audio_snr(str(output_path))
        snr_improvement_db = snr_after.snr_db - snr_before.snr_db
        
        return jsonify({
            "success": True,
            "filename": output_path.name,
            "path": str(output_path),
            "voice_isolated": voice_isolated,
            "noise_reduced": reduce_noise,
            "snr_before_db": round(snr_before.snr_db, 1),
            "snr_after_db": round(snr_after.snr_db, 1),
            "snr_improvement": f"+{snr_improvement_db:.1f}dB" if snr_improvement_db > 0 else f"{snr_improvement_db:.1f}dB",
            "quality_before": snr_before.quality_tier,
            "quality_after": snr_after.quality_tier,
            "method": "cgpu+ffmpeg" if voice_isolated else "ffmpeg"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/audio/analyze', methods=['POST'])
def api_audio_analyze():
    """
    Analyze audio quality and suggest improvements.
    
    Returns:
    - SNR estimate
    - Noise type detection (hiss, hum, background)
    - Recommended actions
    """
    data = request.json or {}
    audio_path = data.get('audio_path') or data.get('video_path')
    
    if not audio_path or not Path(audio_path).exists():
        return jsonify({"error": "Audio file not found"}), 404
    
    try:
        # Extract audio stats using FFmpeg
        cmd = build_ffprobe_cmd([
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            '-select_streams', 'a:0',
            str(audio_path)
        ])
        result = subprocess.run(cmd, capture_output=True, text=True)
        probe_data = json.loads(result.stdout) if result.stdout else {}
        
        # Get loudness stats
        loudness_cmd = build_ffmpeg_cmd(
            [
                '-hide_banner', '-i', str(audio_path),
                '-af', 'volumedetect', '-f', 'null', '/dev/null'
            ],
            overwrite=True
        )
        loudness_result = subprocess.run(loudness_cmd, capture_output=True, text=True, timeout=30)
        
        # Parse volume stats from stderr
        stderr = loudness_result.stderr
        mean_volume = -20.0
        max_volume = -10.0
        
        for line in stderr.split('\n'):
            if 'mean_volume' in line:
                try:
                    mean_volume = float(line.split(':')[1].strip().replace(' dB', ''))
                except:
                    pass
            if 'max_volume' in line:
                try:
                    max_volume = float(line.split(':')[1].strip().replace(' dB', ''))
                except:
                    pass
        
        # Use proper SNR estimation from audio_analysis module
        from ..audio_analysis import estimate_audio_snr
        audio_quality = estimate_audio_snr(str(audio_path))
        
        # Build recommendations based on actual quality
        recommendations = []
        
        if audio_quality.mean_volume_db < -30:
            recommendations.append("Audio is very quiet - consider normalizing")
        elif audio_quality.max_volume_db > -1:
            recommendations.append("Audio may be clipping - check peaks")
        
        # Dynamic range check
        dynamic_range = audio_quality.max_volume_db - audio_quality.mean_volume_db
        if dynamic_range > 20:
            recommendations.append("High dynamic range - consider compression for social media")
        
        # SNR-based recommendations
        if audio_quality.quality_tier in ("poor", "unusable"):
            recommendations.append(f"Low SNR ({audio_quality.snr_db:.1f}dB) - audio cleaning recommended")
        elif audio_quality.quality_tier == "acceptable":
            recommendations.append(f"Moderate SNR ({audio_quality.snr_db:.1f}dB) - audio cleaning may help")
        
        # Check if CGPU available for voice isolation
        if is_cgpu_available():
            recommendations.append("Voice isolation available via cloud acceleration")
        
        response_data = {
            "success": True,
            "analysis": {
                "snr_db": round(audio_quality.snr_db, 1),
                "mean_volume": round(audio_quality.mean_volume_db, 1),
                "max_volume": round(audio_quality.max_volume_db, 1),
                "dynamic_range": round(dynamic_range, 1),
                "quality_tier": audio_quality.quality_tier,
                "is_usable": audio_quality.is_usable,
                "sample_rate": probe_data.get('streams', [{}])[0].get('sample_rate', 'unknown'),
                "channels": probe_data.get('streams', [{}])[0].get('channels', 1),
            },
            "recommendations": recommendations,
            "clean_audio_available": True
        }
        
        # If timeline data requested (energy curve + beats)
        include_timeline = data.get('include_timeline', True)  # Default true for backwards compat
        if include_timeline:
            try:
                from ..audio_analysis import get_beat_times, analyze_music_energy
                from ..video_metadata import probe_duration
                
                duration = probe_duration(str(audio_path))
                
                # Get beats
                beat_info = get_beat_times(str(audio_path), verbose=False)
                beats_list = beat_info.beat_times.tolist() if hasattr(beat_info.beat_times, 'tolist') else list(beat_info.beat_times)
                
                # Get energy curve (downsample for frontend performance)
                energy_profile = analyze_music_energy(str(audio_path), verbose=False)
                energy_list = energy_profile.rms.tolist() if hasattr(energy_profile.rms, 'tolist') else list(energy_profile.rms)
                
                # Downsample energy to max 500 points
                if len(energy_list) > 500:
                    step = len(energy_list) // 500
                    energy_list = energy_list[::step]
                
                response_data["energy"] = energy_list
                response_data["beats"] = beats_list[:100]  # Cap at 100 beats
                response_data["duration"] = duration
                response_data["tempo"] = beat_info.tempo
                
            except Exception as timeline_err:
                logger.warning(f"Timeline data extraction failed: {timeline_err}")
                # Don't fail the whole request, just skip timeline data
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/shorts/create', methods=['POST'])
def api_shorts_create():
    """Create a short video - convenience alias for render endpoint."""
    # This is the endpoint the frontend calls
    return api_shorts_render()


def _build_timeline_from_session(session, main_video) -> Optional[Timeline]:
    """Convert session state into a Timeline object."""
    
    # 1. Get duration of source
    duration = main_video.metadata.get('duration')
    if not duration:
        # Fallback: try to get from transcript if available
        if 'transcript' in session.state:
             try:
                 # Last segment end
                 segs = session.state['transcript'].get('segments', [])
                 if segs:
                    duration = segs[-1]['end']
                 else:
                    words = session.state['transcript'].get('words', [])
                    if words:
                        duration = words[-1]['end']
             except:
                 pass
    
    # 2. Determine kept segments
    kept_segments = [] # list of (start, end) tuples
    
    if 'edits' in session.state and 'transcript' in session.state:
        # Reconstruct from transcript + edits
        transcript = session.state['transcript']
        edits = session.state['edits'] # list of {index: int, removed: bool}
        removed_indices = {e['index'] for e in edits if e.get('removed')}
        
        words = []
        if 'segments' in transcript:
            for seg in transcript['segments']:
                if 'words' in seg:
                    words.extend(seg['words'])
        elif 'words' in transcript:
            words = transcript['words']
            
        if not words:
            return None
            
        # Iterate words and build segments
        current_start = None
        current_end = None
        
        for i, word in enumerate(words):
            is_kept = (i not in removed_indices)
            
            if is_kept:
                if current_start is None:
                    current_start = word['start']
                current_end = word['end']
            else:
                if current_start is not None:
                    # Close segment
                    kept_segments.append((current_start, current_end))
                    current_start = None
                    current_end = None
                    
        # Close final segment
        if current_start is not None:
            kept_segments.append((current_start, current_end))
            
    else:
        # Default: keep whole video if duration known
        if duration:
            kept_segments.append((0.0, duration))
        else:
            return None

    # 3. Create Clips
    clips = []
    timeline_time = 0.0
    for start, end in kept_segments:
        dur = end - start
        if dur <= 0: continue
        
        clip = Clip(
            source_path=main_video.path,
            start_time=start,
            duration=dur,
            timeline_start=timeline_time
        )
        clips.append(clip)
        timeline_time += dur
        
    return Timeline(
        clips=clips,
        audio_path=main_video.path, # Assuming embedded audio
        total_duration=timeline_time,
        project_name=f"session_{session.id}"
    )


@app.route('/api/session/<session_id>/export', methods=['POST'])
def api_session_export(session_id):
    """Export the session to EDL/OTIO/Video."""
    Session = get_session_manager()
    session = Session.load(session_id)
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
        
    data = request.json or {}
    export_format = data.get('format', 'edl') # edl, otio, video
    
    main_video = session.get_main_video()
    if not main_video:
        return jsonify({"error": "No video asset in session"}), 400
        
    # Build Timeline
    timeline = _build_timeline_from_session(session, main_video)
    if not timeline:
         return jsonify({"error": "Could not build timeline from session state (missing duration or transcript)"}), 400

    exporter = TimelineExporter(output_dir=str(OUTPUT_DIR))
    
    try:
        output_path = None
        if export_format == 'edl':
            output_path = exporter._export_edl(timeline)
        elif export_format == 'otio':
            if not exporter.OTIO_AVAILABLE:
                 return jsonify({"error": "OpenTimelineIO not installed"}), 501
            output_path = exporter._export_otio(timeline)
        elif export_format == 'video':
             # TODO: Implement video render
             return jsonify({"error": "Video export not yet implemented via this API"}), 501
        else:
            return jsonify({"error": f"Unknown format: {export_format}"}), 400
            
        # Return relative path or download URL
        filename = os.path.basename(output_path)
        return jsonify({
            "success": True,
            "path": output_path,
            "url": f"/downloads/{filename}"
        })
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/downloads/<path:filename>')
def download_file(filename):
    """Serve exported files."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route('/health')
def health_check():
    """Health check endpoint for Kubernetes probes."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "montage-ai-web"
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Starting Montage AI Web UI on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
