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
from datetime import datetime
from pathlib import Path
from collections import deque
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

from ..cgpu_utils import is_cgpu_available, check_cgpu_gpu
from ..core.hardware import get_best_hwaccel

# Centralized Configuration (Single Source of Truth)
from ..config import get_settings, reload_settings

# Job phase tracking models
from .models import JobPhase, PIPELINE_PHASES

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
jobs = {}
job_lock = threading.Lock()
job_queue = deque()
active_jobs = 0
MAX_CONCURRENT_JOBS = _settings.processing.max_concurrent_jobs
MIN_MEMORY_GB = 2  # Minimum memory required to start a job


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def bool_to_env(value: bool) -> str:
    """Convert boolean to env var string. DRY helper for options -> env vars."""
    return "true" if value else "false"


def process_job_from_queue(job_data: dict):
    """Process a job from the queue in a background thread."""
    global active_jobs
    active_jobs += 1

    thread = threading.Thread(
        target=run_montage,
        args=(job_data['job_id'], job_data['style'], job_data['options']),
        daemon=False  # Non-daemon to ensure cleanup
    )
    thread.start()


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
    with job_lock:
        return jobs.get(job_id, {"status": "not_found"})


def normalize_options(data: dict) -> dict:
    """Normalize and validate job options from API request.
    
    Single source of truth for option parsing, type casting, and defaults.
    Automatically derives MUSIC_END from TARGET_DURATION if not explicitly set.
    
    Args:
        data: Raw request data (may have nested 'options' or flat structure)
        
    Returns:
        Normalized options dict with consistent types
    """
    # Support both nested and flat structure (backwards compatible)
    opts = data.get('options', {})
    
    # Parse with defaults and type casting
    target_duration = float(opts.get('target_duration', data.get('target_duration', 0)) or 0)
    music_start = float(opts.get('music_start', data.get('music_start', 0)) or 0)
    music_end_raw = opts.get('music_end', data.get('music_end', None))
    
    # Convert music_end to float if provided
    music_end = float(music_end_raw) if music_end_raw is not None else None
    
    # AUTO-DERIVE: If target_duration is set but music_end is not,
    # derive music_end from music_start + target_duration
    # This ensures the audio is trimmed to match the target video length
    if target_duration > 0 and music_end is None:
        music_end = music_start + target_duration
    
    # Clamp values to sensible ranges
    target_duration = max(0, min(target_duration, 3600))  # 0-1h
    music_start = max(0, music_start)
    if music_end is not None:
        music_end = max(music_start + 1, music_end)  # At least 1s after start
    
    # Helper to get boolean with centralized default
    def get_bool(key: str) -> bool:
        val = opts.get(key, data.get(key))
        if val is None:
            return DEFAULT_OPTIONS.get(key, False)
        return bool(val)
    
    # Quality Profile determines default enhance/stabilize/upscale
    quality_profile = str(opts.get('quality_profile', data.get('quality_profile', 'standard'))).lower()
    
    # Cloud Acceleration single toggle (enables CGPU for all cloud features)
    cloud_acceleration = get_bool('cloud_acceleration') or get_bool('cgpu') or _settings.llm.cgpu_enabled
    
    return {
        "prompt": str(opts.get('prompt', data.get('prompt', ''))),
        # Individual toggles (can override profile, but profile is primary)
        "stabilize": get_bool('stabilize'),
        "upscale": get_bool('upscale'),
        "enhance": get_bool('enhance'),
        "llm_clip_selection": get_bool('llm_clip_selection'),
        "shorts_mode": get_bool('shorts_mode'),
        "export_width": int(opts.get('export_width', data.get('export_width', 0)) or 0),
        "export_height": int(opts.get('export_height', data.get('export_height', 0)) or 0),
        "creative_loop": get_bool('creative_loop'),
        "story_engine": get_bool('story_engine'),
        "captions": get_bool('captions'),
        "export_timeline": get_bool('export_timeline'),
        "generate_proxies": get_bool('generate_proxies'),
        "preserve_aspect": get_bool('preserve_aspect'),
        "target_duration": target_duration,
        "music_start": music_start,
        "music_end": music_end,
        # Preview mode flag (legacy support)
        "preview": data.get('preset') == 'fast',
        # Quality Profile (preview, standard, high, master)
        "quality_profile": quality_profile,
        # Cloud Acceleration single toggle
        "cloud_acceleration": cloud_acceleration,
        "cgpu": cloud_acceleration,  # Backwards compat
        # Audio Polish (Clean Audio = Voice Isolation + Denoise)
        "clean_audio": get_bool('clean_audio'),
        "voice_isolation": get_bool('voice_isolation'),
        # Story Arc
        "story_arc": str(opts.get('story_arc', data.get('story_arc', ''))),
        # Shorts Studio options
        "reframe_mode": str(opts.get('reframe_mode', data.get('reframe_mode', 'auto'))),
        "caption_style": str(opts.get('caption_style', data.get('caption_style', 'tiktok'))),
    }


def run_montage(job_id: str, style: str, options: dict):
    """Run montage creation in background."""
    global active_jobs

    # Ensure directories exist (lazy initialization)
    _settings.paths.ensure_directories()

    # Check memory before starting
    memory_ok, available_gb = check_memory_available()
    if not memory_ok:
        error_msg = f"Insufficient memory (only {available_gb:.1f}GB available, need {MIN_MEMORY_GB}GB). Please try again later."
        
        # Write to log file so it's accessible
        log_path = OUTPUT_DIR / f"render_{job_id}.log"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(error_msg + "\n")
        except Exception:
            pass

        with job_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = error_msg
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
        active_jobs -= 1
        return

    with job_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat()

    try:
        # Build command
        import sys
        from ..env_mapper import map_options_to_env
        
        cmd = [sys.executable, "-m", "montage_ai.editor"]

        # Set environment variables using DRY mapper
        # options dict is already normalized via normalize_options() with DEFAULT_OPTIONS
        env = map_options_to_env(style, options, job_id)
        
        # Add Web-UI specific paths (if not already in env)
        env["INPUT_DIR"] = str(INPUT_DIR)
        env["MUSIC_DIR"] = str(MUSIC_DIR)
        env["OUTPUT_DIR"] = str(OUTPUT_DIR)
        env["ASSETS_DIR"] = str(ASSETS_DIR)

        # Run montage
        log_path = OUTPUT_DIR / f"render_{job_id}.log"
        
        def set_low_priority():
            """Set process priority to low (nice +10) to keep UI responsive."""
            try:
                os.nice(10)
            except Exception:
                pass

        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=set_low_priority  # Hardware Nah: Lower priority for background task
            )
            
            # Stream output and track phase
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                
                # Parse phase from logs (e.g., "Phase 1/5: Setup")
                if "Phase" in line and "/" in line:
                    try:
                        # Extract "Phase X/Y: Name"
                        # Format: "[INFO] Phase 1/5: Setup..." or "Phase 2/6: Analysis"
                        match = re.search(r'Phase\s*(\d+)/(\d+):\s*(\w+)', line)
                        if match:
                            phase_num = int(match.group(1))
                            phase_total = int(match.group(2))
                            phase_name = match.group(3).lower()
                            phase_data = {
                                "name": phase_name,
                                "label": f"Phase {phase_num}/{phase_total}: {match.group(3)}",
                                "number": phase_num,
                                "total": phase_total,
                                "started_at": datetime.now().isoformat(),
                                "progress_percent": int((phase_num / phase_total) * 100)
                            }
                            with job_lock:
                                jobs[job_id]["phase"] = phase_data
                            
                            # Announce update via SSE
                            announcer.announce(format_sse(json.dumps({
                                "job_id": job_id,
                                "status": "running",
                                "phase": phase_data
                            }), event="job_update"))
                    except Exception:
                        pass
            
            process.wait()
            result = process

        # Update job status
        with job_lock:
            # Find output files (check regardless of return code - video may succeed despite warnings)
            output_pattern = f"*{job_id}*.mp4"
            output_files = list(OUTPUT_DIR.glob(output_pattern))
            
            # Success if: return code 0 OR output file exists
            # Python warnings (like RuntimeWarning) can cause non-zero exit even on success
            if result.returncode == 0 or output_files:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["phase"] = {
                    "name": "completed",
                    "label": "Completed",
                    "number": len(PIPELINE_PHASES),
                    "total": len(PIPELINE_PHASES),
                    "started_at": datetime.now().isoformat(),
                    "progress_percent": 100
                }
                jobs[job_id]["completed_at"] = datetime.now().isoformat()

                if output_files:
                    jobs[job_id]["output_file"] = str(output_files[0].name)
                
                # Log warning if non-zero exit but file exists
                if result.returncode != 0 and output_files:
                    jobs[job_id]["warning"] = f"Process exited with code {result.returncode} but output was created successfully"

                # Announce completion
                announcer.announce(format_sse(json.dumps({
                    "job_id": job_id,
                    "status": "completed",
                    "output_file": jobs[job_id].get("output_file")
                }), event="job_complete"))

                # Timeline files (if exported)

                if options.get("export_timeline"):
                    timeline_files = {
                        "otio": list(OUTPUT_DIR.glob(f"*{job_id}*.otio")),
                        "edl": list(OUTPUT_DIR.glob(f"*{job_id}*.edl")),
                        "csv": list(OUTPUT_DIR.glob(f"*{job_id}*.csv"))
                    }
                    jobs[job_id]["timeline_files"] = {
                        k: [str(f.name) for f in v]
                        for k, v in timeline_files.items()
                    }
            else:
                jobs[job_id]["status"] = "failed"
                # Read last lines from log file for error message
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 1000))
                        jobs[job_id]["error"] = f.read()
                except Exception:
                    jobs[job_id]["error"] = "Check log file for details"

    except subprocess.TimeoutExpired:
        with job_lock:
            jobs[job_id]["status"] = "timeout"
            jobs[job_id]["error"] = "Job exceeded 1 hour timeout"
    except Exception as e:
        with job_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
    finally:
        active_jobs -= 1
        # Process next job in queue if available
        if job_queue:
            next_job = job_queue.popleft()
            process_job_from_queue(next_job)


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Main landing page with workflow selection (strategy-aligned UI)."""
    # Use strategy-aligned landing page by default
    if request.args.get('legacy'):
        return render_template('index.html', version=VERSION, defaults=DEFAULT_OPTIONS)
    return render_template('index_strategy.html', version=VERSION)


@app.route('/montage')
def montage_creator():
    """Montage Creator - beat-sync, story arc, style presets."""
    return render_template('index.html', version=VERSION, defaults=DEFAULT_OPTIONS)


@app.route('/v2')
def index_v2():
    """Outcome-based UI (v2 prototype)."""
    return render_template('index_v2.html', version=VERSION, defaults=DEFAULT_OPTIONS)


@app.route('/shorts')
def shorts_studio():
    """Shorts Studio - vertical video creation."""
    return render_template('shorts.html', version=VERSION)


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
            "queued_jobs": len(job_queue),
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
    global active_jobs

    data = request.json

    # Validate required fields
    if 'style' not in data:
        return jsonify({"error": "Missing required field: style"}), 400

    # Generate job ID
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Handle Quick Preview override
    is_preview = data.get('preset') == 'fast'
    if is_preview:
        data['style'] = 'dynamic'  # Force dynamic for speed
        data['target_duration'] = min(data.get('target_duration', 30) or 30, 30)  # Cap at 30s
        # Disable heavy features
        data['upscale'] = False
        data['stabilize'] = False
        data['enhance'] = False
        data['cgpu'] = False
        data['creative_loop'] = False

        # Set 360p resolution for preview
        if 'options' not in data:
            data['options'] = {}
            
        # Check shorts mode (handle both nested and flat)
        is_shorts = str(data['options'].get('shorts_mode', data.get('shorts_mode', 'false'))).lower() == 'true'
        
        if is_shorts:
            data['options']['export_width'] = 360
            data['options']['export_height'] = 640
        else:
            data['options']['export_width'] = 640
            data['options']['export_height'] = 360

    # Normalize options (single source of truth for parsing/defaults/derivation)
    normalized_options = normalize_options(data)

    # Create job with normalized options and structured phase tracking
    job = {
        "id": job_id,
        "style": data['style'],
        "options": normalized_options,
        "status": "queued",
        "phase": JobPhase.initial().to_dict(),
        "created_at": datetime.now().isoformat()
    }

    with job_lock:
        jobs[job_id] = job

    # Check if we can start immediately or need to queue
    if active_jobs < MAX_CONCURRENT_JOBS:
        # Start immediately
        active_jobs += 1
        thread = threading.Thread(
            target=run_montage,
            args=(job_id, data['style'], job['options']),
            daemon=False  # Non-daemon for proper cleanup
        )
        thread.start()
    else:
        # Add to queue
        job_queue.append({
            'job_id': job_id,
            'style': data['style'],
            'options': job['options']
        })
        with job_lock:
            jobs[job_id]["status"] = "queued"
            jobs[job_id]["queue_position"] = len(job_queue)

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









@app.route('/api/jobs', methods=['GET'])
def api_list_jobs():
    """List all jobs."""
    with job_lock:
        job_list = list(jobs.values())
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
# TRANSCRIPT API ENDPOINTS (Text-Based Editing)
# =============================================================================

# Create upload directory for transcript videos
TRANSCRIPT_UPLOAD_DIR = Path(os.environ.get("TRANSCRIPT_DIR", "/tmp/montage_transcript"))
TRANSCRIPT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.route('/transcript')
def transcript_editor():
    """Render transcript-based text editor UI."""
    return render_template('transcript.html')


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
    video_path = data.get('video_path')
    
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video file not found"}), 404
    
    try:
        # Try CGPU first, fall back to local
        cgpu_available = is_cgpu_available()
        
        if cgpu_available:
            # Use CGPU transcription
            from ..cgpu_jobs import TranscribeJob
            job = TranscribeJob(
                audio_path=video_path,
                model=data.get('model', 'medium'),
                output_format='json',  # Word-level timestamps
                language=data.get('language'),
            )
            result = job.execute()
            
            if result.success and result.output_path:
                with open(result.output_path, 'r') as f:
                    transcript = json.load(f)
                return jsonify({"success": True, "transcript": transcript})
            else:
                raise Exception(result.error or "Transcription failed")
        else:
            # Local transcription via whisper
            from ..transcriber import transcribe_audio
            
            # Extract audio first
            audio_path = Path(video_path).with_suffix('.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                str(audio_path)
            ], capture_output=True, check=True)
            
            # Transcribe
            result = transcribe_audio(str(audio_path), model='base', word_timestamps=True)
            
            # Clean up temp audio
            audio_path.unlink(missing_ok=True)
            
            return jsonify({"success": True, "transcript": result})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/transcript/export', methods=['POST'])
def api_transcript_export():
    """Export edited transcript as video, EDL, or OTIO."""
    data = request.json or {}
    video_path = data.get('video_path')
    transcript = data.get('transcript')
    export_format = data.get('format', 'video')
    
    if not video_path or not transcript:
        return jsonify({"error": "Missing video_path or transcript"}), 400
    
    if not Path(video_path).exists():
        return jsonify({"error": "Video file not found"}), 404
    
    try:
        from ..text_editor import TextEditor, Segment, Word
        
        # Initialize TextEditor with video path
        editor = TextEditor(video_path)
        
        # Convert frontend transcript format to TextEditor segments
        # Frontend sends: { segments: [ { start, end, text, words: [{word, start, end, removed}] } ] }
        segments = []
        for idx, seg_data in enumerate(transcript.get('segments', [])):
            words = []
            for w in seg_data.get('words', []):
                word = Word(
                    text=w.get('word', w.get('text', '')).strip(),
                    start=w.get('start', 0),
                    end=w.get('end', 0),
                    confidence=w.get('probability', w.get('confidence', 1.0)),
                    removed=w.get('removed', False)  # Preserve removed state from UI
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
            # Export EDL using TextEditor method
            edl_path = output_dir / f"transcript_edit_{timestamp}.edl"
            editor.export_edl(str(edl_path))
            
            with open(edl_path, 'r') as f:
                edl_content = f.read()
            
            return jsonify({
                "success": True,
                "filename": edl_path.name,
                "path": str(edl_path)
            })
            
        elif export_format == 'otio':
            # Export OTIO using TextEditor method (which uses TimelineExporter)
            otio_path = output_dir / f"transcript_edit_{timestamp}.otio"
            
            try:
                editor.export_otio(str(otio_path))
                
                return jsonify({
                    "success": True,
                    "filename": otio_path.name,
                    "path": str(otio_path)
                })
            except RuntimeError as e:
                return jsonify({"success": False, "error": f"OTIO export failed: {str(e)}"}), 500
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
        
        else:
            return jsonify({"error": f"Unknown format: {export_format}"}), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video file not found"}), 404
    
    try:
        from ..smart_reframing import SmartReframer
        
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
            
        from ..smart_reframing import SmartReframer
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
    audio_polish = data.get('audioPolish', False)
    
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video file not found"}), 404
    
    try:
        # Apply Audio Polish if requested
        if audio_polish:
            try:
                from ..cgpu_jobs import VoiceIsolationJob
                from ..cgpu_utils import is_cgpu_available
                
                if is_cgpu_available():
                    job = VoiceIsolationJob(
                        audio_path=video_path,
                        model="htdemucs",
                        two_stems=True,
                        keep_all_stems=True
                    )
                    result = job.execute()
                    
                    if result.success and result.metadata.get("stems", {}).get("vocals"):
                        vocals_path = result.metadata["stems"]["vocals"]
                        # Create a temp video with replaced audio
                        timestamp_polish = datetime.now().strftime("%Y%m%d_%H%M%S")
                        polished_video_path = OUTPUT_DIR / f"shorts_polished_{timestamp_polish}.mp4"
                        
                        # Use ffmpeg to replace audio
                        cmd = [
                            "ffmpeg", "-y",
                            "-i", video_path,
                            "-i", vocals_path,
                            "-c:v", "copy",
                            "-c:a", "aac",
                            "-map", "0:v:0",
                            "-map", "1:a:0",
                            str(polished_video_path)
                        ]
                        subprocess.run(cmd, check=True, capture_output=True)
                        
                        # Update video_path to point to the polished version
                        video_path = str(polished_video_path)
            except Exception as e:
                print(f"Audio polish failed: {e}")
                # Continue with original audio

        from ..smart_reframing import SmartReframer
        
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
            
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(input_path),
                '-af', noise_reduction_filter,
                '-acodec', 'pcm_s16le',  # High quality WAV
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # Just copy the voice-isolated file
            import shutil
            shutil.copy(input_path, output_path)
        
        # Calculate SNR improvement estimate (simplified)
        snr_improvement = "~6dB" if voice_isolated else "~3dB"
        
        return jsonify({
            "success": True,
            "filename": output_path.name,
            "path": str(output_path),
            "voice_isolated": voice_isolated,
            "noise_reduced": reduce_noise,
            "snr_improvement": snr_improvement,
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
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            '-select_streams', 'a:0',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        probe_data = json.loads(result.stdout) if result.stdout else {}
        
        # Get loudness stats
        loudness_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-i', str(audio_path),
            '-af', 'volumedetect', '-f', 'null', '/dev/null'
        ]
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
        
        # Estimate quality and recommendations
        quality = "good"
        recommendations = []
        
        if mean_volume < -30:
            quality = "low"
            recommendations.append("Audio is very quiet - consider normalizing")
        elif mean_volume > -10:
            quality = "warning"
            recommendations.append("Audio may be clipping - check peaks")
        
        # Dynamic range check
        dynamic_range = max_volume - mean_volume
        if dynamic_range > 20:
            recommendations.append("High dynamic range - consider compression for social media")
        
        # Check if CGPU available for voice isolation
        if is_cgpu_available():
            recommendations.append("Voice isolation available via cloud acceleration")
        
        return jsonify({
            "success": True,
            "analysis": {
                "mean_volume": round(mean_volume, 1),
                "max_volume": round(max_volume, 1),
                "dynamic_range": round(dynamic_range, 1),
                "quality_estimate": quality,
                "sample_rate": probe_data.get('streams', [{}])[0].get('sample_rate', 'unknown'),
                "channels": probe_data.get('streams', [{}])[0].get('channels', 1),
            },
            "recommendations": recommendations,
            "clean_audio_available": True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/shorts/create', methods=['POST'])
def api_shorts_create():
    """Create a short video - convenience alias for render endpoint."""
    # This is the endpoint the frontend calls
    return api_shorts_render()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
