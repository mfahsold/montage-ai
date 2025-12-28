"""
Montage AI Web Interface

Simple Flask web UI for creating video montages.
DRY + KISS: Minimal dependencies, no over-engineering.
"""

import os
import json
import subprocess
import threading
import psutil
from datetime import datetime
from pathlib import Path
from collections import deque
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename


def get_version() -> str:
    """Get version from git commit hash (short) or fallback to env/default.
    
    Priority:
    1. GIT_COMMIT env var (set at build time in Dockerfile)
    2. Live git rev-parse (if in git repo)
    3. Fallback to "dev"
    """
    # Check env var first (set at Docker build time)
    git_commit = os.environ.get("GIT_COMMIT", "").strip()
    if git_commit:
        return git_commit[:8]  # Short hash
    
    # Try live git command (works in dev, not in container usually)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).parent.parent.parent.parent  # repo root
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return "dev"


VERSION = get_version()

# Paths
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/data/input"))
MUSIC_DIR = Path(os.environ.get("MUSIC_DIR", "/data/music"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", "/data/assets"))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
MUSIC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DEFAULT OPTIONS (single source of truth - controlled via environment)
# =============================================================================
DEFAULT_OPTIONS = {
    "enhance": os.environ.get("DEFAULT_ENHANCE", "true").lower() == "true",
    "stabilize": os.environ.get("DEFAULT_STABILIZE", "false").lower() == "true",
    "upscale": os.environ.get("DEFAULT_UPSCALE", "false").lower() == "true",
    "cgpu": os.environ.get("DEFAULT_CGPU", "false").lower() == "true",
    "llm_clip_selection": os.environ.get("LLM_CLIP_SELECTION", "true").lower() == "true",
    "export_timeline": os.environ.get("DEFAULT_EXPORT_TIMELINE", "false").lower() == "true",
    "generate_proxies": os.environ.get("DEFAULT_GENERATE_PROXIES", "false").lower() == "true",
    "preserve_aspect": os.environ.get("PRESERVE_ASPECT", "false").lower() == "true",
}

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload
app.config['UPLOAD_FOLDER'] = INPUT_DIR

# Job queue and management
jobs = {}
job_lock = threading.Lock()
job_queue = deque()
active_jobs = 0
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
MIN_MEMORY_GB = 2  # Minimum memory required to start a job


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


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
    
    return {
        "prompt": str(opts.get('prompt', data.get('prompt', ''))),
        "stabilize": get_bool('stabilize'),
        "upscale": get_bool('upscale'),
        "enhance": get_bool('enhance'),
        "llm_clip_selection": get_bool('llm_clip_selection'),
        "export_timeline": get_bool('export_timeline'),
        "generate_proxies": get_bool('generate_proxies'),
        # Default cgpu to true if env var is set, otherwise use UI value
        "cgpu": get_bool('cgpu') or os.environ.get("CGPU_ENABLED", "false").lower() == "true",
        "preserve_aspect": get_bool('preserve_aspect'),
        "target_duration": target_duration,
        "music_start": music_start,
        "music_end": music_end,
    }


def run_montage(job_id: str, style: str, options: dict):
    """Run montage creation in background."""
    global active_jobs

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
        cmd = [sys.executable, "-m", "montage_ai.editor"]

        # Set environment variables
        # options dict is already normalized via normalize_options() with DEFAULT_OPTIONS
        env = os.environ.copy()
        env["INPUT_DIR"] = str(INPUT_DIR)
        env["MUSIC_DIR"] = str(MUSIC_DIR)
        env["OUTPUT_DIR"] = str(OUTPUT_DIR)
        env["ASSETS_DIR"] = str(ASSETS_DIR)
        env["JOB_ID"] = job_id
        env["CUT_STYLE"] = style
        env["CREATIVE_PROMPT"] = options.get("prompt", "")
        env["STABILIZE"] = "true" if options.get("stabilize") else "false"
        env["UPSCALE"] = "true" if options.get("upscale") else "false"
        env["ENHANCE"] = "true" if options.get("enhance") else "false"
        env["LLM_CLIP_SELECTION"] = "true" if options.get("llm_clip_selection") else "false"
        env["EXPORT_TIMELINE"] = "true" if options.get("export_timeline") else "false"
        env["GENERATE_PROXIES"] = "true" if options.get("generate_proxies") else "false"
        # Aspect ratio handling: letterbox/pillarbox vs crop
        env["PRESERVE_ASPECT"] = "true" if options.get("preserve_aspect") else "false"
        # cgpu checkbox enables BOTH LLM and GPU upscaling
        env["CGPU_ENABLED"] = "true" if options.get("cgpu") else "false"
        env["CGPU_GPU_ENABLED"] = "true" if options.get("cgpu") else "false"
        # Video duration & music trimming (already normalized by normalize_options)
        env["TARGET_DURATION"] = str(options.get("target_duration", 0))
        env["MUSIC_START"] = str(options.get("music_start", 0))
        music_end = options.get("music_end")
        env["MUSIC_END"] = str(music_end) if music_end is not None else ""
        env["VERBOSE"] = "true"
        env["PYTHONUNBUFFERED"] = "1"

        # Run montage
        log_path = OUTPUT_DIR / f"render_{job_id}.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1 hour max
            )

        # Update job status
        with job_lock:
            # Find output files (check regardless of return code - video may succeed despite warnings)
            output_pattern = f"*{job_id}*.mp4"
            output_files = list(OUTPUT_DIR.glob(output_pattern))
            
            # Success if: return code 0 OR output file exists
            # Python warnings (like RuntimeWarning) can cause non-zero exit even on success
            if result.returncode == 0 or output_files:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()

                if output_files:
                    jobs[job_id]["output_file"] = str(output_files[0].name)
                
                # Log warning if non-zero exit but file exists
                if result.returncode != 0 and output_files:
                    jobs[job_id]["warning"] = f"Process exited with code {result.returncode} but output was created successfully"

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
    """Main page."""
    return render_template('index.html', version=VERSION, defaults=DEFAULT_OPTIONS)


@app.route('/api/status')
def api_status():
    """API health check with system stats."""
    mem = psutil.virtual_memory()
    memory_ok, available_gb = check_memory_available()

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
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS
        }
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

    # Validate file type
    video_extensions = {'mp4', 'mov', 'avi', 'mkv'}
    music_extensions = {'mp3', 'wav', 'flac', 'm4a'}

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

    # Normalize options (single source of truth for parsing/defaults/derivation)
    normalized_options = normalize_options(data)
    
    # Create job with normalized options
    job = {
        "id": job_id,
        "style": data['style'],
        "options": normalized_options,
        "status": "queued",
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
