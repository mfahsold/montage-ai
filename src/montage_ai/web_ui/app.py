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

VERSION = "0.3.0"

# Paths
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/data/input"))
MUSIC_DIR = Path(os.environ.get("MUSIC_DIR", "/data/music"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/output"))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
MUSIC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
MIN_MEMORY_GB = 4  # Minimum memory required to start a job


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


def run_montage(job_id: str, style: str, options: dict):
    """Run montage creation in background."""
    global active_jobs

    # Check memory before starting
    memory_ok, available_gb = check_memory_available()
    if not memory_ok:
        with job_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"Insufficient memory (only {available_gb:.1f}GB available, need {MIN_MEMORY_GB}GB). Please try again later."
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
        active_jobs -= 1
        return

    with job_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat()

    try:
        # Build command
        cmd = ["python", "-m", "montage_ai.editor"]

        # Set environment variables
        env = os.environ.copy()
        env["JOB_ID"] = job_id
        env["CUT_STYLE"] = style
        env["CREATIVE_PROMPT"] = options.get("prompt", "")
        env["STABILIZE"] = "true" if options.get("stabilize", False) else "false"
        env["UPSCALE"] = "true" if options.get("upscale", False) else "false"
        env["ENHANCE"] = "true" if options.get("enhance", True) else "false"
        env["EXPORT_TIMELINE"] = "true" if options.get("export_timeline", False) else "false"
        env["GENERATE_PROXIES"] = "true" if options.get("generate_proxies", False) else "false"
        # cgpu checkbox enables BOTH LLM and GPU upscaling
        env["CGPU_ENABLED"] = "true" if options.get("cgpu", False) else "false"
        env["CGPU_GPU_ENABLED"] = "true" if options.get("cgpu", False) else "false"
        env["VERBOSE"] = "true"

        # Run montage
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max
        )

        # Update job status
        with job_lock:
            if result.returncode == 0:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()

                # Find output files
                output_pattern = f"*{job_id}*.mp4"
                output_files = list(OUTPUT_DIR.glob(output_pattern))
                if output_files:
                    jobs[job_id]["output_file"] = str(output_files[0].name)

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
                jobs[job_id]["error"] = result.stderr[-500:]  # Last 500 chars

            jobs[job_id]["stdout"] = result.stdout[-1000:]  # Last 1000 chars

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
    return render_template('index.html', version=VERSION)


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

    # Extract options from nested object or top-level (backwards compatible)
    options_data = data.get('options', {})
    
    # Create job
    job = {
        "id": job_id,
        "style": data['style'],
        "options": {
            "prompt": options_data.get('prompt', data.get('prompt', '')),
            "stabilize": options_data.get('stabilize', data.get('stabilize', False)),
            "upscale": options_data.get('upscale', data.get('upscale', False)),
            "enhance": options_data.get('enhance', data.get('enhance', True)),
            "export_timeline": options_data.get('export_timeline', data.get('export_timeline', False)),
            "generate_proxies": options_data.get('generate_proxies', data.get('generate_proxies', False)),
            "cgpu": options_data.get('cgpu', data.get('cgpu', False))
        },
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
    """Download output file."""
    filepath = OUTPUT_DIR / secure_filename(filename)

    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(filepath, as_attachment=True)


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
