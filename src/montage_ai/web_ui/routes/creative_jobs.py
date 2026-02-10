"""
Creative Cut Planning & Rendering Routes
Integrates analyze_footage_creative.py and render_creative_cut.py into the web UI.
"""

import json
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Blueprint, request, jsonify, send_file
import time
import logging

from ...config import get_settings
from ...logger import logger

# Create Blueprint
creative_bp = Blueprint('creative_jobs', __name__, url_prefix='/api/creative')

# Configuration
settings = get_settings()
INPUT_DIR = Path(settings.paths.input_dir)
OUTPUT_DIR = Path(settings.paths.output_dir)

# Job registry (in-memory; could be backed by DB)
_job_registry: Dict[str, Dict[str, Any]] = {}


def _get_job_id(session_id: str) -> str:
    """Generate unique job ID for this session."""
    return f"creative_{session_id}_{int(time.time())}"


def _run_analysis_threaded(job_id: str, user_callback=None):
    """Run analyze_footage_creative.py in background thread."""
    try:
        logger.info(f"[{job_id}] Starting creative footage analysis...")
        _job_registry[job_id]['status'] = 'analyzing'
        _job_registry[job_id]['progress'] = 0
        _job_registry[job_id]['started_at'] = time.time()

        # Run analysis script
        cmd = [
            'python3', '-c',
            """
import sys
sys.path.insert(0, '/home/codeai/montage-ai')
from analyze_footage_creative import analyze_and_plan_creative_cut
result = analyze_and_plan_creative_cut()
print(json.dumps(result))
""",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            cwd='/home/codeai/montage-ai'
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            logger.error(f"[{job_id}] Analysis failed: {error_msg}")
            _job_registry[job_id]['status'] = 'failed'
            _job_registry[job_id]['error'] = error_msg
            if user_callback:
                user_callback(job_id, 'failed', error_msg)
            return

        # Parse cut plan
        try:
            output = result.stdout.strip()
            cut_plan = json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"[{job_id}] Failed to parse analysis output: {e}")
            _job_registry[job_id]['status'] = 'failed'
            _job_registry[job_id]['error'] = f"JSON parse error: {str(e)}"
            if user_callback:
                user_callback(job_id, 'failed', str(e))
            return

        logger.info(f"[{job_id}] Analysis complete. Generated {cut_plan.get('total_cuts', 0)} cuts")
        _job_registry[job_id]['status'] = 'analyzed'
        _job_registry[job_id]['progress'] = 100
        _job_registry[job_id]['cut_plan'] = cut_plan
        _job_registry[job_id]['completed_at'] = time.time()

        if user_callback:
            user_callback(job_id, 'analyzed', cut_plan)

    except subprocess.TimeoutExpired:
        logger.error(f"[{job_id}] Analysis timed out")
        _job_registry[job_id]['status'] = 'failed'
        _job_registry[job_id]['error'] = 'Analysis timeout (exceeded 5 minutes)'
        if user_callback:
            user_callback(job_id, 'failed', 'Timeout')
    except Exception as e:
        logger.error(f"[{job_id}] Unexpected error during analysis: {str(e)}")
        _job_registry[job_id]['status'] = 'failed'
        _job_registry[job_id]['error'] = str(e)
        if user_callback:
            user_callback(job_id, 'failed', str(e))


def _run_rendering_threaded(job_id: str, cut_plan: Dict, color_grade: str = "none", user_callback=None):
    """Run render_creative_cut.py in background thread with optional color grading."""
    try:
        logger.info(f"[{job_id}] Starting creative cut rendering (grade={color_grade})...")
        _job_registry[job_id]['status'] = 'rendering'
        _job_registry[job_id]['progress'] = 0
        _job_registry[job_id]['render_started_at'] = time.time()
        _job_registry[job_id]['color_grade'] = color_grade

        # Save cut plan to temp location
        plan_file = Path('/tmp/creative_cut_plan_ui.json')
        plan_file.write_text(json.dumps(cut_plan))

        # Run rendering script with color grading
        cmd = [
            'python3', '-c',
            f"""
import sys
sys.path.insert(0, '/home/codeai/montage-ai')
from render_creative_cut import render_with_plan
cut_plan = {cut_plan}
result = render_with_plan(cut_plan, color_grade='{color_grade}')
import json
print(json.dumps(result))
""",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 min timeout for rendering
            cwd='/home/codeai/montage-ai'
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            logger.error(f"[{job_id}] Rendering failed: {error_msg}")
            _job_registry[job_id]['status'] = 'failed'
            _job_registry[job_id]['error'] = error_msg
            if user_callback:
                user_callback(job_id, 'failed', error_msg)
            return

        # Parse rendering result
        try:
            output = result.stdout.strip()
            # Find JSON in output (might be mixed with logs)
            import re
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                render_result = json.loads(json_match.group())
            else:
                render_result = json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"[{job_id}] Failed to parse rendering output: {e}")
            _job_registry[job_id]['status'] = 'failed'
            _job_registry[job_id]['error'] = f"JSON parse error: {str(e)}"
            if user_callback:
                user_callback(job_id, 'failed', str(e))
            return

        if not render_result.get('success'):
            error_msg = render_result.get('error', 'Unknown rendering error')
            logger.error(f"[{job_id}] Rendering failed: {error_msg}")
            _job_registry[job_id]['status'] = 'failed'
            _job_registry[job_id]['error'] = error_msg
            if user_callback:
                user_callback(job_id, 'failed', error_msg)
            return

        output_file = render_result.get('output_file')
        if not output_file or not Path(output_file).exists():
            logger.error(f"[{job_id}] Output file not found: {output_file}")
            _job_registry[job_id]['status'] = 'failed'
            _job_registry[job_id]['error'] = f"Output file not created: {output_file}"
            if user_callback:
                user_callback(job_id, 'failed', 'Output file not created')
            return

        logger.info(f"[{job_id}] Rendering complete (grade={color_grade}). Output: {output_file}")
        _job_registry[job_id]['status'] = 'completed'
        _job_registry[job_id]['progress'] = 100
        _job_registry[job_id]['output_file'] = output_file
        _job_registry[job_id]['file_size'] = Path(output_file).stat().st_size
        _job_registry[job_id]['download_path'] = render_result.get('download_path', output_file)
        _job_registry[job_id]['completed_at'] = time.time()

        if user_callback:
            user_callback(job_id, 'completed', {'output_file': output_file})

    except subprocess.TimeoutExpired:
        logger.error(f"[{job_id}] Rendering timed out")
        _job_registry[job_id]['status'] = 'failed'
        _job_registry[job_id]['error'] = 'Rendering timeout (exceeded 15 minutes)'
        if user_callback:
            user_callback(job_id, 'failed', 'Timeout')
    except Exception as e:
        logger.error(f"[{job_id}] Unexpected error during rendering: {str(e)}")
        _job_registry[job_id]['status'] = 'failed'
        _job_registry[job_id]['error'] = str(e)
        if user_callback:
            user_callback(job_id, 'failed', str(e))


# ============================================================================
# ROUTES
# ============================================================================

@creative_bp.route('/analyze', methods=['POST'])
def start_analysis():
    """Start creative footage analysis job."""
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    target_duration = data.get('target_duration', 45)

    job_id = _get_job_id(session_id)
    _job_registry[job_id] = {
        'job_id': job_id,
        'session_id': session_id,
        'status': 'queued',
        'progress': 0,
        'target_duration': target_duration,
        'created_at': time.time(),
    }

    logger.info(f"Starting analysis job {job_id} for session {session_id}")

    # Run analysis in background
    thread = threading.Thread(
        target=_run_analysis_threaded,
        args=(job_id,),
        daemon=True
    )
    thread.start()

    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': 'Analysis job queued'
    })


@creative_bp.route('/analyze/<job_id>', methods=['GET'])
def get_analysis_status(job_id):
    """Get status of analysis job."""
    if job_id not in _job_registry:
        return jsonify({'error': 'Job not found'}), 404

    job = _job_registry[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job.get('progress', 0),
        'error': job.get('error'),
        'cut_plan': job.get('cut_plan'),
        'created_at': job.get('created_at'),
        'completed_at': job.get('completed_at'),
    })


@creative_bp.route('/render', methods=['POST'])
def start_rendering():
    """Start creative cut rendering job with optional color grading.
    
    Request JSON:
        cut_plan (dict): The cut plan from analysis
        session_id (str): Session identifier
        color_grade (str): Color grading preset ('none', 'warm', 'cool', 'vibrant', 'high_contrast', 'cinematic')
    """
    data = request.json or {}
    cut_plan = data.get('cut_plan')
    session_id = data.get('session_id', 'default')
    color_grade = data.get('color_grade', 'none')  # Default to no grading

    if not cut_plan:
        return jsonify({'error': 'cut_plan required'}), 400

    # Validate color grade
    valid_grades = ['none', 'warm', 'cool', 'vibrant', 'high_contrast', 'cinematic']
    if color_grade not in valid_grades:
        return jsonify({'error': f'Invalid color_grade. Must be one of: {", ".join(valid_grades)}'}), 400

    job_id = _get_job_id(session_id) + '_render'
    _job_registry[job_id] = {
        'job_id': job_id,
        'session_id': session_id,
        'status': 'queued',
        'progress': 0,
        'color_grade': color_grade,
        'created_at': time.time(),
    }

    logger.info(f"Starting rendering job {job_id} (grade={color_grade}) for session {session_id}")

    # Run rendering in background with color grading
    thread = threading.Thread(
        target=_run_rendering_threaded,
        args=(job_id, cut_plan, color_grade),
        daemon=True
    )
    thread.start()

    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': f'Rendering job queued (color_grade={color_grade})',
        'color_grade': color_grade,
    })


@creative_bp.route('/render/<job_id>', methods=['GET'])
def get_rendering_status(job_id):
    """Get status of rendering job."""
    if job_id not in _job_registry:
        return jsonify({'error': 'Job not found'}), 404

    job = _job_registry[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job.get('progress', 0),
        'error': job.get('error'),
        'output_file': job.get('output_file'),
        'download_path': job.get('download_path'),
        'file_size': job.get('file_size'),
        'color_grade': job.get('color_grade', 'none'),
        'created_at': job.get('created_at'),
        'completed_at': job.get('completed_at'),
    })


@creative_bp.route('/download/<job_id>', methods=['GET'])
def download_video(job_id):
    """Download rendered video from completed job."""
    if job_id not in _job_registry:
        return jsonify({'error': 'Job not found'}), 404

    job = _job_registry[job_id]
    output_file = job.get('output_file')

    if not output_file or not Path(output_file).exists():
        return jsonify({'error': 'Video file not found'}), 404

    if job['status'] != 'completed':
        return jsonify({'error': f'Job not completed (status: {job["status"]})'}), 400

    return send_file(
        output_file,
        as_attachment=True,
        download_name=f"creative_trailer_{job_id}.mp4",
        mimetype='video/mp4'
    )


@creative_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """List all creative jobs."""
    return jsonify({
        'jobs': list(_job_registry.values())
    })


@creative_bp.route('/jobs/<job_id>', methods=['DELETE'])
def cancel_job(job_id):
    """Cancel a job (if still queued/running)."""
    if job_id not in _job_registry:
        return jsonify({'error': 'Job not found'}), 404

    job = _job_registry[job_id]
    if job['status'] in ['completed', 'failed']:
        return jsonify({'error': f'Cannot cancel job in {job["status"]} state'}), 400

    job['status'] = 'cancelled'
    logger.info(f"Job {job_id} cancelled by user")

    return jsonify({
        'job_id': job_id,
        'status': 'cancelled',
        'message': 'Job cancelled'
    })
