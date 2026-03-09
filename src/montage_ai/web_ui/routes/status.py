"""
Status Routes - Health, Metrics, Configuration
"""

from flask import jsonify, Response
from . import status_bp


@status_bp.route("/status")
def api_status():
    """System status endpoint."""
    return jsonify(
        {
            "status": "ok",
            "version": "1.0.0",
        }
    )


@status_bp.route("/config/defaults")
def config_defaults():
    """Default configuration values."""
    return jsonify(
        {
            "paths": {
                "input_dir": "/data/input",
                "music_dir": "/data/music",
                "output_dir": "/data/output",
            },
            "features": {
                "upscale": True,
                "stabilize": True,
                "color_grade": True,
            },
        }
    )


@status_bp.route("/telemetry")
def telemetry():
    """System telemetry data."""
    return jsonify(
        {
            "active_jobs": 0,
            "queue_depth": 0,
            "cpu_percent": 0,
            "memory_percent": 0,
        }
    )


@status_bp.route("/quality-profiles")
def quality_profiles():
    """Available quality profiles."""
    return jsonify(
        [
            {"id": "preview", "name": "Preview (360p)", "fast": True},
            {"id": "standard", "name": "Standard (1080p)", "fast": False},
            {"id": "high", "name": "High (1440p)", "fast": False},
            {"id": "master", "name": "Master (4K)", "fast": False},
        ]
    )


@status_bp.route("/cloud/status")
def cloud_status():
    """Cloud GPU status."""
    return jsonify(
        {
            "cgpu_available": False,
            "cloud_jobs_active": 0,
        }
    )
