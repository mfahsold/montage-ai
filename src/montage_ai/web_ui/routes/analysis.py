from flask import Blueprint, request, jsonify
import subprocess
import json
from pathlib import Path
from ...cgpu_utils import is_cgpu_available

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')

@analysis_bp.route('/audio/analyze', methods=['POST'])
def api_audio_analyze():
    """
    Analyze audio quality and suggest improvements.
    """
    data = request.json or {}
    audio_path = data.get('audio_path') or data.get('video_path')
    
    if not audio_path or not Path(audio_path).exists():
        return jsonify({"error": "Audio file not found"}), 404
    
    try:
        # Extract audio stats using FFmpeg
        cmd = [