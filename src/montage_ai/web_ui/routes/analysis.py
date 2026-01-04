from flask import Blueprint, request, jsonify
import subprocess
import json
from pathlib import Path
from ...cgpu_utils import is_cgpu_available
from ...ffmpeg_utils import build_ffmpeg_cmd, build_ffprobe_cmd

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
