from flask import Blueprint, request, jsonify
import subprocess
import json
from pathlib import Path
from ...cgpu_utils import is_cgpu_available
from ...ffmpeg_utils import build_ffmpeg_cmd, build_ffprobe_cmd
from ..highlights import detect_highlights

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
                except (ValueError, IndexError):
                    pass  # Skip malformed line
            if 'max_volume' in line:
                try:
                    max_volume = float(line.split(':')[1].strip().replace(' dB', ''))
                except (ValueError, IndexError):
                    pass  # Skip malformed line
        
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


@analysis_bp.route('/highlights/detect', methods=['POST'])
def api_detect_highlights():
    """
    Detect highlight moments in video using multi-signal analysis.
    
    POST /api/highlights/detect
    Body: {
        "video_path": "/path/to/video.mp4",
        "max_clips": 5,
        "min_duration": 5.0,
        "max_duration": 60.0,
        "include_speech": true
    }
    
    Returns:
        {
            "success": true,
            "highlights": [
                {
                    "time": 10.5,
                    "start": 10.0,
                    "end": 15.0,
                    "duration": 5.0,
                    "score": 0.85,
                    "type": "Energy",
                    "label": "🔥 High Energy (85%)"
                },
                ...
            ]
        }
    """
    data = request.json or {}
    video_path = data.get('video_path')
    
    if not video_path:
        return jsonify({"error": "video_path is required"}), 400
    
    if not Path(video_path).exists():
        return jsonify({"error": f"Video file not found: {video_path}"}), 404
    
    try:
        # Parse options
        max_clips = data.get('max_clips', 5)
        min_duration = data.get('min_duration', 5.0)
        max_duration = data.get('max_duration', 60.0)
        include_speech = data.get('include_speech', True)
        
        # Detect highlights using multi-signal analysis
        highlights = detect_highlights(
            video_path=video_path,
            max_clips=max_clips,
            min_duration=min_duration,
            max_duration=max_duration,
            include_speech=include_speech
        )
        
        return jsonify({
            "success": True,
            "highlights": highlights,
            "count": len(highlights),
            "video_path": video_path
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
