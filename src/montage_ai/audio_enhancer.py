"""
Audio Enhancer - Polish audio tracks for professional output.

Features:
- Voice Isolation (Denoise + EQ + Gate)
- Audio Ducking (Music lowers when voice is present)
- SNR Check (Signal-to-Noise Ratio)
"""

import os
import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union

from .ffmpeg_config import FFmpegConfig, AUDIO_FILTERS
from .ffmpeg_utils import build_ffmpeg_cmd, build_ffprobe_cmd
from .core.cmd_runner import run_command

logger = logging.getLogger(__name__)

class AudioEnhancer:
    """Handles audio post-production tasks."""
    
    def __init__(self):
        self._check_available_filters()

    def _check_available_filters(self):
        """Check availability of advanced filters."""
        # TODO: Run ffmpeg -filters and parse
        pass

    def enhance_voice(self, input_path: str, output_path: str, strength: float = 0.5) -> bool:
        """
        Apply 'podcast polish' processing to a voice track.
        """
        logger.info(f"Enhancing voice audio: {input_path}")
        
        # Use configuration constant for DRY
        filter_str = AUDIO_FILTERS["voice_polish"]
        
        cmd = build_ffmpeg_cmd([
            "-i", input_path,
            "-af", filter_str,
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path)
        ])
        
        try:
            run_command(cmd)
            return True
        except Exception as e:
            logger.error(f"Voice enhancement failed: {e}")
            return False

    def auto_duck(self, 
                  voice_track: str, 
                  music_track: str, 
                  output_path: str, 
                  ducking_amount: float = 15.0) -> bool:
        """
        Mix voice and music, ducking music when voice is present.
        """
        logger.info(f"Applying auto-ducking (Music: {music_track} under Voice: {voice_track})")
        
        # FFmpeg sidechaincompress filter params from config
        # TODO: Dynamically adjust ratio based on ducking_amount
        core_filter = AUDIO_FILTERS.get("ducking_core", "sidechaincompress=threshold=0.1:ratio=5:attack=50:release=300:link=average")
        
        # [1] (music) is compressed based on signal from [0] (voice)
        # acompressor inputs: [main][sidechain]
        filter_complex = (
            f"[1:a]volume=0.8[music];" # Lower music base level slightly
            f"[0:a][music]{core_filter}[voice][ducked_music];"
            f"[voice][ducked_music]amix=inputs=2:duration=first:dropout_transition=2[out]"
        )
        
        cmd = build_ffmpeg_cmd([
            "-i", voice_track,
            "-i", music_track,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:a", "aac",
            str(output_path)
        ])
         
        try:
            run_command(cmd)
            return True
        except Exception as e:
            logger.error(f"Auto-ducking failed: {e}")
            return False

    def check_snr(self, input_path: str) -> float:
        """
        Approximate Signal-to-Noise Ratio using rms levels of silence vs speech.
        Returns estimated SNR in dB.
        """
        # 1. Detect silence levels (volumedetect? astats?)
        # A simple robust way: 10th percentile (noise) vs 90th percentile (signal) RMS
        
        # Using astats
        cmd = ["ffmpeg", "-i", input_path, "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level", "-f", "null", "-"]
        # This produces A LOT of output.
        
        # Alternative: measure mean volume.
        # It's hard to get true SNR without VAD.
        
        # For MVP: Run volumedetect filter
        return 0.0 # TODO: Implement accurate measurement
