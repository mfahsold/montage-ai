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
        try:
            result = subprocess.run(
                ["ffmpeg", "-filters"],
                capture_output=True,
                text=True,
                timeout=5
            )
            available_filters = result.stdout
            
            # Check for key filters
            self.has_sidechaincompress = "sidechaincompress" in available_filters
            self.has_astats = "astats" in available_filters
            self.has_volumedetect = "volumedetect" in available_filters
            self.has_silencedetect = "silencedetect" in available_filters
            
            logger.debug(f"Audio filters available: sidechaincompress={self.has_sidechaincompress}, "
                        f"astats={self.has_astats}, volumedetect={self.has_volumedetect}")
        except Exception as e:
            logger.warning(f"Could not check filter availability: {e}. Assuming all available.")
            self.has_sidechaincompress = True
            self.has_astats = True
            self.has_volumedetect = True
            self.has_silencedetect = True

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
        # Dynamically adjust ratio based on ducking_amount (dB reduction)
        # ducking_amount: 15dB = ratio ~5, 20dB = ratio ~10, 10dB = ratio ~3
        ratio = max(2, min(20, ducking_amount / 3.0))  # Convert dB to ratio (clamped 2-20)
        threshold = 0.1 if ducking_amount < 20 else 0.05  # Lower threshold for more aggressive ducking
        
        core_filter = AUDIO_FILTERS.get(
            "ducking_core", 
            f"sidechaincompress=threshold={threshold}:ratio={ratio:.1f}:attack=50:release=300:link=average"
        )
        
        logger.debug(f"Ducking parameters: amount={ducking_amount}dB, ratio={ratio:.1f}, threshold={threshold}")
        
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
        Approximate Signal-to-Noise Ratio using volumedetect and astats filters.
        Returns estimated SNR in dB.
        
        Method: Use astats to get mean and peak RMS levels, then estimate SNR
        by comparing peak signal (90th percentile) to mean noise floor.
        """
        if not self.has_astats:
            logger.warning("astats filter not available, cannot measure SNR")
            return 0.0
        
        try:
            # Use astats to get detailed audio statistics
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-af", "astats=measure_perchannel=none:measure_overall=Peak_level+RMS_level+RMS_peak:metadata=1",
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse stderr for statistics
            stderr = result.stderr
            
            # Extract RMS levels (dB)
            rms_level = None
            rms_peak = None
            
            for line in stderr.split('\n'):
                if 'RMS level dB:' in line:
                    try:
                        rms_level = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass
                elif 'RMS peak dB:' in line:
                    try:
                        rms_peak = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass
            
            if rms_level is not None and rms_peak is not None:
                # Estimate SNR as difference between peak signal and average level
                # This is a rough approximation: peak signal represents "signal",
                # average RMS represents noise floor + signal
                estimated_snr = abs(rms_peak - rms_level)
                
                logger.info(f"Audio SNR analysis: RMS={rms_level:.1f}dB, Peak={rms_peak:.1f}dB, "
                           f"Estimated SNR={estimated_snr:.1f}dB")
                
                return max(0.0, estimated_snr)  # Clamp to positive values
            else:
                logger.warning(f"Could not parse audio statistics from ffmpeg output")
                return 0.0
                
        except subprocess.TimeoutExpired:
            logger.error(f"SNR measurement timed out for {input_path}")
            return 0.0
        except Exception as e:
            logger.error(f"SNR measurement failed: {e}")
            return 0.0
