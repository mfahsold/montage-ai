"""
Dialogue Detection & Auto-Ducking Module for Montage AI

Detects speech/dialogue in audio and generates volume automation keyframes
for automatic music ducking during speech segments.

Features:
- VAD (Voice Activity Detection) using silero-vad or FFmpeg fallback
- Configurable duck levels and attack/release times
- NLE-compatible keyframe export (Resolve, Premiere, FCP)
- Smooth fade curves to avoid audio artifacts

Based on industry standards:
- Broadcast: -6dB to -12dB ducking during speech
- Podcast: -8dB to -15dB for clearer voice
- Film: -3dB to -6dB for subtle background

Usage:
    from montage_ai.dialogue_ducking import (
        detect_dialogue_segments,
        generate_duck_keyframes,
        DialogueDetector
    )

    detector = DialogueDetector()
    segments = detector.detect(voice_audio_path)
    keyframes = detector.generate_keyframes(segments, music_duration)
"""

import os
import json
import tempfile
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np

from .config import get_settings
from .logger import logger
from .core.cmd_runner import run_command, CommandError
from .ffmpeg_utils import build_ffmpeg_cmd

_settings = get_settings()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DialogueSegment:
    """A detected dialogue/speech segment."""
    start_time: float  # Seconds
    end_time: float  # Seconds
    confidence: float = 1.0  # 0-1 detection confidence
    speaker_id: Optional[str] = None  # For multi-speaker detection
    transcript: Optional[str] = None  # If transcription available

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id,
            "transcript": self.transcript,
        }


@dataclass
class DuckKeyframe:
    """A volume automation keyframe for ducking."""
    time: float  # Seconds
    level_db: float  # Volume level in dB (0 = unity, negative = reduced)
    curve: str = "linear"  # linear, ease_in, ease_out, ease_in_out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "level_db": self.level_db,
            "curve": self.curve,
        }


@dataclass
class DuckingConfig:
    """Configuration for auto-ducking behavior."""
    # Duck level (how much to reduce music during speech)
    duck_level_db: float = -12.0  # dB reduction during speech

    # Timing parameters
    attack_time: float = 0.15  # Seconds to ramp down
    release_time: float = 0.30  # Seconds to ramp back up
    hold_time: float = 0.10  # Minimum time at ducked level

    # Threshold for segment merging
    merge_gap: float = 0.5  # Merge segments closer than this (seconds)

    # Minimum segment duration to trigger ducking
    min_segment_duration: float = 0.3  # Seconds

    # Pre-roll/post-roll (anticipate speech)
    pre_roll: float = 0.05  # Start ducking slightly before speech
    post_roll: float = 0.15  # Hold duck slightly after speech ends

    # Output format
    output_format: str = "keyframes"  # keyframes, sidechain, or both


@dataclass
class DialogueDetectionResult:
    """Complete dialogue detection results."""
    segments: List[DialogueSegment]
    keyframes: List[DuckKeyframe]
    total_speech_duration: float
    speech_percentage: float
    config: DuckingConfig
    method: str  # Detection method used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "keyframes": [k.to_dict() for k in self.keyframes],
            "total_speech_duration": self.total_speech_duration,
            "speech_percentage": self.speech_percentage,
            "method": self.method,
            "config": asdict(self.config),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Detection Methods
# =============================================================================

def _check_silero_available() -> bool:
    """Check if Silero VAD is available."""
    try:
        import torch
        # Try to load silero model
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        return True
    except Exception:
        return False


def _detect_with_silero(audio_path: str, sample_rate: int = 16000) -> List[DialogueSegment]:
    """
    Detect speech using Silero VAD (high accuracy).

    Silero VAD is a pre-trained voice activity detection model that
    achieves high accuracy on diverse audio sources.
    """
    import torch
    import torchaudio

    # Load model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    (get_speech_timestamps, _, read_audio, _, _) = utils

    # Load audio
    wav = read_audio(audio_path, sampling_rate=sample_rate)

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sample_rate,
        threshold=0.5,  # Speech probability threshold
        min_speech_duration_ms=250,  # Minimum speech segment
        min_silence_duration_ms=100,  # Minimum silence between segments
        window_size_samples=512,
        speech_pad_ms=30,
    )

    # Convert to DialogueSegment objects
    segments = []
    for ts in speech_timestamps:
        start_sec = ts['start'] / sample_rate
        end_sec = ts['end'] / sample_rate

        # Calculate confidence from model probability if available
        confidence = ts.get('probability', 1.0)

        segments.append(DialogueSegment(
            start_time=start_sec,
            end_time=end_sec,
            confidence=confidence,
        ))

    logger.info(f"Silero VAD: Detected {len(segments)} speech segments")
    return segments


def _detect_with_ffmpeg(audio_path: str, duration: float) -> List[DialogueSegment]:
    """
    Detect speech using FFmpeg's silencedetect filter (fallback).

    This inverts silence detection: periods that are NOT silent
    are assumed to be speech. Less accurate than ML-based VAD
    but works without additional dependencies.
    """
    # Step 1: Run silencedetect with voice-appropriate threshold
    # Voice threshold: -35dB (louder than typical background noise)
    # Minimum silence: 0.5s (pauses in speech)
    cmd = build_ffmpeg_cmd(
        [
            "-i", audio_path,
            "-af", "silencedetect=n=-35dB:d=0.5",
            "-f", "null", "-"
        ],
        overwrite=False
    )

    try:
        result = run_command(
            cmd,
            capture_output=True,
            timeout=_settings.processing.analysis_timeout,
            check=False
        )
        stderr = result.stderr
    except Exception as e:
        logger.warning(f"FFmpeg silencedetect failed: {e}")
        return []

    # Step 2: Parse silence segments
    silence_segments = []
    current_start = None

    for line in stderr.split('\n'):
        if 'silence_start:' in line:
            try:
                start_str = line.split('silence_start:')[1].strip().split()[0]
                current_start = float(start_str)
            except (ValueError, IndexError):
                pass
        elif 'silence_end:' in line and current_start is not None:
            try:
                end_str = line.split('silence_end:')[1].split('|')[0].strip()
                end_time = float(end_str)
                silence_segments.append((current_start, end_time))
                current_start = None
            except (ValueError, IndexError):
                pass

    # Handle unclosed silence at end
    if current_start is not None:
        silence_segments.append((current_start, duration))

    # Step 3: Invert to get speech segments
    speech_segments = []

    # Start of audio to first silence
    if silence_segments:
        if silence_segments[0][0] > 0.1:  # At least 100ms of speech at start
            speech_segments.append(DialogueSegment(
                start_time=0.0,
                end_time=silence_segments[0][0],
                confidence=0.8,  # Lower confidence for FFmpeg method
            ))
    else:
        # No silence detected = all speech
        speech_segments.append(DialogueSegment(
            start_time=0.0,
            end_time=duration,
            confidence=0.7,
        ))
        return speech_segments

    # Between silences
    for i in range(len(silence_segments) - 1):
        speech_start = silence_segments[i][1]  # End of current silence
        speech_end = silence_segments[i + 1][0]  # Start of next silence

        if speech_end - speech_start > 0.1:  # At least 100ms
            speech_segments.append(DialogueSegment(
                start_time=speech_start,
                end_time=speech_end,
                confidence=0.8,
            ))

    # After last silence to end of audio
    if silence_segments and silence_segments[-1][1] < duration - 0.1:
        speech_segments.append(DialogueSegment(
            start_time=silence_segments[-1][1],
            end_time=duration,
            confidence=0.8,
        ))

    logger.info(f"FFmpeg VAD: Detected {len(speech_segments)} speech segments (inverted silence)")
    return speech_segments


def _detect_with_webrtc(audio_path: str) -> List[DialogueSegment]:
    """
    Detect speech using WebRTC VAD (medium accuracy, fast).

    WebRTC VAD is a lightweight voice activity detector used in
    web conferencing applications.
    """
    try:
        import webrtcvad
        import wave
    except ImportError:
        logger.warning("webrtcvad not installed, falling back to FFmpeg")
        return []

    # WebRTC VAD requires specific audio format
    # Convert to 16kHz mono WAV
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Convert audio to WebRTC-compatible format
        cmd = build_ffmpeg_cmd([
            "-i", audio_path,
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            tmp_path
        ])
        run_command(cmd, check=True, timeout=60)

        # Read WAV file
        with wave.open(tmp_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            audio_data = wf.readframes(wf.getnframes())

        # Initialize VAD
        vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3 (2 = moderate)

        # Process in 30ms frames
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000) * sample_width

        # Detect speech in frames
        speech_frames = []
        frame_index = 0

        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i + frame_size]
            if len(frame) < frame_size:
                break

            is_speech = vad.is_speech(frame, sample_rate)
            speech_frames.append((frame_index * frame_duration_ms / 1000, is_speech))
            frame_index += 1

        # Convert frames to segments
        segments = []
        in_speech = False
        speech_start = 0.0

        for time, is_speech in speech_frames:
            if is_speech and not in_speech:
                # Speech started
                speech_start = time
                in_speech = True
            elif not is_speech and in_speech:
                # Speech ended
                if time - speech_start > 0.1:  # Minimum 100ms
                    segments.append(DialogueSegment(
                        start_time=speech_start,
                        end_time=time,
                        confidence=0.85,
                    ))
                in_speech = False

        # Handle speech at end
        if in_speech:
            segments.append(DialogueSegment(
                start_time=speech_start,
                end_time=speech_frames[-1][0],
                confidence=0.85,
            ))

        logger.info(f"WebRTC VAD: Detected {len(segments)} speech segments")
        return segments

    except Exception as e:
        logger.warning(f"WebRTC VAD failed: {e}")
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# =============================================================================
# Keyframe Generation
# =============================================================================

def _merge_close_segments(
    segments: List[DialogueSegment],
    merge_gap: float
) -> List[DialogueSegment]:
    """Merge segments that are closer together than merge_gap."""
    if len(segments) <= 1:
        return segments

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda s: s.start_time)
    merged = [sorted_segments[0]]

    for segment in sorted_segments[1:]:
        last = merged[-1]
        gap = segment.start_time - last.end_time

        if gap <= merge_gap:
            # Merge: extend last segment to include this one
            merged[-1] = DialogueSegment(
                start_time=last.start_time,
                end_time=segment.end_time,
                confidence=min(last.confidence, segment.confidence),
                speaker_id=last.speaker_id,
            )
        else:
            merged.append(segment)

    return merged


def _generate_keyframes(
    segments: List[DialogueSegment],
    total_duration: float,
    config: DuckingConfig
) -> List[DuckKeyframe]:
    """
    Generate volume automation keyframes for ducking.

    Creates smooth ramps for attack and release to avoid audio pops.
    """
    if not segments:
        return []

    keyframes = []

    # Initial keyframe at start (full volume)
    keyframes.append(DuckKeyframe(time=0.0, level_db=0.0, curve="linear"))

    for segment in segments:
        # Skip segments shorter than minimum duration
        if segment.duration < config.min_segment_duration:
            continue

        # Calculate duck timing with pre/post roll
        duck_start = max(0, segment.start_time - config.pre_roll)
        duck_end = min(total_duration, segment.end_time + config.post_roll)

        # Attack: ramp down to duck level
        attack_start = duck_start - config.attack_time
        if attack_start < 0:
            attack_start = 0

        # Only add attack keyframe if it doesn't overlap with previous release
        if keyframes and keyframes[-1].time < attack_start:
            # Add unity keyframe before attack (if not already there)
            if keyframes[-1].level_db != 0.0:
                keyframes.append(DuckKeyframe(
                    time=attack_start,
                    level_db=0.0,
                    curve="linear"
                ))

        # Duck keyframe (start of speech)
        keyframes.append(DuckKeyframe(
            time=duck_start,
            level_db=config.duck_level_db,
            curve="ease_out"
        ))

        # Hold at duck level (optional explicit keyframe)
        if config.hold_time > 0 and duck_end - duck_start > config.hold_time:
            keyframes.append(DuckKeyframe(
                time=duck_end - config.release_time,
                level_db=config.duck_level_db,
                curve="linear"
            ))

        # Release: ramp back to full volume
        release_end = duck_end + config.release_time
        if release_end > total_duration:
            release_end = total_duration

        keyframes.append(DuckKeyframe(
            time=release_end,
            level_db=0.0,
            curve="ease_in"
        ))

    # Final keyframe at end (ensure full volume)
    if keyframes[-1].time < total_duration:
        keyframes.append(DuckKeyframe(
            time=total_duration,
            level_db=0.0,
            curve="linear"
        ))

    # Remove duplicate/overlapping keyframes
    cleaned = []
    for kf in keyframes:
        if not cleaned or kf.time > cleaned[-1].time + 0.01:  # 10ms minimum gap
            cleaned.append(kf)
        elif abs(kf.level_db) > abs(cleaned[-1].level_db):
            # Keep the more ducked keyframe if overlapping
            cleaned[-1] = kf

    return cleaned


# =============================================================================
# Main Detector Class
# =============================================================================

class DialogueDetector:
    """
    Detects dialogue/speech and generates auto-ducking keyframes.

    Usage:
        detector = DialogueDetector()
        result = detector.analyze(voice_track_path, music_duration=30.0)

        # Use keyframes in FFmpeg
        keyframes = result.keyframes

        # Export for NLE
        detector.export_to_nle(result, output_path, format="resolve")
    """

    def __init__(self, config: Optional[DuckingConfig] = None):
        self.config = config or DuckingConfig()
        self._silero_available = None
        self._webrtc_available = None

    def _check_backends(self) -> None:
        """Check available detection backends."""
        if self._silero_available is None:
            try:
                import torch
                self._silero_available = True
            except ImportError:
                self._silero_available = False

        if self._webrtc_available is None:
            try:
                import webrtcvad
                self._webrtc_available = True
            except ImportError:
                self._webrtc_available = False

    def detect_segments(
        self,
        audio_path: str,
        method: str = "auto"
    ) -> Tuple[List[DialogueSegment], str]:
        """
        Detect speech segments in audio file.

        Args:
            audio_path: Path to audio file (voice track)
            method: Detection method (auto, silero, webrtc, ffmpeg)

        Returns:
            Tuple of (segments, method_used)
        """
        self._check_backends()

        # Get audio duration for FFmpeg fallback
        from .video_metadata import probe_duration
        duration = probe_duration(audio_path)

        segments = []
        method_used = method

        if method == "auto":
            # Try methods in order of quality
            if self._silero_available:
                try:
                    segments = _detect_with_silero(audio_path)
                    method_used = "silero"
                except Exception as e:
                    logger.warning(f"Silero VAD failed: {e}")

            if not segments and self._webrtc_available:
                try:
                    segments = _detect_with_webrtc(audio_path)
                    method_used = "webrtc"
                except Exception as e:
                    logger.warning(f"WebRTC VAD failed: {e}")

            if not segments:
                segments = _detect_with_ffmpeg(audio_path, duration)
                method_used = "ffmpeg"

        elif method == "silero":
            segments = _detect_with_silero(audio_path)
            method_used = "silero"

        elif method == "webrtc":
            segments = _detect_with_webrtc(audio_path)
            method_used = "webrtc"

        else:  # ffmpeg
            segments = _detect_with_ffmpeg(audio_path, duration)
            method_used = "ffmpeg"

        return segments, method_used

    def analyze(
        self,
        audio_path: str,
        music_duration: Optional[float] = None,
        method: str = "auto"
    ) -> DialogueDetectionResult:
        """
        Full dialogue analysis with keyframe generation.

        Args:
            audio_path: Path to voice/dialogue audio file
            music_duration: Total duration of music track (for keyframes)
            method: Detection method

        Returns:
            DialogueDetectionResult with segments and keyframes
        """
        logger.info(f"Analyzing dialogue in {os.path.basename(audio_path)}...")

        # Detect segments
        segments, method_used = self.detect_segments(audio_path, method)

        # Merge close segments
        merged = _merge_close_segments(segments, self.config.merge_gap)
        logger.info(f"Merged {len(segments)} segments into {len(merged)}")

        # Get audio duration
        from .video_metadata import probe_duration
        audio_duration = probe_duration(audio_path)
        duration = music_duration or audio_duration

        # Generate keyframes
        keyframes = _generate_keyframes(merged, duration, self.config)
        logger.info(f"Generated {len(keyframes)} ducking keyframes")

        # Calculate stats
        total_speech = sum(s.duration for s in merged)
        speech_pct = (total_speech / audio_duration * 100) if audio_duration > 0 else 0

        return DialogueDetectionResult(
            segments=merged,
            keyframes=keyframes,
            total_speech_duration=total_speech,
            speech_percentage=speech_pct,
            config=self.config,
            method=method_used,
        )

    def generate_ffmpeg_filter(self, result: DialogueDetectionResult) -> str:
        """
        Generate FFmpeg volume filter with keyframe automation.

        Returns a filter string like:
        volume='if(between(t,0,1),1,if(between(t,1,2),0.25,1))'
        """
        if not result.keyframes:
            return "volume=1"

        # Build conditional expression for FFmpeg
        # Convert dB to linear: 10^(dB/20)
        conditions = []

        for i, kf in enumerate(result.keyframes[:-1]):
            next_kf = result.keyframes[i + 1]

            # Calculate linear volume
            vol_linear = 10 ** (kf.level_db / 20)

            # Create condition for this segment
            conditions.append(
                f"between(t,{kf.time:.3f},{next_kf.time:.3f})*{vol_linear:.4f}"
            )

        # Combine with addition (overlapping segments sum)
        filter_str = f"volume='{'+'.join(conditions)}'"

        return filter_str

    def export_to_json(self, result: DialogueDetectionResult, output_path: str) -> str:
        """Export results to JSON file."""
        with open(output_path, 'w') as f:
            f.write(result.to_json())
        logger.info(f"Exported dialogue detection to {output_path}")
        return output_path

    def export_to_nle(
        self,
        result: DialogueDetectionResult,
        output_path: str,
        format: str = "resolve"
    ) -> str:
        """
        Export keyframes to NLE-compatible format.

        Formats:
        - resolve: DaVinci Resolve automation (.txt)
        - premiere: Premiere Pro keyframes (.xml)
        - fcpx: Final Cut Pro X automation (.fcpxml)
        - csv: Universal CSV format
        """
        if format == "csv":
            return self._export_csv(result, output_path)
        elif format == "resolve":
            return self._export_resolve(result, output_path)
        elif format == "premiere":
            return self._export_premiere_xml(result, output_path)
        else:
            # Default to CSV
            return self._export_csv(result, output_path)

    def _export_csv(self, result: DialogueDetectionResult, output_path: str) -> str:
        """Export as CSV (universal format)."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time (sec)", "Level (dB)", "Level (linear)", "Curve"])

            for kf in result.keyframes:
                linear = 10 ** (kf.level_db / 20)
                writer.writerow([
                    f"{kf.time:.3f}",
                    f"{kf.level_db:.1f}",
                    f"{linear:.4f}",
                    kf.curve
                ])

        logger.info(f"Exported {len(result.keyframes)} keyframes to {output_path}")
        return output_path

    def _export_resolve(self, result: DialogueDetectionResult, output_path: str) -> str:
        """Export for DaVinci Resolve (text format for manual input)."""
        with open(output_path, 'w') as f:
            f.write("# DaVinci Resolve Volume Automation\n")
            f.write("# Copy these values to Fairlight > Volume automation\n")
            f.write("# Time (SMPTE) | Level (dB)\n")
            f.write("#" + "=" * 40 + "\n\n")

            fps = 30.0  # Assume 30fps
            for kf in result.keyframes:
                # Convert to SMPTE timecode
                hours = int(kf.time // 3600)
                minutes = int((kf.time % 3600) // 60)
                seconds = int(kf.time % 60)
                frames = int((kf.time % 1) * fps)

                tc = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
                f.write(f"{tc}  |  {kf.level_db:+.1f} dB\n")

            f.write(f"\n# Total keyframes: {len(result.keyframes)}\n")
            f.write(f"# Speech duration: {result.total_speech_duration:.1f}s ({result.speech_percentage:.1f}%)\n")

        return output_path

    def _export_premiere_xml(self, result: DialogueDetectionResult, output_path: str) -> str:
        """Export as Premiere Pro compatible XML."""
        fps = 30.0

        with open(output_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<keyframes>\n')
            f.write('  <parameter name="Volume">\n')

            for kf in result.keyframes:
                frame = int(kf.time * fps)
                linear = 10 ** (kf.level_db / 20)

                # Premiere uses 0-100 scale for volume
                premiere_vol = linear * 100

                f.write(f'    <keyframe time="{frame}" value="{premiere_vol:.2f}"/>\n')

            f.write('  </parameter>\n')
            f.write('</keyframes>\n')

        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_dialogue_segments(
    audio_path: str,
    method: str = "auto"
) -> List[DialogueSegment]:
    """
    Detect speech/dialogue segments in audio.

    Args:
        audio_path: Path to audio file
        method: Detection method (auto, silero, webrtc, ffmpeg)

    Returns:
        List of DialogueSegment objects
    """
    detector = DialogueDetector()
    segments, _ = detector.detect_segments(audio_path, method)
    return segments


def generate_duck_keyframes(
    segments: List[DialogueSegment],
    total_duration: float,
    duck_level_db: float = -12.0,
    attack_time: float = 0.15,
    release_time: float = 0.30
) -> List[DuckKeyframe]:
    """
    Generate volume keyframes for music ducking.

    Args:
        segments: List of detected speech segments
        total_duration: Total duration of music track
        duck_level_db: Volume reduction in dB (negative)
        attack_time: Ramp down time in seconds
        release_time: Ramp up time in seconds

    Returns:
        List of DuckKeyframe objects
    """
    config = DuckingConfig(
        duck_level_db=duck_level_db,
        attack_time=attack_time,
        release_time=release_time,
    )
    return _generate_keyframes(segments, total_duration, config)


def apply_ducking_to_audio(
    music_path: str,
    voice_path: str,
    output_path: str,
    duck_level_db: float = -12.0,
    method: str = "auto"
) -> str:
    """
    Apply automatic ducking to music track based on voice detection.

    This is the main convenience function for one-shot ducking.

    Args:
        music_path: Path to music/background audio
        voice_path: Path to voice/dialogue audio
        output_path: Path for ducked output
        duck_level_db: Volume reduction in dB
        method: VAD method to use

    Returns:
        Path to ducked audio file
    """
    from .video_metadata import probe_duration

    # Detect dialogue
    config = DuckingConfig(duck_level_db=duck_level_db)
    detector = DialogueDetector(config)

    music_duration = probe_duration(music_path)
    result = detector.analyze(voice_path, music_duration, method)

    if not result.keyframes:
        logger.warning("No speech detected, returning original audio")
        import shutil
        shutil.copy(music_path, output_path)
        return output_path

    # Generate FFmpeg filter
    vol_filter = detector.generate_ffmpeg_filter(result)

    # Apply ducking with FFmpeg
    cmd = build_ffmpeg_cmd([
        "-i", music_path,
        "-af", vol_filter,
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ])

    try:
        run_command(cmd, check=True, timeout=120)
        logger.info(f"Applied ducking to {os.path.basename(output_path)}")
        return output_path
    except CommandError as e:
        logger.error(f"Ducking failed: {e}")
        raise


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "DialogueSegment",
    "DuckKeyframe",
    "DuckingConfig",
    "DialogueDetectionResult",
    # Main class
    "DialogueDetector",
    # Convenience functions
    "detect_dialogue_segments",
    "generate_duck_keyframes",
    "apply_ducking_to_audio",
]
