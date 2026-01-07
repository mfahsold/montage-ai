"""
Threshold Configuration Module for Montage AI

Centralized configuration for all detection thresholds and sensitivity parameters.
Enables hardware-agnostic, environment-tunable detection sensitivity across:
- Scene detection (visual change sensitivity)
- Speech detection (voice probability)
- Silence detection (audio silence level)
- Face detection confidence (MediaPipe face recognition)
- Audio ducking filters (sidechain compression)
- Music section detection (minimum duration)

Usage:
    from montage_ai.config_thresholds import ThresholdConfig

    # Scene detection
    scene_threshold = ThresholdConfig.scene_detection()  # 27-35 range
    
    # Speech detection
    speech_threshold = ThresholdConfig.speech_detection()  # 0.3-0.7 range
    
    # Face detection confidence
    face_confidence = ThresholdConfig.face_detection_confidence()  # 0.5-0.8
    
    # Print all thresholds
    ThresholdConfig.print_config()
    
Environment Variables:
    SCENE_THRESHOLD: Scene detection sensitivity (default: 30.0)
    SPEECH_THRESHOLD: Speech probability threshold (default: 0.5)
    SILENCE_THRESHOLD: Silence detection level in dB (default: -35)
    SILENCE_DURATION: Minimum silence duration in seconds (default: 0.5)
    FACE_CONFIDENCE: Face detection confidence (default: 0.6)
    DUCKING_CORE_THRESHOLD: Core ducking threshold (default: 0.1)
    DUCKING_SOFT_THRESHOLD: Soft ducking threshold (default: 0.03)
    MUSIC_MIN_DURATION: Minimum music section duration (default: 5.0)
    SPEECH_MIN_DURATION: Minimum speech segment (default: 250ms)
    SPEECH_MIN_SILENCE: Minimum silence between speech segments (default: 100ms)
"""

from typing import Tuple, Dict, Any
from .config import get_settings


class ThresholdConfig:
    """
    Single source of truth for all detection thresholds and sensitivity parameters.
    
    Design Principle:
    - Default values match current production behavior (backward compatible)
    - Environment variables allow per-deployment tuning without code changes
    - Ranges documented for each parameter with typical use cases
    """

    @staticmethod
    def scene_detection(name: str = "SCENE_THRESHOLD") -> float:
        """
        Scene detection visual change threshold.
        
        Range: 15.0 (very sensitive) to 40.0 (insensitive)
        Default: 30.0 (balanced for typical video)
        
        Rationale:
        - Lower values (15-20): Detect more subtle scene changes, creates more cuts
        - Mid values (25-35): Standard usage, balanced for most content
        - Higher values (35-40): Less cuts, good for static content
        
        Use cases:
        - Action video: 20-25 (detect more action transitions)
        - Documentary: 30-35 (fewer cuts, more stable)
        - Music video: 15-25 (rapid scene changes expected)
        
        Args:
            name: Environment variable name override
            
        Returns:
            float: Threshold value for scenedetect.ContentDetector
        """
        return float(get_settings().thresholds.scene_threshold)

    @staticmethod
    def speech_detection(name: str = "SPEECH_THRESHOLD") -> float:
        """
        Speech probability threshold for pyannote.audio VAD (Voice Activity Detection).
        
        Range: 0.3 (very permissive) to 0.8 (very strict)
        Default: 0.5 (balanced)
        
        Rationale:
        - 0.3-0.4: Include all speech, more false positives, better for quiet speakers
        - 0.5: Standard usage, balanced for typical dialogue
        - 0.6-0.7: More selective, fewer false positives, may miss quiet speech
        - 0.8: Very strict, only include high-confidence speech
        
        Use cases:
        - Podcast (clear speech): 0.6-0.7
        - Whisper/quiet dialogue: 0.3-0.4
        - Standard video: 0.5
        - Noisy environment: 0.4-0.5
        
        Args:
            name: Environment variable name override
            
        Returns:
            float: Probability threshold (0.0-1.0)
        """
        return float(get_settings().thresholds.speech_threshold)

    @staticmethod
    def silence_detection(name: str = "SILENCE_THRESHOLD") -> str:
        """
        Silence detection level in dB for ffmpeg silencedetect filter.
        
        Range: -50dB (very loud) to -10dB (very quiet)
        Default: -35dB (typical background noise level)
        
        Rationale:
        - -50dB: Only detect absolute silence
        - -35dB: Detect pause in speech (over background noise)
        - -20dB: Detect louder silence
        - -10dB: Detect very subtle silence
        
        Context: Silence at -35dB is standard for detecting speech pauses
        while ignoring typical room noise (usually -40 to -45dB).
        
        Args:
            name: Environment variable name override
            
        Returns:
            str: dB level for ffmpeg (e.g., "-35")
        """
        return str(get_settings().thresholds.silence_level_db)

    @staticmethod
    def silence_duration(name: str = "SILENCE_DURATION") -> float:
        """
        Minimum silence duration in seconds to register as a silence event.
        
        Range: 0.1 (very sensitive) to 2.0 (less sensitive)
        Default: 0.5 (balanced for speech pauses)
        
        Rationale:
        - 0.1-0.2: Detect very short pauses, more granular speech analysis
        - 0.5: Standard for speech pause detection
        - 1.0-2.0: Only detect long silences, less granular
        
        Use cases:
        - Lip-sync detection: 0.2-0.3
        - Speech pause detection: 0.5
        - Section boundary detection: 1.0-2.0
        
        Args:
            name: Environment variable name override
            
        Returns:
            float: Duration in seconds
        """
        return float(get_settings().thresholds.silence_duration_s)

    @staticmethod
    def face_detection_confidence(name: str = "FACE_CONFIDENCE") -> float:
        """
        Face detection confidence threshold for MediaPipe face detection.
        
        Range: 0.3 (very permissive) to 0.95 (very strict)
        Default: 0.6 (balanced)
        
        Rationale:
        - 0.3-0.4: Detect partial faces, low occlusion tolerance
        - 0.5-0.6: Standard usage, balanced for clear faces
        - 0.7-0.8: Higher quality faces, fewer false positives
        - 0.9+: Only very clear, centered faces
        
        Use cases:
        - Close-up interviews: 0.7-0.8
        - General B-roll: 0.6
        - Crowd scenes: 0.5
        - Artistic/profile shots: 0.4-0.5
        
        Args:
            name: Environment variable name override
            
        Returns:
            float: Confidence threshold (0.0-1.0)
        """
        return float(get_settings().thresholds.face_confidence)

    @staticmethod
    def ducking_core_threshold(name: str = "DUCKING_CORE_THRESHOLD") -> float:
        """
        Sidechain compression threshold for core (strong) ducking.
        
        Range: 0.01 (very aggressive) to 0.5 (gentle)
        Default: 0.1 (standard ducking)
        
        Rationale:
        - 0.01-0.05: Very aggressive, strong voice elevation
        - 0.1: Standard, clear voice over background
        - 0.2-0.5: Gentle, preserves music, subtle voice boost
        
        Context: This is used for sidechaincompress audio filter.
        Lower threshold = more aggressive ducking = louder voice.
        
        Use cases:
        - Action movie narration: 0.05-0.1
        - Standard dialogue: 0.1-0.15
        - Voiceover music: 0.15-0.2
        - Ambient dialogue: 0.2+
        
        Args:
            name: Environment variable name override
            
        Returns:
            float: Threshold value (0.0-1.0)
        """
        return float(get_settings().thresholds.ducking_core_threshold)

    @staticmethod
    def ducking_soft_threshold(name: str = "DUCKING_SOFT_THRESHOLD") -> float:
        """
        Sidechain compression threshold for soft (gentle) ducking.
        
        Range: 0.01 (aggressive) to 0.1 (very gentle)
        Default: 0.03 (subtle ducking)
        
        Rationale:
        - 0.01: Very subtle, barely noticeable
        - 0.03: Standard soft ducking, slight voice boost
        - 0.05-0.1: Very gentle, preserves music ambience
        
        Context: This is for background music reduction without obvious cutting.
        Used when music should remain prominent but voice should be clear.
        
        Use cases:
        - Background music with dialogue: 0.03-0.05
        - Subtle narration over music: 0.05-0.1
        - Music-forward content: 0.08-0.1
        
        Args:
            name: Environment variable name override
            
        Returns:
            float: Threshold value (0.0-1.0)
        """
        return float(get_settings().thresholds.ducking_soft_threshold)

    @staticmethod
    def music_min_duration(name: str = "MUSIC_MIN_DURATION") -> float:
        """
        Minimum music section duration in seconds.
        
        Range: 1.0 (very short) to 15.0 (long sections)
        Default: 5.0 (balanced)
        
        Rationale:
        - 1.0-2.0: Very granular analysis, may include jingles/stings
        - 5.0: Standard, captures verse/chorus sections
        - 10.0+: Long sections only, good for orchestral music
        
        Use cases:
        - Pop/hip-hop: 3.0-5.0
        - Orchestral: 8.0-12.0
        - Ambient: 10.0-15.0
        - Music video: 5.0-8.0
        
        Args:
            name: Environment variable name override
            
        Returns:
            float: Duration in seconds
        """
        return float(get_settings().thresholds.music_min_duration_s)

    @staticmethod
    def speech_min_duration(name: str = "SPEECH_MIN_DURATION") -> int:
        """
        Minimum speech segment duration in milliseconds (pyannote.audio parameter).
        
        Range: 100 (very short) to 1000 (long words)
        Default: 250 (balanced for typical words)
        
        Rationale:
        - 100-150: Detect single-syllable words
        - 250: Standard for typical words/short phrases
        - 500+: Only longer utterances
        
        Use cases:
        - Clear dialogue: 200-300
        - Mumbling/unclear: 100-200
        - Long speeches: 300-500
        
        Args:
            name: Environment variable name override
            
        Returns:
            int: Duration in milliseconds
        """
        return int(get_settings().thresholds.speech_min_duration_ms)

    @staticmethod
    def speech_min_silence(name: str = "SPEECH_MIN_SILENCE") -> int:
        """
        Minimum silence duration between speech segments in milliseconds.
        
        Range: 50 (tight, sensitive to noise) to 500 (loose)
        Default: 100 (balanced)
        
        Rationale:
        - 50: Separate on very short pauses, may fragment speech
        - 100: Standard for natural speech gaps
        - 200-500: Larger gaps, treats brief pauses as part of segment
        
        Use cases:
        - Stuttering detection: 50-100
        - Natural dialogue: 100-150
        - Continuous speech: 200+
        
        Args:
            name: Environment variable name override
            
        Returns:
            int: Duration in milliseconds
        """
        return int(get_settings().thresholds.speech_min_silence_ms)

    @staticmethod
    def get(operation: str, default: float) -> float:
        """
        Get threshold value for any operation by name.
        
        Generic threshold getter for custom operations not covered by specific methods.
        Convention: environment variable name is uppercase version of operation name.
        
        Example:
            custom_threshold = ThresholdConfig.get("my_custom_threshold", 0.5)
            # Looks for MY_CUSTOM_THRESHOLD env var, defaults to 0.5
        
        Args:
            operation: Operation name (e.g., "my_custom_threshold")
            default: Default value if not found in environment
            
        Returns:
            float: Threshold value
        """
        # Generic passthrough is no longer supported via settings; return default
        # Callers should prefer explicit fields on settings.thresholds
        return float(default)

    @staticmethod
    def print_config() -> None:
        """
        Print all threshold configuration values to stdout.
        
        Useful for debugging and configuration verification.
        Shows current values including any environment variable overrides.
        """
        print("\n" + "="*70)
        print("THRESHOLD CONFIGURATION")
        print("="*70)
        
        print(f"\nScene Detection:")
        print(f"  Scene Threshold:           {ThresholdConfig.scene_detection():.1f}")
        print(f"    Range: 15.0-40.0 (lower = more sensitive)")
        
        print(f"\nSpeech Detection:")
        print(f"  Speech Threshold:          {ThresholdConfig.speech_detection():.2f}")
        print(f"  Min Speech Duration:       {ThresholdConfig.speech_min_duration()} ms")
        print(f"  Min Silence Between:       {ThresholdConfig.speech_min_silence()} ms")
        print(f"    Range: 0.3-0.8 (lower = more permissive)")
        
        print(f"\nSilence Detection:")
        print(f"  Silence Level:             {ThresholdConfig.silence_detection()} dB")
        print(f"  Silence Duration:          {ThresholdConfig.silence_duration():.1f} s")
        print(f"    Range: -50dB to -10dB (more negative = quieter)")
        
        print(f"\nFace Detection:")
        print(f"  Face Confidence:           {ThresholdConfig.face_detection_confidence():.2f}")
        print(f"    Range: 0.3-0.95 (higher = more strict)")
        
        print(f"\nAudio Ducking:")
        print(f"  Ducking Core Threshold:    {ThresholdConfig.ducking_core_threshold():.3f}")
        print(f"  Ducking Soft Threshold:    {ThresholdConfig.ducking_soft_threshold():.3f}")
        print(f"    Range: 0.01-0.5 (lower = more aggressive)")
        
        print(f"\nMusic Analysis:")
        print(f"  Min Music Duration:        {ThresholdConfig.music_min_duration():.1f} s")
        print(f"    Range: 1.0-15.0 seconds")
        
        print("\nEnvironment Variables:")
        print(f"  SCENE_THRESHOLD, SPEECH_THRESHOLD, SILENCE_THRESHOLD")
        print(f"  SILENCE_DURATION, FACE_CONFIDENCE")
        print(f"  DUCKING_CORE_THRESHOLD, DUCKING_SOFT_THRESHOLD")
        print(f"  MUSIC_MIN_DURATION, SPEECH_MIN_DURATION, SPEECH_MIN_SILENCE")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Print current configuration when run directly
    ThresholdConfig.print_config()
