"""
Transcriber - Audio Transcription via cgpu (OpenAI Whisper)

Thin wrapper around TranscribeJob for backward compatibility.
All heavy lifting is now in cgpu_jobs/transcribe.py.

Usage:
    from montage_ai.transcriber import Transcriber

    transcriber = Transcriber()
    if transcriber.is_available():
        srt_path = transcriber.transcribe("audio.wav", output_format="srt")
"""

from typing import Optional

from .cgpu_utils import is_cgpu_available
from .cgpu_jobs import TranscribeJob


class Transcriber:
    """Audio transcription via Whisper on cgpu."""

    def __init__(self, model: str = "medium"):
        """
        Initialize the Transcriber.

        Args:
            model: Whisper model size (tiny, base, small, medium, large)
        """
        self.model = model

    def is_available(self) -> bool:
        """Check if cgpu is available for transcription."""
        return is_cgpu_available()

    def transcribe(
        self,
        audio_path: str,
        output_format: str = "srt",
        language: Optional[str] = None,
    ) -> Optional[str]:
        """
        Transcribe an audio file using Whisper on cgpu.

        Args:
            audio_path: Path to local audio file
            output_format: Output format (srt, vtt, txt, json)
            language: Optional language code (e.g., "en", "de")

        Returns:
            Path to the generated subtitle file, or None if failed.
        """
        job = TranscribeJob(
            audio_path=audio_path,
            model=self.model,
            output_format=output_format,
            language=language,
        )
        result = job.execute()
        return result.output_path if result.success else None


# =============================================================================
# Convenience function
# =============================================================================
def transcribe_audio(
    audio_path: str,
    model: str = "medium",
    output_format: str = "srt",
    language: Optional[str] = None,
) -> Optional[str]:
    """
    Quick transcription helper.

    Args:
        audio_path: Path to audio/video file
        model: Whisper model size
        output_format: Output format (srt, vtt, txt, json)
        language: Optional language code

    Returns:
        Path to generated subtitle file, or None if failed
    """
    transcriber = Transcriber(model=model)
    return transcriber.transcribe(audio_path, output_format, language)


# =============================================================================
# CLI Interface (KISS)
# =============================================================================
if __name__ == "__main__":
    import sys

    def print_usage():
        print("Transcriber - Audio Transcription via cgpu")
        print()
        print("Usage:")
        print("  python -m montage_ai.transcriber <audio_file> [format] [model]")
        print()
        print("Examples:")
        print("  python -m montage_ai.transcriber interview.wav")
        print("  python -m montage_ai.transcriber audio.mp3 vtt")
        print("  python -m montage_ai.transcriber video.mp4 srt large")
        print()
        print("Formats: srt, vtt, txt, json (default: srt)")
        print("Models: tiny, base, small, medium, large (default: medium)")

    def main():
        if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
            print_usage()
            sys.exit(0)

        if not is_cgpu_available():
            print("❌ cgpu not available")
            print("   Set CGPU_ENABLED=true and ensure cgpu is installed")
            sys.exit(1)

        audio_file = sys.argv[1]
        output_format = sys.argv[2] if len(sys.argv) > 2 else "srt"
        model = sys.argv[3] if len(sys.argv) > 3 else "medium"

        print(f"Input:  {audio_file}")
        print(f"Model:  {model}")
        print(f"Format: {output_format}")
        print()

        result = transcribe_audio(audio_file, model=model, output_format=output_format)

        if result:
            print()
            print(f"✅ Success: {result}")
            sys.exit(0)
        else:
            print()
            print("❌ Transcription failed")
            sys.exit(1)

    main()
