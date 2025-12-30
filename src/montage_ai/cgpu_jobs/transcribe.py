"""
TranscribeJob - Audio transcription via OpenAI Whisper on cgpu.

Converts audio/video files to subtitle files (.srt, .vtt, .txt, .json).

Usage:
    job = TranscribeJob(
        audio_path="/data/audio.wav",
        model="medium",
        output_format="srt"
    )
    result = job.execute()
    print(result.output_path)  # /data/audio.srt
"""

import os
from pathlib import Path
from typing import List, Optional

from ..logger import logger
from .base import CGPUJob, JobResult
from ..cgpu_utils import run_cgpu_command, copy_to_remote, download_via_base64


class TranscribeJob(CGPUJob):
    """
    Whisper transcription job for cgpu.

    Attributes:
        timeout: 600 seconds (10 minutes) - long audio may take time
        job_type: "transcribe"
    """

    timeout: int = 600
    max_retries: int = 2
    job_type: str = "transcribe"

    # Valid Whisper models
    VALID_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    VALID_FORMATS = ["srt", "vtt", "txt", "json"]

    def __init__(
        self,
        audio_path: str,
        model: str = "medium",
        output_format: str = "srt",
        language: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize transcription job.

        Args:
            audio_path: Path to audio/video file
            model: Whisper model size (tiny, base, small, medium, large)
            output_format: Output format (srt, vtt, txt, json)
            language: Optional language code (e.g., "en", "de")
            output_dir: Output directory (default: same as input)
        """
        super().__init__()
        self.audio_path = Path(audio_path).resolve()
        self.model = model if model in self.VALID_MODELS else "medium"
        self.output_format = output_format if output_format in self.VALID_FORMATS else "srt"
        self.language = language
        self.output_dir = Path(output_dir) if output_dir else self.audio_path.parent

        # Will be set after transcription
        self._output_path: Optional[Path] = None

    def prepare_local(self) -> bool:
        """Validate audio file exists."""
        if not self.audio_path.exists():
            self._error = f"Audio file not found: {self.audio_path}"
            return False

        # Check file size (warn if large)
        size_mb = self.audio_path.stat().st_size / (1024 * 1024)
        if size_mb > 500:
            logger.warning(f"Large file ({size_mb:.1f} MB) - transcription may take a while")

        return True

    def get_requirements(self) -> List[str]:
        """Whisper requires openai-whisper package."""
        return ["openai-whisper"]

    def upload(self) -> bool:
        """Upload audio file to remote."""
        remote_path = f"{self.remote_work_dir}/{self.audio_path.name}"

        logger.info(f"Uploading {self.audio_path.name}...")
        if not copy_to_remote(str(self.audio_path), remote_path):
            self._error = "Failed to upload audio file"
            return False

        return True

    def run_remote(self) -> bool:
        """Run Whisper transcription on cgpu."""
        audio_name = self.audio_path.name

        # Build whisper command
        cmd_parts = [
            f"cd {self.remote_work_dir}",
            f"whisper '{audio_name}'",
            f"--model {self.model}",
            f"--output_format {self.output_format}",
            "--output_dir ."
        ]

        if self.language:
            cmd_parts.append(f"--language {self.language}")

        cmd = " && ".join([cmd_parts[0], " ".join(cmd_parts[1:])])

        logger.info(f"Running Whisper ({self.model})...")
        success, stdout, stderr = run_cgpu_command(cmd, timeout=self.timeout)

        if not success:
            self._error = f"Whisper failed: {stderr or stdout}"
            return False

        return True

    def download(self) -> JobResult:
        """Download transcription result."""
        # Whisper outputs: <stem>.<format>
        output_filename = f"{self.audio_path.stem}.{self.output_format}"
        remote_output = f"{self.remote_work_dir}/{output_filename}"
        local_output = self.output_dir / output_filename

        logger.info(f"Downloading {output_filename}...")

        if download_via_base64(remote_output, str(local_output)):
            self._output_path = local_output
            return JobResult(
                success=True,
                output_path=str(local_output),
                metadata={
                    "model": self.model,
                    "format": self.output_format,
                    "language": self.language,
                }
            )
        else:
            return JobResult(
                success=False,
                error="Failed to download transcription"
            )

    def expected_output_path(self) -> Optional[Path]:
        """Expected output path for idempotent reuse."""
        return self.output_dir / f"{self.audio_path.stem}.{self.output_format}"

    @property
    def output_path(self) -> Optional[Path]:
        """Path to output file (available after successful execution)."""
        return self._output_path
