"""
VoiceIsolationJob - Audio voice isolation via demucs on cgpu.

Separates voice from background audio (music, noise, ambience).
Uses Meta's demucs model for high-quality stem separation.

Usage:
    job = VoiceIsolationJob(audio_path="/data/audio.wav")
    result = job.execute()
    print(result.output_path)  # /data/audio_vocals.wav
"""

import os
from pathlib import Path
from typing import List, Optional

from .base import CGPUJob, JobResult
from ..cgpu_utils import run_cgpu_command, copy_to_remote, download_via_base64


class VoiceIsolationJob(CGPUJob):
    """
    Voice isolation job using demucs on cgpu.

    Demucs separates audio into stems:
    - vocals: Isolated voice
    - drums: Percussion
    - bass: Bass instruments
    - other: Everything else (guitars, synths, etc.)

    For voice isolation, we only need the vocals stem.

    Attributes:
        timeout: 900 seconds (15 minutes) - separation can be slow
        job_type: "voice_isolation"
    """

    timeout: int = 900
    max_retries: int = 2
    job_type: str = "voice_isolation"

    # Available demucs models (quality vs speed tradeoff)
    MODELS = {
        "htdemucs": "Best quality, slower (default)",
        "htdemucs_ft": "Fine-tuned, highest quality, slowest",
        "mdx_extra": "Good quality, faster",
        "mdx": "Fastest, lower quality",
    }

    # Output stems
    STEMS = ["vocals", "drums", "bass", "other"]

    def __init__(
        self,
        audio_path: str,
        model: str = "htdemucs",
        output_dir: Optional[str] = None,
        keep_all_stems: bool = False,
        two_stems: bool = True,
    ):
        """
        Initialize voice isolation job.

        Args:
            audio_path: Path to audio/video file
            model: Demucs model (htdemucs, htdemucs_ft, mdx_extra, mdx)
            output_dir: Output directory (default: same as input)
            keep_all_stems: Keep all stems, not just vocals
            two_stems: Use two-stem mode (vocals + accompaniment) - faster
        """
        super().__init__()
        self.audio_path = Path(audio_path).resolve()
        self.model = model if model in self.MODELS else "htdemucs"
        self.output_dir = Path(output_dir) if output_dir else self.audio_path.parent
        self.keep_all_stems = keep_all_stems
        self.two_stems = two_stems

        # Will be set after processing
        self._output_paths: dict = {}

    def prepare_local(self) -> bool:
        """Validate audio file exists."""
        if not self.audio_path.exists():
            self._error = f"Audio file not found: {self.audio_path}"
            return False

        # Check file size
        size_mb = self.audio_path.stat().st_size / (1024 * 1024)
        if size_mb > 500:
            print(f"   ⚠️ Large file ({size_mb:.1f} MB) - processing may take a while")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        return True

    def get_requirements(self) -> List[str]:
        """Demucs requirements."""
        return ["demucs", "torch", "torchaudio"]

    def upload(self) -> bool:
        """Upload audio file to remote."""
        remote_path = f"{self.remote_work_dir}/{self.audio_path.name}"

        print(f"   ⬆️ Uploading {self.audio_path.name}...")
        if not copy_to_remote(str(self.audio_path), remote_path):
            self._error = "Failed to upload audio file"
            return False

        return True

    def run_remote(self) -> bool:
        """Run demucs separation on cgpu."""
        audio_name = self.audio_path.name

        # Build demucs command
        cmd_parts = [
            f"cd {self.remote_work_dir}",
            "&&",
            "python", "-m", "demucs",
            f"--name {self.model}",
            "-o", ".",  # Output to work dir
        ]

        # Two-stem mode is faster (vocals + no_vocals)
        if self.two_stems:
            cmd_parts.append("--two-stems vocals")

        cmd_parts.append(f"'{audio_name}'")

        cmd = " ".join(cmd_parts)

        print(f"   🎤 Running demucs ({self.model})...")
        print(f"   ⏳ This may take several minutes...")

        success, stdout, stderr = run_cgpu_command(cmd, timeout=self.timeout)

        if not success:
            # Check for common errors
            if "CUDA" in (stderr or ""):
                self._error = f"GPU error: {stderr}"
            elif "memory" in (stderr or "").lower():
                self._error = f"Out of memory - try mdx model instead"
            else:
                self._error = f"Demucs failed: {stderr or stdout}"
            return False

        return True

    def download(self) -> JobResult:
        """Download isolated vocals (and optionally other stems)."""
        stem = self.audio_path.stem

        # Demucs output structure: <model>/<track>/<stem>.wav
        # With two-stems: vocals.wav, no_vocals.wav
        # Without: vocals.wav, drums.wav, bass.wav, other.wav

        if self.two_stems:
            stems_to_download = ["vocals", "no_vocals"] if self.keep_all_stems else ["vocals"]
        else:
            stems_to_download = self.STEMS if self.keep_all_stems else ["vocals"]

        downloaded = {}
        primary_output = None

        for stem_name in stems_to_download:
            remote_stem = f"{self.remote_work_dir}/{self.model}/{stem}/{stem_name}.wav"
            local_stem = self.output_dir / f"{stem}_{stem_name}.wav"

            print(f"   ⬇️ Downloading {stem_name}...")

            if download_via_base64(remote_stem, str(local_stem)):
                downloaded[stem_name] = str(local_stem)
                if stem_name == "vocals":
                    primary_output = str(local_stem)
            else:
                print(f"   ⚠️ Could not download {stem_name}")

        if not primary_output:
            return JobResult(
                success=False,
                error="Failed to download vocals stem"
            )

        self._output_paths = downloaded

        return JobResult(
            success=True,
            output_path=primary_output,
            metadata={
                "model": self.model,
                "stems": downloaded,
                "two_stems": self.two_stems,
            }
        )

    @property
    def vocals_path(self) -> Optional[str]:
        """Path to isolated vocals (available after successful execution)."""
        return self._output_paths.get("vocals")

    @property
    def accompaniment_path(self) -> Optional[str]:
        """Path to accompaniment/no_vocals (available after successful execution)."""
        return self._output_paths.get("no_vocals")


# =============================================================================
# DeepFilterNet Alternative (lighter weight noise reduction)
# =============================================================================

class NoiseReductionJob(CGPUJob):
    """
    Noise reduction job using DeepFilterNet on cgpu.

    Unlike demucs (full stem separation), DeepFilterNet focuses on:
    - Removing background noise
    - Enhancing speech clarity
    - Preserving voice naturalness

    Better for: Podcasts, interviews, vlogs with noise
    Faster than: Full demucs separation
    """

    timeout: int = 300  # 5 minutes - faster than demucs
    max_retries: int = 2
    job_type: str = "noise_reduction"

    def __init__(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        attenuation_limit: int = 100,
    ):
        """
        Initialize noise reduction job.

        Args:
            audio_path: Path to audio/video file
            output_dir: Output directory (default: same as input)
            attenuation_limit: Noise attenuation in dB (0-100, default: 100)
        """
        super().__init__()
        self.audio_path = Path(audio_path).resolve()
        self.output_dir = Path(output_dir) if output_dir else self.audio_path.parent
        self.attenuation_limit = max(0, min(100, attenuation_limit))

        self._output_path: Optional[Path] = None

    def prepare_local(self) -> bool:
        """Validate audio file exists."""
        if not self.audio_path.exists():
            self._error = f"Audio file not found: {self.audio_path}"
            return False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def get_requirements(self) -> List[str]:
        """DeepFilterNet requirements."""
        return ["deepfilternet"]

    def upload(self) -> bool:
        """Upload audio file to remote."""
        remote_path = f"{self.remote_work_dir}/{self.audio_path.name}"

        print(f"   ⬆️ Uploading {self.audio_path.name}...")
        if not copy_to_remote(str(self.audio_path), remote_path):
            self._error = "Failed to upload audio file"
            return False

        return True

    def run_remote(self) -> bool:
        """Run DeepFilterNet on cgpu."""
        audio_name = self.audio_path.name
        output_name = f"{self.audio_path.stem}_clean.wav"

        cmd = (
            f"cd {self.remote_work_dir} && "
            f"deepFilter '{audio_name}' "
            f"--output-dir . "
            f"--atten-lim {self.attenuation_limit}"
        )

        print(f"   🔇 Running DeepFilterNet...")
        success, stdout, stderr = run_cgpu_command(cmd, timeout=self.timeout)

        if not success:
            self._error = f"DeepFilterNet failed: {stderr or stdout}"
            return False

        return True

    def download(self) -> JobResult:
        """Download cleaned audio."""
        # DeepFilterNet outputs: <stem>_DeepFilterNet3.wav
        output_name = f"{self.audio_path.stem}_DeepFilterNet3.wav"
        remote_output = f"{self.remote_work_dir}/{output_name}"
        local_output = self.output_dir / f"{self.audio_path.stem}_clean.wav"

        print(f"   ⬇️ Downloading cleaned audio...")

        if download_via_base64(remote_output, str(local_output)):
            self._output_path = local_output
            return JobResult(
                success=True,
                output_path=str(local_output),
                metadata={
                    "method": "deepfilternet",
                    "attenuation_limit": self.attenuation_limit,
                }
            )
        else:
            return JobResult(
                success=False,
                error="Failed to download cleaned audio"
            )

    @property
    def output_path(self) -> Optional[Path]:
        """Path to output file (available after successful execution)."""
        return self._output_path


# =============================================================================
# High-Level API
# =============================================================================

def isolate_voice(
    audio_path: str,
    model: str = "htdemucs",
    fast: bool = True,
) -> Optional[str]:
    """
    Quick voice isolation helper.

    Args:
        audio_path: Path to audio/video file
        model: Demucs model to use
        fast: Use two-stem mode for faster processing

    Returns:
        Path to isolated vocals, or None if failed
    """
    job = VoiceIsolationJob(
        audio_path=audio_path,
        model=model,
        two_stems=fast,
    )
    result = job.execute()
    return result.output_path if result.success else None


def reduce_noise(
    audio_path: str,
    strength: int = 100,
) -> Optional[str]:
    """
    Quick noise reduction helper.

    Args:
        audio_path: Path to audio/video file
        strength: Noise reduction strength (0-100)

    Returns:
        Path to cleaned audio, or None if failed
    """
    job = NoiseReductionJob(
        audio_path=audio_path,
        attenuation_limit=strength,
    )
    result = job.execute()
    return result.output_path if result.success else None


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys

    def print_usage():
        print("Voice Isolation - Extract vocals from audio via cgpu")
        print()
        print("Usage:")
        print("  python -m montage_ai.cgpu_jobs.voice_isolation <audio_file> [options]")
        print()
        print("Options:")
        print("  --model MODEL    Demucs model (htdemucs, htdemucs_ft, mdx_extra, mdx)")
        print("  --all-stems      Keep all stems, not just vocals")
        print("  --denoise        Use DeepFilterNet instead (faster, noise reduction only)")
        print()
        print("Examples:")
        print("  python -m montage_ai.cgpu_jobs.voice_isolation interview.wav")
        print("  python -m montage_ai.cgpu_jobs.voice_isolation podcast.mp3 --model mdx_extra")
        print("  python -m montage_ai.cgpu_jobs.voice_isolation noisy.wav --denoise")

    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print_usage()
        sys.exit(0)

    from ..cgpu_utils import is_cgpu_available

    if not is_cgpu_available():
        print("❌ cgpu not available")
        print("   Set CGPU_ENABLED=true and ensure cgpu is installed")
        sys.exit(1)

    audio_file = sys.argv[1]
    args = sys.argv[2:]

    # Parse options
    model = "htdemucs"
    all_stems = "--all-stems" in args
    use_denoise = "--denoise" in args

    for i, arg in enumerate(args):
        if arg == "--model" and i + 1 < len(args):
            model = args[i + 1]

    print(f"Input:  {audio_file}")

    if use_denoise:
        print("Method: DeepFilterNet (noise reduction)")
        result_path = reduce_noise(audio_file)
    else:
        print(f"Model:  {model}")
        print(f"Stems:  {'all' if all_stems else 'vocals only'}")

        job = VoiceIsolationJob(
            audio_path=audio_file,
            model=model,
            keep_all_stems=all_stems,
        )
        result = job.execute()
        result_path = result.output_path if result.success else None

        if result.success and all_stems:
            print("\nAll stems:")
            for stem, path in result.metadata.get("stems", {}).items():
                print(f"  {stem}: {path}")

    if result_path:
        print(f"\n✅ Success: {result_path}")
        sys.exit(0)
    else:
        print("\n❌ Voice isolation failed")
        sys.exit(1)
