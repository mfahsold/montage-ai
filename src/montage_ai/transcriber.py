"""
Transcriber - Audio Transcription via cgpu (OpenAI Whisper)

Offloads heavy audio transcription to Cloud GPU using cgpu.
Uses OpenAI's Whisper model to generate subtitles and transcripts.

Architecture:
    Audio File → cgpu Upload → Whisper (Cloud GPU) → SRT/VTT Download

Usage:
    from montage_ai.transcriber import Transcriber
    
    transcriber = Transcriber()
    if transcriber.is_available():
        srt_path = transcriber.transcribe("audio.wav", output_format="srt")

"""

import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from .cgpu_utils import (
    is_cgpu_available,
    run_cgpu_command,
    copy_to_remote,
    download_via_base64,
)

class Transcriber:
    def __init__(self, model: str = "medium"):
        """
        Initialize the Transcriber.
        
        Args:
            model: Whisper model size (tiny, base, small, medium, large)
        """
        self.model = model
        self.remote_work_dir = f"/content/montage_transcribe_{uuid.uuid4().hex[:8]}"

    def is_available(self) -> bool:
        """Check if cgpu is available for transcription."""
        return is_cgpu_available()

    def transcribe(self, audio_path: str, output_format: str = "srt") -> Optional[str]:
        """
        Transcribe an audio file using Whisper on cgpu.
        
        Args:
            audio_path: Path to local audio file
            output_format: Output format (srt, vtt, txt, json)
            
        Returns:
            Path to the generated subtitle file (locally), or None if failed.
        """
        if not self.is_available():
            print("⚠️ cgpu not available. Skipping cloud transcription.")
            return None

        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            return None

        print(f"🎤 Transcribing {audio_path.name} using Whisper ({self.model}) on cgpu...")

        # 1. Setup remote environment
        setup_cmd = f"mkdir -p {self.remote_work_dir} && pip install -q openai-whisper"
        print("   • Setting up remote environment...")
        if not run_cgpu_command(setup_cmd):
            print("❌ Failed to setup remote environment")
            return None

        # 2. Upload audio
        remote_audio_path = f"{self.remote_work_dir}/{audio_path.name}"
        print(f"   • Uploading audio ({os.path.getsize(audio_path) / 1024 / 1024:.1f} MB)...")
        if not copy_to_remote(str(audio_path), remote_audio_path):
            print("❌ Failed to upload audio file")
            return None

        # 3. Run Whisper
        # whisper input.wav --model medium --output_format srt --output_dir .
        whisper_cmd = (
            f"cd {self.remote_work_dir} && "
            f"whisper '{audio_path.name}' --model {self.model} --output_format {output_format} --output_dir ."
        )
        print("   • Running Whisper (this may take a moment)...")
        if not run_cgpu_command(whisper_cmd, timeout=600): # 10 min timeout
            print("❌ Whisper transcription failed")
            return None

        # 4. Download result
        # Whisper output filename is usually input filename + .srt (e.g. audio.wav.srt or audio.srt? It's usually audio.srt if input is audio.wav)
        # Actually whisper CLI outputs as <filename_without_extension>.<format>
        output_filename = f"{audio_path.stem}.{output_format}"
        remote_output_path = f"{self.remote_work_dir}/{output_filename}"
        
        local_output_path = audio_path.parent / output_filename
        
        print(f"   • Downloading {output_format.upper()}...")
        if download_via_base64(remote_output_path, str(local_output_path)):
            print(f"✅ Transcription saved to: {local_output_path}")
            
            # Cleanup remote
            run_cgpu_command(f"rm -rf {self.remote_work_dir}")
            return str(local_output_path)
        else:
            print("❌ Failed to download transcript")
            return None

