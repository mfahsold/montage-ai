"""
Render Jobs - Offload FFmpeg rendering/transcoding to Cloud GPU.

Includes:
- FFmpegRenderJob: Runs arbitrary FFmpeg commands remotely (supports NVENC).
"""

import os
from pathlib import Path
from typing import List, Optional

from .base import CGPUJob, JobResult
from ..cgpu_utils import run_cgpu_command, copy_to_remote, download_via_base64


class FFmpegRenderJob(CGPUJob):
    """
    Offload FFmpeg processing to cgpu (with NVENC support).
    """
    job_type: str = "ffmpeg_render"
    timeout: int = 1800  # 30 minutes

    def __init__(
        self, 
        input_paths: List[str], 
        command_args: List[str], 
        output_filename: str
    ):
        """
        Initialize FFmpeg job.

        Args:
            input_paths: List of local paths to input files.
            command_args: List of FFmpeg arguments (e.g. ["-i", "input.mp4", ...]).
                          NOTE: Use filenames only, not full paths, for inputs.
            output_filename: Name of the output file to download.
        """
        super().__init__()
        self.input_paths = [Path(p).resolve() for p in input_paths]
        self.command_args = command_args
        self.output_filename = output_filename

    def prepare_local(self) -> bool:
        for p in self.input_paths:
            if not p.exists():
                print(f"Error: Input file not found: {p}")
                return False
        return True

    def get_requirements(self) -> List[str]:
        return []  # ffmpeg is usually installed system-wide

    def upload(self) -> bool:
        print(f"Uploading {len(self.input_paths)} input files...")
        for p in self.input_paths:
            print(f"  - {p.name}")
            if not copy_to_remote(str(p), self.remote_work_dir):
                return False
        return True

    def run_remote(self) -> bool:
        print("Running FFmpeg remotely...")
        # Construct command
        # Ensure we use the remote working directory
        cmd_str = " ".join(self.command_args)
        full_cmd = f"cd {self.remote_work_dir} && ffmpeg -y {cmd_str}"
        
        print(f"Command: {full_cmd}")
        success, stdout, stderr = run_cgpu_command(full_cmd)
        if not success:
            print(f"   ❌ FFmpeg failed: {stderr or stdout}")
        return success

    def download(self) -> JobResult:
        print("Downloading result...")
        # We save to the directory of the first input file, or current dir?
        # Let's save to the same dir as the first input, or a specific output dir if we had one.
        # For now, save to local current directory or alongside input.
        
        # Better: allow user to specify local output path? 
        # But __init__ didn't take it.
        # Let's assume we download to the directory of the first input file.
        if self.input_paths:
            local_dir = self.input_paths[0].parent
        else:
            local_dir = Path.cwd()
            
        local_output = local_dir / self.output_filename
        remote_path = f"{self.remote_work_dir}/{self.output_filename}"
        
        if download_via_base64(remote_path, str(local_output)):
            return JobResult(success=True, output_path=str(local_output))
        
        return JobResult(success=False, error="Failed to download result")

    @staticmethod
    def create_nvenc_command(input_file: str, output_file: str, bitrate: str = "5M") -> List[str]:
        """Helper to create a standard NVENC transcoding command."""
        return [
            "-i", input_file,
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", bitrate,
            "-c:a", "copy",
            output_file
        ]
