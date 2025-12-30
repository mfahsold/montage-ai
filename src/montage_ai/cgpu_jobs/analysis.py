"""
Analysis Jobs - Offload CPU-intensive analysis to Cloud GPU/CPU.

Includes:
- SceneDetectionJob: Runs PySceneDetect remotely.
- BeatAnalysisJob: Runs Librosa beat tracking remotely.
"""

import json
import tempfile
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import CGPUJob, JobResult
from ..cgpu_utils import run_cgpu_command, copy_to_remote, download_via_base64


def _pick_local_output_path(input_path: Path, suffix: str) -> Path:
    """Choose a writable local output path for cgpu job results."""
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    candidate = input_path.with_suffix(safe_suffix)
    if os.access(candidate.parent, os.W_OK):
        return candidate

    cache_root = os.environ.get("CGPU_OUTPUT_DIR")
    for root in [cache_root, "/data/output", "/tmp"]:
        if root and os.path.isdir(root) and os.access(root, os.W_OK):
            path_hash = hashlib.sha1(str(input_path).encode("utf-8")).hexdigest()[:8]
            filename = f"{input_path.stem}.{path_hash}{safe_suffix}"
            return Path(root) / "cgpu" / filename

    return candidate


class SceneDetectionJob(CGPUJob):
    """
    Offload scene detection to cgpu.
    """
    job_type: str = "scene_detection"
    timeout: int = 600

    def __init__(self, input_path: str, threshold: float = 30.0):
        super().__init__()
        self.input_path = Path(input_path).resolve()
        self.threshold = threshold
        self.output_filename = "scenes.json"

    def prepare_local(self) -> bool:
        if not self.input_path.exists():
            print(f"Error: Input file not found: {self.input_path}")
            return False
        return True

    def get_requirements(self) -> List[str]:
        return ["scenedetect[opencv]", "opencv-python-headless", "numpy"]

    def upload(self) -> bool:
        # 1. Upload video
        print(f"Uploading {self.input_path.name}...")
        remote_video = f"{self.remote_work_dir}/{self.input_path.name}"
        if not copy_to_remote(str(self.input_path), remote_video):
            return False

        # 2. Create and upload python script
        script_content = f"""
import sys
import json
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

def main():
    video_path = "{self.input_path.name}"
    output_json = "{self.output_filename}"
    threshold = {self.threshold}

    print(f"Processing {{video_path}} with threshold {{threshold}}...")
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    result = []
    for scene in scenes:
        result.append({{
            "start": scene[0].get_seconds(),
            "end": scene[1].get_seconds(),
            "frames": (scene[0].get_frames(), scene[1].get_frames())
        }})

    with open(output_json, 'w') as f:
        json.dump(result, f)
    print(f"Saved {{len(result)}} scenes to {{output_json}}")

if __name__ == "__main__":
    main()
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(script_content)
            tmp_path = tmp.name
        
        print("Uploading analysis script...")
        remote_script = f"{self.remote_work_dir}/detect_scenes.py"
        success = copy_to_remote(tmp_path, remote_script)
        os.unlink(tmp_path)
        return success

    def run_remote(self) -> bool:
        print("Running scene detection remotely...")
        cmd = f"cd {self.remote_work_dir} && python detect_scenes.py"
        success, stdout, stderr = run_cgpu_command(cmd)
        if not success:
            print(f"   ❌ Scene detection failed: {stderr or stdout}")
        return success

    def download(self) -> JobResult:
        print("Downloading results...")
        local_output = _pick_local_output_path(self.input_path, ".scenes.json")
        remote_path = f"{self.remote_work_dir}/{self.output_filename}"
        
        local_output.parent.mkdir(parents=True, exist_ok=True)
        if download_via_base64(remote_path, str(local_output)):
            with open(local_output, 'r') as f:
                data = json.load(f)
            return JobResult(success=True, output_path=str(local_output), metadata={"scene_count": len(data)})
        
        return JobResult(success=False, error="Failed to download results")


class BeatAnalysisJob(CGPUJob):
    """
    Offload beat detection and energy analysis to cgpu.
    """
    job_type: str = "beat_analysis"
    timeout: int = 300

    def __init__(self, input_path: str):
        super().__init__()
        self.input_path = Path(input_path).resolve()
        self.output_filename = "analysis.json"

    def prepare_local(self) -> bool:
        if not self.input_path.exists():
            print(f"Error: Input file not found: {self.input_path}")
            return False
        return True

    def get_requirements(self) -> List[str]:
        return ["librosa", "numpy", "soundfile"]

    def upload(self) -> bool:
        # 1. Upload audio
        print(f"Uploading {self.input_path.name}...")
        remote_audio = f"{self.remote_work_dir}/{self.input_path.name}"
        if not copy_to_remote(str(self.input_path), remote_audio):
            return False

        # 2. Create and upload python script
        script_content = f"""
import sys
import json
import numpy as np
import librosa

def main():
    audio_path = "{self.input_path.name}"
    output_json = "{self.output_filename}"

    print(f"Analyzing {{audio_path}}...")
    y, sr = librosa.load(audio_path)
    
    print("Detecting beats...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    print("Analyzing energy...")
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    result = {{
        "tempo": float(tempo),
        "beat_times": beat_times.tolist(),
        "duration": librosa.get_duration(y=y, sr=sr),
        "sample_rate": sr,
        "energy": {{
            "times": times.tolist(),
            "rms": rms.tolist()
        }}
    }}

    with open(output_json, 'w') as f:
        json.dump(result, f)
    print("Analysis complete.")

if __name__ == "__main__":
    main()
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(script_content)
            tmp_path = tmp.name
        
        print("Uploading analysis script...")
        remote_script = f"{self.remote_work_dir}/analyze_audio.py"
        success = copy_to_remote(tmp_path, remote_script)
        os.unlink(tmp_path)
        return success

    def run_remote(self) -> bool:
        print("Running audio analysis remotely...")
        cmd = f"cd {self.remote_work_dir} && python analyze_audio.py"
        success, stdout, stderr = run_cgpu_command(cmd)
        if not success:
            print(f"   ❌ Audio analysis failed: {stderr or stdout}")
        return success

    def download(self) -> JobResult:
        print("Downloading results...")
        local_output = _pick_local_output_path(self.input_path, ".analysis.json")
        remote_path = f"{self.remote_work_dir}/{self.output_filename}"
        
        local_output.parent.mkdir(parents=True, exist_ok=True)
        if download_via_base64(remote_path, str(local_output)):
            return JobResult(success=True, output_path=str(local_output))
        
        return JobResult(success=False, error="Failed to download results")
