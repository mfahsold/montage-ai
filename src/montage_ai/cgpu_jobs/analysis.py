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
import tarfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import CGPUJob, JobResult
from ..cgpu_utils import run_cgpu_command, copy_to_remote, download_via_base64
from ..logger import logger
from ..config import get_settings
from ..storytelling.tension_provider import TensionProvider


def _pick_local_output_path(input_path: Path, suffix: str) -> Path:
    """Choose a writable local output path for cgpu job results."""
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    candidate = input_path.with_suffix(safe_suffix)
    if os.access(candidate.parent, os.W_OK):
        return candidate

    settings = get_settings()
    cache_root = settings.llm.cgpu_output_dir
    
    # Use configured paths instead of hardcoded ones
    search_paths = [cache_root, str(settings.paths.output_dir), str(settings.paths.temp_dir)]
    
    for root in search_paths:
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
            logger.error(f"Input file not found: {self.input_path}")
            return False
        return True

    def get_requirements(self) -> List[str]:
        return ["scenedetect[opencv]", "opencv-python-headless", "numpy"]

    def upload(self) -> bool:
        # 1. Upload video
        logger.info(f"Uploading {self.input_path.name}...")
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
        
        logger.info("Uploading analysis script...")
        remote_script = f"{self.remote_work_dir}/detect_scenes.py"
        success = copy_to_remote(tmp_path, remote_script)
        os.unlink(tmp_path)
        return success

    def run_remote(self) -> bool:
        logger.info("Running scene detection remotely...")
        cmd = f"cd {self.remote_work_dir} && python detect_scenes.py"
        success, stdout, stderr = run_cgpu_command(cmd)
        if not success:
            logger.error(f"Scene detection failed: {stderr or stdout}")
        return success

    def download(self) -> JobResult:
        logger.info("Downloading results...")
        local_output = _pick_local_output_path(self.input_path, ".scenes.json")
        remote_path = f"{self.remote_work_dir}/{self.output_filename}"
        
        local_output.parent.mkdir(parents=True, exist_ok=True)
        if download_via_base64(remote_path, str(local_output)):
            with open(local_output, 'r') as f:
                data = json.load(f)
            return JobResult(success=True, output_path=str(local_output), metadata={"scene_count": len(data)})
        
        return JobResult(success=False, error="Failed to download results")

    def expected_output_path(self) -> Optional[Path]:
        """Expected output path for idempotent reuse."""
        return _pick_local_output_path(self.input_path, ".scenes.json")


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
            logger.error(f"Input file not found: {self.input_path}")
            return False
        return True

    def get_requirements(self) -> List[str]:
        return ["librosa", "numpy", "soundfile"]

    def upload(self) -> bool:
        # 1. Upload audio
        logger.info(f"Uploading {self.input_path.name}...")
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
        
        logger.info("Uploading analysis script...")
        remote_script = f"{self.remote_work_dir}/analyze_audio.py"
        success = copy_to_remote(tmp_path, remote_script)
        os.unlink(tmp_path)
        return success

    def run_remote(self) -> bool:
        logger.info("Running audio analysis remotely...")
        cmd = f"cd {self.remote_work_dir} && python analyze_audio.py"
        success, stdout, stderr = run_cgpu_command(cmd)
        if not success:
            logger.error(f"Audio analysis failed: {stderr or stdout}")
        return success

    def download(self) -> JobResult:
        logger.info("Downloading results...")
        local_output = _pick_local_output_path(self.input_path, ".analysis.json")
        remote_path = f"{self.remote_work_dir}/{self.output_filename}"
        
        local_output.parent.mkdir(parents=True, exist_ok=True)
        if download_via_base64(remote_path, str(local_output)):
            return JobResult(success=True, output_path=str(local_output))
        
        return JobResult(success=False, error="Failed to download results")

    def expected_output_path(self) -> Optional[Path]:
        """Expected output path for idempotent reuse."""
        return _pick_local_output_path(self.input_path, ".analysis.json")


class TensionAnalysisBatchJob(CGPUJob):
    """
    Offload tension analysis (optical flow + edge density + audio energy) to cgpu.
    """
    job_type: str = "tension_analysis_batch"
    timeout: int = 1200
    requires_gpu: bool = False

    def __init__(self, input_paths: List[str], output_dir: str, sample_fps: float = 2.0):
        super().__init__()
        self.input_paths = [Path(p).resolve() for p in input_paths]
        self.output_dir = Path(output_dir)
        self.sample_fps = max(0.5, float(sample_fps))
        self.output_archive = "tension_results.tar.gz"
        self.manifest_name = "manifest.json"

    def prepare_local(self) -> bool:
        if not self.input_paths:
            logger.error("No input clips provided for tension analysis")
            return False
        missing = [p for p in self.input_paths if not p.exists()]
        if missing:
            logger.error(f"Missing input clips: {missing}")
            return False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def get_requirements(self) -> List[str]:
        return ["opencv-python-headless", "numpy", "librosa", "soundfile", "imageio-ffmpeg"]

    def upload(self) -> bool:
        # Upload video clips + manifest
        manifest = {"clips": []}

        for idx, clip in enumerate(self.input_paths):
            remote_name = f"clip_{idx}{clip.suffix.lower()}"
            remote_path = f"{self.remote_work_dir}/{remote_name}"
            logger.info(f"Uploading {clip.name} -> {remote_name}...")
            if not copy_to_remote(str(clip), remote_path):
                return False

            clip_id = TensionProvider._get_clip_id(str(clip))
            manifest["clips"].append(
                {
                    "clip_id": clip_id,
                    "remote_name": remote_name,
                    "source_path": str(clip),
                }
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(manifest, tmp)
            manifest_path = tmp.name

        remote_manifest = f"{self.remote_work_dir}/{self.manifest_name}"
        success = copy_to_remote(manifest_path, remote_manifest)
        os.unlink(manifest_path)
        return success

    def run_remote(self) -> bool:
        logger.info("Running tension analysis remotely...")
        script_content = f"""
import json
import math
import os
import tarfile
import subprocess
import cv2
import numpy as np
import librosa
import imageio_ffmpeg

MANIFEST = "{self.manifest_name}"
OUTPUT_DIR = "outputs"
SAMPLE_FPS = {self.sample_fps}
ARCHIVE = "{self.output_archive}"

def extract_audio(video_path, wav_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "22050",
        wav_path
    ]
    # Use subprocess.run directly in remote script (no cmd_runner available there)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def analyze_visual(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps / SAMPLE_FPS))
    prev_gray = None
    flow_vals = []
    edge_vals = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_vals.append(float(edges.mean() / 255.0))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_vals.append(float(np.mean(mag)))
        prev_gray = gray
        frame_idx += 1
    cap.release()
    motion = float(np.mean(flow_vals)) if flow_vals else 0.0
    edge_density = float(np.mean(edge_vals)) if edge_vals else 0.0
    motion_norm = math.tanh(motion)
    tension = max(0.0, min(1.0, 0.6 * motion_norm + 0.4 * edge_density))
    return motion, edge_density, tension

def analyze_audio(video_path):
    audio_rms = 0.0
    spectral_flux = 0.0
    wav_path = video_path + ".wav"
    try:
        extract_audio(video_path, wav_path)
        y, sr = librosa.load(wav_path, sr=22050)
        rms = librosa.feature.rms(y=y)[0]
        audio_rms = float(np.mean(rms))
        spec = np.abs(librosa.stft(y))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
        spectral_flux = float(np.mean(flux)) if flux.size else 0.0
    except Exception:
        pass
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    return audio_rms, spectral_flux

def main():
    with open(MANIFEST, "r") as f:
        manifest = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for entry in manifest.get("clips", []):
        clip_id = entry["clip_id"]
        remote_name = entry["remote_name"]
        motion, edge_density, tension = analyze_visual(remote_name)
        audio_rms, audio_flux = analyze_audio(remote_name)
        output = {{
            "clip_id": clip_id,
            "visual": {{
                "motion_score": motion,
                "edge_density": edge_density
            }},
            "audio": {{
                "rms": audio_rms,
                "spectral_flux": audio_flux
            }},
            "tension": tension
        }}
        out_path = os.path.join(OUTPUT_DIR, f"{{clip_id}}_analysis.json")
        with open(out_path, "w") as out_file:
            json.dump(output, out_file)

    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(OUTPUT_DIR, arcname=OUTPUT_DIR)

if __name__ == "__main__":
    main()
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(script_content)
            local_script = tmp.name

        remote_script = f"{self.remote_work_dir}/tension_batch.py"
        if not copy_to_remote(local_script, remote_script):
            os.unlink(local_script)
            return False
        os.unlink(local_script)

        cmd = f"cd {self.remote_work_dir} && python tension_batch.py"
        success, stdout, stderr = run_cgpu_command(cmd, timeout=self.timeout)
        if not success:
            logger.error(f"Tension analysis failed: {stderr or stdout}")
        return success

    def download(self) -> JobResult:
        logger.info("Downloading tension analysis results...")
        remote_archive = f"{self.remote_work_dir}/{self.output_archive}"
        local_archive = self.output_dir / self.output_archive
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not download_via_base64(remote_archive, str(local_archive)):
            return JobResult(success=False, error="Failed to download tension archive")

        try:
            with tarfile.open(local_archive, "r:gz") as tar:
                tar.extractall(path=self.output_dir)
        except Exception as exc:
            return JobResult(success=False, error=f"Failed to extract archive: {exc}")
        finally:
            try:
                local_archive.unlink()
            except OSError:
                pass

        outputs = list((self.output_dir / "outputs").glob("*_analysis.json"))
        for item in outputs:
            item.replace(self.output_dir / item.name)
        try:
            (self.output_dir / "outputs").rmdir()
        except OSError:
            pass

        return JobResult(
            success=True,
            output_path=str(self.output_dir),
            metadata={"clip_count": len(outputs)},
        )
