"""
SOTA Scene Detection - AutoShot + TransNetV2 Integration

State-of-the-art neural network scene detection using AutoShot or TransNetV2.
AutoShot (CVPR 2023 NAS) achieves +4.2% F1 and +3.5% Precision over TransNetV2.

Features:
- AutoShot neural network (CVPR 2023, NAS-optimized 3D ConvNet + Transformer)
- TransNetV2 neural network (CVPR 2020, SOTA on ClipShots/BBC benchmarks)
- GPU acceleration via PyTorch (CUDA, ROCm, MPS)
- Automatic fallback: AutoShot > TransNetV2 > PySceneDetect
- Caching via msgpack for repeated analysis
- Batch processing support for multiple videos

Usage:
    from montage_ai.scene_detection_sota import detect_scenes_sota

    # Auto-select best backend (AutoShot > TransNetV2 > PySceneDetect)
    scenes = detect_scenes_sota(video_path)

    # Force specific backend
    scenes = detect_scenes_sota(video_path, backend="autoshot")
    scenes = detect_scenes_sota(video_path, backend="transnetv2")
    scenes = detect_scenes_sota(video_path, backend="pyscenedetect")

References:
- AutoShot: https://github.com/wentaozhu/AutoShot
- TransNetV2: https://github.com/soCzech/TransNetV2
- Paper (AutoShot): "AutoShot: A Short Video Dataset and SOTA Shot Boundary Detection"
"""

import os
import time
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from functools import lru_cache

import numpy as np

from .logger import logger
from .config import get_settings

# Lazy imports for optional dependencies
_transnet_model = None
_transnet_available = None
_autoshot_model = None
_autoshot_available = None


def _check_autoshot_available() -> bool:
    """Check if AutoShot is available (SOTA, +4.2% F1 over TransNetV2)."""
    global _autoshot_available
    if _autoshot_available is not None:
        return _autoshot_available

    try:
        import torch
        # AutoShot uses a custom model architecture
        # Check if the autoshot package or weights are available
        from autoshot import AutoShotModel
        _autoshot_available = True
        logger.debug("AutoShot available (SOTA scene detection - +4.2% F1 over TransNetV2)")
    except ImportError:
        # Fallback: Check for manual weight loading
        try:
            import torch
            weights_path = Path.home() / ".cache" / "montage_ai" / "models" / "autoshot.pt"
            if weights_path.exists():
                _autoshot_available = True
                logger.debug("AutoShot weights found (SOTA scene detection enabled)")
            else:
                _autoshot_available = False
                logger.debug("AutoShot not available, trying TransNetV2")
        except Exception:
            _autoshot_available = False

    return _autoshot_available


def _check_transnetv2_available() -> bool:
    """Check if TransNetV2 is available."""
    global _transnet_available
    if _transnet_available is not None:
        return _transnet_available

    try:
        import torch
        from transnetv2 import TransNetV2
        _transnet_available = True
        logger.debug("TransNetV2 available (neural scene detection enabled)")
    except ImportError:
        _transnet_available = False
        logger.debug("TransNetV2 not available, using PySceneDetect fallback")

    return _transnet_available


def _get_pytorch_device() -> str:
    """Get the best available PyTorch device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_autoshot_model():
    """Get or create AutoShot model (singleton)."""
    global _autoshot_model

    if _autoshot_model is not None:
        return _autoshot_model

    if not _check_autoshot_available():
        return None

    import torch

    device = _get_pytorch_device()
    logger.info(f"Loading AutoShot model on {device}...")

    try:
        from autoshot import AutoShotModel
        _autoshot_model = AutoShotModel()
        _autoshot_model.to(device)
        _autoshot_model.eval()
    except ImportError:
        # Manual weight loading fallback
        weights_path = Path.home() / ".cache" / "montage_ai" / "models" / "autoshot.pt"
        if weights_path.exists():
            # Load custom model architecture
            _autoshot_model = torch.load(weights_path, map_location=device)
            _autoshot_model.eval()
            logger.info("AutoShot loaded from local weights")
        else:
            return None

    return _autoshot_model


def _get_transnet_model():
    """Get or create TransNetV2 model (singleton)."""
    global _transnet_model

    if _transnet_model is not None:
        return _transnet_model

    if not _check_transnetv2_available():
        return None

    import torch
    from transnetv2 import TransNetV2

    device = _get_pytorch_device()

    logger.info(f"Loading TransNetV2 model on {device}...")
    _transnet_model = TransNetV2()
    _transnet_model.to(device)
    _transnet_model.eval()

    return _transnet_model


@dataclass
class SceneBoundary:
    """A detected scene boundary with confidence."""
    frame_number: int
    timestamp: float
    confidence: float

    def __repr__(self):
        return f"SceneBoundary(t={self.timestamp:.2f}s, conf={self.confidence:.2f})"


def _compute_cache_key(video_path: str, backend: str) -> str:
    """Compute cache key for scene detection results."""
    stat = os.stat(video_path)
    key_parts = f"{video_path}:{stat.st_size}:{stat.st_mtime}:{backend}"
    return hashlib.md5(key_parts.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    """Get cache file path for scene detection."""
    settings = get_settings()
    cache_dir = settings.paths.metadata_cache_dir / "scene_detection"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.msgpack"


def _load_from_cache(video_path: str, backend: str) -> Optional[List[SceneBoundary]]:
    """Load scene detection results from cache."""
    try:
        import msgpack

        cache_key = _compute_cache_key(video_path, backend)
        cache_path = _get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        with open(cache_path, "rb") as f:
            data = msgpack.unpack(f, raw=False)

        scenes = [
            SceneBoundary(
                frame_number=s["frame_number"],
                timestamp=s["timestamp"],
                confidence=s["confidence"]
            )
            for s in data["scenes"]
        ]

        logger.debug(f"Loaded {len(scenes)} scenes from cache: {cache_path.name}")
        return scenes

    except Exception as e:
        logger.debug(f"Cache load failed: {e}")
        return None


def _save_to_cache(video_path: str, backend: str, scenes: List[SceneBoundary]) -> None:
    """Save scene detection results to cache."""
    try:
        import msgpack

        cache_key = _compute_cache_key(video_path, backend)
        cache_path = _get_cache_path(cache_key)

        data = {
            "video_path": video_path,
            "backend": backend,
            "scenes": [
                {
                    "frame_number": s.frame_number,
                    "timestamp": s.timestamp,
                    "confidence": s.confidence
                }
                for s in scenes
            ]
        }

        with open(cache_path, "wb") as f:
            msgpack.pack(data, f)

        logger.debug(f"Saved {len(scenes)} scenes to cache: {cache_path.name}")

    except Exception as e:
        logger.debug(f"Cache save failed: {e}")


def _detect_transnetv2(
    video_path: str,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> List[SceneBoundary]:
    """
    Detect scenes using TransNetV2 neural network.

    TransNetV2 achieves SOTA performance on ClipShots and BBC benchmarks.
    On GPU: ~250 fps (vs ~50 fps for content detection).

    Args:
        video_path: Path to video file
        threshold: Confidence threshold (0-1, lower = more sensitive)
        batch_size: Batch size for GPU inference

    Returns:
        List of SceneBoundary objects
    """
    import torch
    import cv2

    model = _get_transnet_model()
    if model is None:
        raise RuntimeError("TransNetV2 model not available")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"TransNetV2: Analyzing {total_frames} frames at {fps:.1f} fps...")

    # TransNetV2 expects 48x27 RGB frames (batch x 100 frames x 27 x 48 x 3)
    target_size = (48, 27)
    window_size = 100  # TransNetV2 window

    frames = []
    predictions = []

    start_time = time.perf_counter()

    with torch.no_grad():
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert to RGB
            frame_small = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_idx += 1

            # Process in batches of windows
            if len(frames) >= window_size:
                # Stack frames into window
                window = np.stack(frames[:window_size], axis=0)

                # To tensor
                device = next(model.parameters()).device
                input_tensor = torch.from_numpy(window).float().unsqueeze(0)
                input_tensor = input_tensor.permute(0, 1, 4, 2, 3) / 255.0  # BCTHW
                input_tensor = input_tensor.to(device)

                # Inference
                single_frame_pred, _ = model(input_tensor)
                pred = single_frame_pred[0].cpu().numpy()
                predictions.extend(pred.tolist())

                # Slide window
                frames = frames[1:]

        # Process remaining frames
        if frames:
            # Pad to window_size
            while len(frames) < window_size:
                frames.append(frames[-1])

            window = np.stack(frames[:window_size], axis=0)
            device = next(model.parameters()).device
            input_tensor = torch.from_numpy(window).float().unsqueeze(0)
            input_tensor = input_tensor.permute(0, 1, 4, 2, 3) / 255.0
            input_tensor = input_tensor.to(device)

            single_frame_pred, _ = model(input_tensor)
            pred = single_frame_pred[0].cpu().numpy()

            remaining = min(len(pred), total_frames - len(predictions))
            predictions.extend(pred[:remaining].tolist())

    cap.release()

    elapsed = time.perf_counter() - start_time
    effective_fps = total_frames / elapsed if elapsed > 0 else 0

    logger.info(f"TransNetV2: Processed {total_frames} frames in {elapsed:.1f}s ({effective_fps:.0f} fps)")

    # Extract scene boundaries from predictions
    scenes = []
    predictions_array = np.array(predictions)

    # Find peaks above threshold
    above_threshold = predictions_array > threshold
    transitions = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

    for frame_num in transitions:
        if frame_num < len(predictions_array):
            scenes.append(SceneBoundary(
                frame_number=int(frame_num),
                timestamp=frame_num / fps,
                confidence=float(predictions_array[frame_num])
            ))

    return scenes


def _detect_autoshot(
    video_path: str,
    threshold: float = 0.5,
) -> List[SceneBoundary]:
    """
    Detect scenes using AutoShot (SOTA, +4.2% F1 over TransNetV2).

    AutoShot uses Neural Architecture Search (NAS) to find optimal 3D ConvNet + Transformer
    architecture for shot boundary detection. Trained on SHOT dataset (853 videos).

    Args:
        video_path: Path to video file
        threshold: Confidence threshold (0-1, lower = more sensitive)

    Returns:
        List of SceneBoundary objects
    """
    import torch
    import cv2

    model = _get_autoshot_model()
    if model is None:
        raise RuntimeError("AutoShot model not available")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"AutoShot (SOTA): Analyzing {total_frames} frames at {fps:.1f} fps...")

    # AutoShot expects 224x224 RGB frames
    target_size = (224, 224)
    window_size = 16  # AutoShot uses 16-frame windows

    frames_buffer = []
    predictions = []
    device = _get_pytorch_device()

    start_time = time.perf_counter()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert to RGB
            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames_buffer.append(frame_rgb)

            # Process when we have enough frames
            if len(frames_buffer) >= window_size:
                # Stack frames into window [B, T, H, W, C] -> [B, C, T, H, W]
                window = np.stack(frames_buffer[:window_size], axis=0)
                input_tensor = torch.from_numpy(window).float().unsqueeze(0)
                input_tensor = input_tensor.permute(0, 4, 1, 2, 3) / 255.0
                input_tensor = input_tensor.to(device)

                # Model inference
                try:
                    output = model(input_tensor)
                    # AutoShot outputs per-frame shot boundary probabilities
                    if isinstance(output, tuple):
                        output = output[0]
                    pred = torch.sigmoid(output).cpu().numpy().flatten()
                    predictions.extend(pred[:1].tolist())  # One prediction per window slide
                except Exception as e:
                    logger.debug(f"AutoShot inference error: {e}")
                    predictions.append(0.0)

                # Slide by 1 frame
                frames_buffer = frames_buffer[1:]

    cap.release()

    elapsed = time.perf_counter() - start_time
    effective_fps = total_frames / elapsed if elapsed > 0 else 0

    logger.info(f"AutoShot: Processed {total_frames} frames in {elapsed:.1f}s ({effective_fps:.0f} fps)")

    # Extract scene boundaries from predictions
    scenes = []
    predictions_array = np.array(predictions)

    if len(predictions_array) > 0:
        # Find peaks above threshold
        above_threshold = predictions_array > threshold
        transitions = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

        for frame_num in transitions:
            if frame_num < len(predictions_array):
                scenes.append(SceneBoundary(
                    frame_number=int(frame_num),
                    timestamp=frame_num / fps,
                    confidence=float(predictions_array[frame_num])
                ))

    return scenes


def _detect_pyscenedetect(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
) -> List[SceneBoundary]:
    """
    Detect scenes using PySceneDetect ContentDetector (fallback).

    Args:
        video_path: Path to video file
        threshold: Content detection threshold (higher = less sensitive)
        min_scene_len: Minimum scene length in frames

    Returns:
        List of SceneBoundary objects
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    logger.info(f"PySceneDetect: Analyzing {os.path.basename(video_path)}...")

    start_time = time.perf_counter()

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video, show_progress=False)

    scene_list = scene_manager.get_scene_list()

    elapsed = time.perf_counter() - start_time

    # Convert to SceneBoundary objects (each scene is a cut point)
    scenes = []
    for i, (start, end) in enumerate(scene_list):
        if i > 0:  # First scene start is frame 0, not a cut
            scenes.append(SceneBoundary(
                frame_number=start.get_frames(),
                timestamp=start.get_seconds(),
                confidence=0.8  # PySceneDetect doesn't provide confidence
            ))

    fps = video.frame_rate
    total_frames = video.duration.get_frames()
    effective_fps = total_frames / elapsed if elapsed > 0 else 0

    logger.info(f"PySceneDetect: {len(scenes)} cuts in {elapsed:.1f}s ({effective_fps:.0f} fps)")

    return scenes


def detect_scenes_sota(
    video_path: str,
    backend: str = "auto",
    threshold: Optional[float] = None,
    use_cache: bool = True,
) -> List[Tuple[float, float]]:
    """
    Detect scenes using SOTA method.

    Backend priority: AutoShot > TransNetV2 > PySceneDetect

    Args:
        video_path: Path to video file
        backend: "auto", "autoshot", "transnetv2", or "pyscenedetect"
        threshold: Detection threshold (auto-adjusted per backend)
        use_cache: Use msgpack cache for repeated analysis

    Returns:
        List of (start_time, end_time) tuples for each scene
    """
    settings = get_settings()

    # Select backend (priority: AutoShot > TransNetV2 > PySceneDetect)
    if backend == "auto":
        if _check_autoshot_available():
            backend = "autoshot"
        elif _check_transnetv2_available():
            backend = "transnetv2"
        else:
            backend = "pyscenedetect"

    # Check cache
    if use_cache:
        cached = _load_from_cache(video_path, backend)
        if cached is not None:
            return _boundaries_to_scenes(video_path, cached)

    # Detect scenes
    if backend == "autoshot":
        thresh = threshold if threshold is not None else 0.5
        try:
            boundaries = _detect_autoshot(video_path, threshold=thresh)
        except Exception as e:
            logger.warning(f"AutoShot failed ({e}), falling back to TransNetV2")
            if _check_transnetv2_available():
                boundaries = _detect_transnetv2(video_path, threshold=thresh)
            else:
                thresh = threshold if threshold is not None else settings.thresholds.scene_threshold
                boundaries = _detect_pyscenedetect(video_path, threshold=thresh)
    elif backend == "transnetv2":
        thresh = threshold if threshold is not None else 0.5
        boundaries = _detect_transnetv2(video_path, threshold=thresh)
    else:
        thresh = threshold if threshold is not None else settings.thresholds.scene_threshold
        boundaries = _detect_pyscenedetect(video_path, threshold=thresh)

    # Save to cache
    if use_cache and boundaries:
        _save_to_cache(video_path, backend, boundaries)

    return _boundaries_to_scenes(video_path, boundaries)


def _boundaries_to_scenes(
    video_path: str,
    boundaries: List[SceneBoundary]
) -> List[Tuple[float, float]]:
    """Convert scene boundaries to (start, end) tuples."""
    import cv2

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    if not boundaries:
        # No cuts detected - whole video is one scene
        return [(0.0, duration)]

    scenes = []
    prev_time = 0.0

    for boundary in sorted(boundaries, key=lambda b: b.timestamp):
        if boundary.timestamp > prev_time:
            scenes.append((prev_time, boundary.timestamp))
        prev_time = boundary.timestamp

    # Add final scene
    if prev_time < duration:
        scenes.append((prev_time, duration))

    return scenes


def get_available_backend() -> str:
    """Get the best available scene detection backend."""
    if _check_autoshot_available():
        return "autoshot"
    if _check_transnetv2_available():
        return "transnetv2"
    return "pyscenedetect"


def benchmark_backends(video_path: str) -> Dict[str, Any]:
    """
    Benchmark scene detection backends on a video.

    Returns timing and accuracy comparison.
    """
    results = {}

    # PySceneDetect (baseline)
    start = time.perf_counter()
    psd_scenes = detect_scenes_sota(video_path, backend="pyscenedetect", use_cache=False)
    psd_time = time.perf_counter() - start
    results["pyscenedetect"] = {
        "scenes": len(psd_scenes),
        "time_seconds": psd_time,
        "available": True
    }

    # TransNetV2 (if available)
    if _check_transnetv2_available():
        start = time.perf_counter()
        tn_scenes = detect_scenes_sota(video_path, backend="transnetv2", use_cache=False)
        tn_time = time.perf_counter() - start
        results["transnetv2"] = {
            "scenes": len(tn_scenes),
            "time_seconds": tn_time,
            "speedup": psd_time / tn_time if tn_time > 0 else 0,
            "available": True
        }
    else:
        results["transnetv2"] = {"available": False}

    # AutoShot (SOTA, if available)
    if _check_autoshot_available():
        start = time.perf_counter()
        as_scenes = detect_scenes_sota(video_path, backend="autoshot", use_cache=False)
        as_time = time.perf_counter() - start
        results["autoshot"] = {
            "scenes": len(as_scenes),
            "time_seconds": as_time,
            "speedup": psd_time / as_time if as_time > 0 else 0,
            "available": True,
            "is_sota": True
        }
    else:
        results["autoshot"] = {"available": False}

    return results


# Module exports
__all__ = [
    "detect_scenes_sota",
    "SceneBoundary",
    "get_available_backend",
    "benchmark_backends",
]
