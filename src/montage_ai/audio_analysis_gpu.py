"""
GPU-Accelerated Audio Analysis Module

Optional GPU acceleration for audio analysis using CuPy/PyTorch.
Falls back to NumPy when GPU is not available.

Features:
- GPU-accelerated FFT for spectral analysis (~10x faster on large files)
- CUDA/ROCm/MPS support via PyTorch
- Automatic fallback to CPU

Usage:
    from montage_ai.audio_analysis_gpu import gpu_spectral_analysis

    # Uses GPU if available, otherwise CPU
    spectral_data = gpu_spectral_analysis(audio_path)

Requirements:
    pip install torch  # For GPU support
    # OR
    pip install cupy   # For CUDA-only GPU support
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .logger import logger
from .config import get_settings
from .ffmpeg_utils import build_ffmpeg_cmd

# Lazy GPU detection
_gpu_backend: Optional[str] = None


def _detect_gpu_backend() -> Optional[str]:
    """Detect available GPU backend for audio processing."""
    global _gpu_backend

    if _gpu_backend is not None:
        return _gpu_backend

    # Try PyTorch first (most portable)
    try:
        import torch
        if torch.cuda.is_available():
            _gpu_backend = "torch-cuda"
            logger.debug(f"Audio GPU: PyTorch CUDA ({torch.cuda.get_device_name(0)})")
            return _gpu_backend
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _gpu_backend = "torch-mps"
            logger.debug("Audio GPU: PyTorch MPS (Apple Silicon)")
            return _gpu_backend
        elif hasattr(torch.backends, "rocm") and torch.backends.rocm.is_built():
            _gpu_backend = "torch-rocm"
            logger.debug("Audio GPU: PyTorch ROCm (AMD)")
            return _gpu_backend
    except ImportError:
        pass

    # Try CuPy (CUDA only, faster for pure FFT)
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        _gpu_backend = "cupy"
        logger.debug("Audio GPU: CuPy CUDA")
        return _gpu_backend
    except (ImportError, Exception):
        pass

    _gpu_backend = "cpu"
    logger.debug("Audio GPU: CPU (no GPU acceleration)")
    return _gpu_backend


def is_gpu_audio_available() -> bool:
    """Check if GPU audio processing is available."""
    backend = _detect_gpu_backend()
    return backend not in (None, "cpu")


@dataclass
class SpectralFeatures:
    """Spectral analysis results."""
    frequencies: np.ndarray
    magnitudes: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    spectral_flux: np.ndarray
    sample_rate: int
    hop_length: int

    @property
    def avg_centroid(self) -> float:
        """Average spectral centroid (brightness indicator)."""
        return float(np.mean(self.spectral_centroid))

    @property
    def avg_flux(self) -> float:
        """Average spectral flux (change indicator)."""
        return float(np.mean(self.spectral_flux))


def _load_audio_raw(audio_path: str, sr: int = 22050) -> np.ndarray:
    """
    Load audio as raw samples using FFmpeg (faster than librosa).

    Args:
        audio_path: Path to audio file
        sr: Target sample rate

    Returns:
        Mono audio samples as numpy array
    """
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = build_ffmpeg_cmd([
            "-i", audio_path,
            "-ac", "1",  # Mono
            "-ar", str(sr),  # Sample rate
            "-f", "f32le",  # 32-bit float little-endian
            tmp_path
        ], overwrite=True, hide_banner=True, loglevel="error")

        subprocess.run(cmd, check=True, capture_output=True)

        # Read raw audio
        audio = np.fromfile(tmp_path, dtype=np.float32)
        return audio

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _stft_torch(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: str = "cuda"
) -> np.ndarray:
    """Compute STFT using PyTorch on GPU."""
    import torch

    # Move to GPU
    audio_tensor = torch.from_numpy(audio).to(device)

    # Compute STFT
    window = torch.hann_window(n_fft, device=device)
    stft = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True
    )

    # Magnitude spectrum
    magnitude = torch.abs(stft).cpu().numpy()
    return magnitude


def _stft_cupy(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Compute STFT using CuPy on GPU."""
    import cupy as cp
    from cupyx.scipy.signal import stft as cupy_stft

    # Move to GPU
    audio_gpu = cp.asarray(audio)

    # Compute STFT
    _, _, Zxx = cupy_stft(audio_gpu, fs=1.0, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Magnitude and back to CPU
    magnitude = cp.abs(Zxx).get()
    return magnitude


def _stft_numpy(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Compute STFT using NumPy (CPU fallback)."""
    from scipy.signal import stft

    _, _, Zxx = stft(audio, fs=1.0, nperseg=n_fft, noverlap=n_fft - hop_length)
    return np.abs(Zxx)


def gpu_spectral_analysis(
    audio_path: str,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> SpectralFeatures:
    """
    Perform GPU-accelerated spectral analysis.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length between frames

    Returns:
        SpectralFeatures with frequencies, magnitudes, and derived features
    """
    backend = _detect_gpu_backend()

    logger.info(f"Spectral analysis using {backend}...")

    # Load audio
    audio = _load_audio_raw(audio_path, sr)

    # Compute STFT
    if backend == "torch-cuda":
        magnitude = _stft_torch(audio, n_fft, hop_length, "cuda")
    elif backend == "torch-mps":
        magnitude = _stft_torch(audio, n_fft, hop_length, "mps")
    elif backend == "torch-rocm":
        magnitude = _stft_torch(audio, n_fft, hop_length, "cuda")  # ROCm uses cuda interface
    elif backend == "cupy":
        magnitude = _stft_cupy(audio, n_fft, hop_length)
    else:
        magnitude = _stft_numpy(audio, n_fft, hop_length)

    # Compute frequency bins
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Compute spectral features
    # Spectral centroid (center of mass of spectrum)
    spectral_centroid = np.sum(frequencies[:, np.newaxis] * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-6)

    # Spectral rolloff (frequency below which X% of energy is contained)
    cumsum = np.cumsum(magnitude, axis=0)
    rolloff_threshold = 0.85 * cumsum[-1]
    spectral_rolloff = np.argmax(cumsum >= rolloff_threshold, axis=0).astype(float)
    spectral_rolloff = spectral_rolloff * sr / n_fft

    # Spectral flux (rate of change)
    diff = np.diff(magnitude, axis=1)
    spectral_flux = np.sqrt(np.sum(diff ** 2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])  # Pad to match length

    return SpectralFeatures(
        frequencies=frequencies,
        magnitudes=magnitude,
        spectral_centroid=spectral_centroid,
        spectral_rolloff=spectral_rolloff,
        spectral_flux=spectral_flux,
        sample_rate=sr,
        hop_length=hop_length
    )


def gpu_onset_detection(
    audio_path: str,
    sr: int = 22050,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    GPU-accelerated onset detection using spectral flux.

    Onsets are detected as local maxima in spectral flux above a threshold.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        threshold: Detection threshold (0-1, normalized)

    Returns:
        Array of onset times in seconds
    """
    # Get spectral features
    features = gpu_spectral_analysis(audio_path, sr)

    # Normalize flux
    flux_normalized = features.spectral_flux / (np.max(features.spectral_flux) + 1e-6)

    # Find peaks above threshold
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(flux_normalized, height=threshold, distance=sr // features.hop_length // 8)

    # Convert to time
    onset_times = peaks * features.hop_length / sr

    return onset_times


def benchmark_audio_gpu(audio_path: str) -> dict:
    """
    Benchmark GPU vs CPU audio analysis.

    Returns timing comparison dict.
    """
    import time

    global _gpu_backend

    results = {}

    # CPU baseline
    backend_original = _gpu_backend
    _gpu_backend = "cpu"

    start = time.perf_counter()
    _ = gpu_spectral_analysis(audio_path)
    cpu_time = time.perf_counter() - start
    results["cpu_seconds"] = cpu_time

    # Restore and test GPU
    _gpu_backend = backend_original
    backend = _detect_gpu_backend()

    if backend != "cpu":
        start = time.perf_counter()
        _ = gpu_spectral_analysis(audio_path)
        gpu_time = time.perf_counter() - start
        results["gpu_seconds"] = gpu_time
        results["gpu_backend"] = backend
        results["speedup"] = cpu_time / gpu_time if gpu_time > 0 else 0
    else:
        results["gpu_available"] = False

    return results


# Module exports
__all__ = [
    "SpectralFeatures",
    "is_gpu_audio_available",
    "gpu_spectral_analysis",
    "gpu_onset_detection",
    "benchmark_audio_gpu",
]
