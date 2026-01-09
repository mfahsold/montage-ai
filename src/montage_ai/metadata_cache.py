"""
Metadata Cache System for Montage-AI

Pre-computes and caches video metadata to eliminate redundant on-demand analysis.
Saves ~200MB of peak memory by avoiding repeated CV2 operations.

Cache files are stored as JSON sidecars: video.mp4.metadata.json
"""

import os
import json
import hashlib
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .logger import logger
from .config import get_settings


class MetadataCache:
    """Manages video metadata caching for performance optimization."""

    # Unified cache version shared with analysis cache (central settings)
    CACHE_VERSION = get_settings().cache.version

    def __init__(self, cache_dir: Optional[str] = None, invalidation_hours: int = 24):
        """
        Initialize metadata cache manager.

        Args:
            cache_dir: Directory to store cache files (default: same as video)
            invalidation_hours: Hours before cache is considered stale
        """
        # Allow override via environment variable if not explicitly provided
        if cache_dir is None:
            cache_dir = str(get_settings().paths.metadata_cache_dir)
            
        self.cache_dir = cache_dir
        self.invalidation_hours = invalidation_hours

    def get_cache_path(self, video_path: str) -> str:
        """Get the cache file path for a video."""
        if self.cache_dir:
            basename = os.path.basename(video_path)
            return os.path.join(self.cache_dir, f"{basename}.metadata.json")
        return f"{video_path}.metadata.json"

    def compute_video_hash(self, video_path: str) -> str:
        """
        Compute hash of video file for cache invalidation.

        Uses file size + mtime for performance (instead of full file hash).
        """
        stat = os.stat(video_path)
        hash_input = f"{video_path}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def is_cache_valid(self, video_path: str) -> bool:
        """
        Check if cache exists and is valid for a video.

        Returns True if cache exists, matches video hash, and isn't expired.
        """
        cache_path = self.get_cache_path(video_path)

        if not os.path.exists(cache_path):
            return False

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            # Check version
            if cache_data.get('version') != self.CACHE_VERSION:
                return False

            # Check video hash
            current_hash = self.compute_video_hash(video_path)
            if cache_data.get('video_hash') != current_hash:
                return False

            # Check expiration
            if self.invalidation_hours > 0:
                computed_at = datetime.fromisoformat(cache_data.get('computed_at', ''))
                expiry = computed_at + timedelta(hours=self.invalidation_hours)
                if datetime.now() > expiry:
                    return False

            return True

        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return False

    def load_metadata_cache(self, video_path: str) -> Optional[Dict]:
        """
        Load metadata from cache file.

        Returns None if cache doesn't exist or is invalid.
        """
        if not self.is_cache_valid(video_path):
            return None

        cache_path = self.get_cache_path(video_path)

        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def save_metadata_cache(self, video_path: str, metadata: Dict) -> None:
        """
        Save metadata to cache file.

        Args:
            video_path: Path to video file
            metadata: Metadata dictionary with 'scenes' key
        """
        cache_path = self.get_cache_path(video_path)

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)

        cache_data = {
            'version': self.CACHE_VERSION,
            'video_path': video_path,
            'computed_at': datetime.now().isoformat(),
            'video_hash': self.compute_video_hash(video_path),
            'scenes': metadata.get('scenes', [])
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except OSError as e:
            logger.warning(f"   ⚠️  Failed to save cache for {video_path}: {e}")

    def compute_visual_histogram(self, video_path: str, time_point: float, bins: int = 64) -> List[float]:
        """
        Compute visual histogram for a frame at specific time point.

        Used for match cut detection (visual similarity between clips).

        Args:
            video_path: Path to video file
            time_point: Time in seconds
            bins: Number of histogram bins (default: 64)

        Returns:
            Normalized histogram as list of floats
        """
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return [0.0] * bins

            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Compute histogram (Hue channel)
            hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])

            # Normalize
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-6)

            return hist.tolist()

        except Exception as e:
            logger.warning(f"   ⚠️  Histogram computation failed for {video_path} @ {time_point}s: {e}")
            return [0.0] * bins

    def compute_brightness(self, video_path: str, time_point: float) -> float:
        """
        Compute brightness level for a frame.

        Used for content-aware enhancement decisions.

        Returns:
            Brightness value 0.0-1.0 (0=black, 1=white)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return 0.5  # Default to medium brightness

            # Convert to grayscale and compute mean
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean() / 255.0

            return float(brightness)

        except Exception:
            return 0.5

    def compute_motion_blur_score(self, video_path: str, time_point: float) -> float:
        """
        Compute motion blur score using Laplacian variance.

        Higher score = more blur (lower focus quality).

        Returns:
            Blur score 0.0-1.0 (0=sharp, 1=very blurry)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return 0.5

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute Laplacian variance (focus measure)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            # Normalize: lower variance = more blur
            # Typical range: sharp ~1000, blurry ~100
            blur_score = 1.0 - min(variance / 1000.0, 1.0)

            return float(blur_score)

        except Exception:
            return 0.5

    def compute_optical_flow_magnitude(self, video_path: str, start_time: float, end_time: float,
                                       samples: int = 5) -> Tuple[float, str]:
        """
        Compute optical flow magnitude for a scene (motion intensity).

        Args:
            video_path: Path to video
            start_time: Scene start in seconds
            end_time: Scene end in seconds
            samples: Number of sample points

        Returns:
            Tuple of (magnitude 0-1, dominant_direction "left"/"right"/"static")
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Sample frames evenly across scene
            sample_times = np.linspace(start_time, end_time, samples + 1)

            magnitudes = []
            directions = []

            for i in range(len(sample_times) - 1):
                # Get two consecutive frames
                cap.set(cv2.CAP_PROP_POS_MSEC, sample_times[i] * 1000)
                ret1, frame1 = cap.read()

                cap.set(cv2.CAP_PROP_POS_MSEC, sample_times[i + 1] * 1000)
                ret2, frame2 = cap.read()

                if not (ret1 and ret2):
                    continue

                # Convert to grayscale
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

                # Compute magnitude and direction
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
                dir_x = flow[..., 0].mean()

                magnitudes.append(mag)
                directions.append(dir_x)

            cap.release()

            if not magnitudes:
                return 0.0, "static"

            # Average magnitude (normalize to 0-1, typical range 0-10)
            avg_magnitude = min(np.mean(magnitudes) / 10.0, 1.0)

            # Determine dominant direction
            avg_direction = np.mean(directions)
            if abs(avg_direction) < 0.5:
                direction = "static"
            elif avg_direction < 0:
                direction = "left"
            else:
                direction = "right"

            return float(avg_magnitude), direction

        except Exception:
            return 0.0, "static"

    def compute_scene_metadata(self, video_path: str, scene_start: float, scene_end: float,
                              scene_meta: Optional[Dict] = None) -> Dict:
        """
        Compute all metadata for a single scene.

        Args:
            video_path: Path to video file
            scene_start: Scene start time in seconds
            scene_end: Scene end time in seconds
            scene_meta: Optional existing metadata (action, shot, etc.)

        Returns:
            Complete scene metadata dictionary
        """
        from .video_metadata import probe_metadata
        from .proxy_generator import ProxyGenerator
        
        duration = scene_end - scene_start
        mid_point = scene_start + (duration / 2.0)
        
        # Determine if we should use proxy for analysis
        analysis_video_path = video_path
        try:
            settings = get_settings()
            metadata = probe_metadata(video_path)
            
            if settings.proxy.should_use_proxy(metadata.duration, metadata.width, metadata.height):
                logger.info(f"🎬 Using proxy analysis for optical flow (large video: {metadata.duration:.0f}s)")
                # Generate analysis proxy
                import tempfile
                ext = os.path.splitext(video_path)[1]
                proxy_fd, proxy_path = tempfile.mkstemp(suffix=f"_of_proxy{ext}", prefix="optical_flow_")
                try:
                    os.close(proxy_fd)
                except OSError:
                    pass
                
                try:
                    ProxyGenerator.generate_analysis_proxy(
                        source_path=video_path,
                        output_path=proxy_path,
                        height=settings.proxy.proxy_height
                    )
                    analysis_video_path = proxy_path
                    logger.debug(f"✓ Using proxy for optical flow analysis: {proxy_path}")
                except Exception as e:
                    logger.warning(f"Proxy generation failed for optical flow: {e}. Using original.")
                    if os.path.exists(proxy_path):
                        try:
                            os.unlink(proxy_path)
                        except:
                            pass
        except Exception as e:
            logger.debug(f"Could not determine if proxy needed: {e}. Using original video.")

        # Compute visual features (can use proxy)
        visual_histogram = self.compute_visual_histogram(analysis_video_path, mid_point)
        brightness = self.compute_brightness(analysis_video_path, mid_point)
        motion_blur = self.compute_motion_blur_score(analysis_video_path, mid_point)

        # Compute motion features (optical flow - benefit from proxy!)
        flow_magnitude, flow_direction = self.compute_optical_flow_magnitude(
            analysis_video_path, scene_start, scene_end
        )

        metadata = {
            'start': scene_start,
            'end': scene_end,
            'duration': duration,

            # Visual features
            'visual_histogram': visual_histogram,
            'brightness': brightness,
            'motion_blur_score': motion_blur,

            # Motion features
            'optical_flow_magnitude': flow_magnitude,
            'motion_direction': flow_direction,

            # Scene metadata (from existing analysis or defaults)
            'action': scene_meta.get('action', 'medium') if scene_meta else 'medium',
            'shot': scene_meta.get('shot', 'medium') if scene_meta else 'medium',
            'energy': scene_meta.get('energy', 0.5) if scene_meta else 0.5,
        }
        
        # Cleanup proxy if we created one
        if analysis_video_path != video_path:
            try:
                os.unlink(analysis_video_path)
                logger.debug(f"Cleaned up optical flow proxy: {analysis_video_path}")
            except Exception as e:
                logger.debug(f"Could not cleanup proxy {analysis_video_path}: {e}")

        return metadata

    def compute_video_metadata(self, video_path: str, scenes: List[Dict]) -> Dict:
        """
        Compute metadata for all scenes in a video.

        Args:
            video_path: Path to video file
            scenes: List of scene dicts with 'start', 'end', optional 'meta'

        Returns:
            Metadata dictionary with 'scenes' key
        """
        logger.info(f"   📊 Computing metadata for {os.path.basename(video_path)} ({len(scenes)} scenes)...")

        scene_metadata = []

        for i, scene in enumerate(scenes):
            scene_meta = self.compute_scene_metadata(
                video_path,
                scene['start'],
                scene['end'],
                scene.get('meta')
            )
            scene_metadata.append(scene_meta)

            if (i + 1) % 10 == 0:
                logger.info(f"      Processed {i + 1}/{len(scenes)} scenes...")

        return {'scenes': scene_metadata}

    def precompute_metadata(self, video_path: str, scenes: List[Dict], force: bool = False) -> Dict:
        """
        Pre-compute and cache metadata for a video.

        Args:
            video_path: Path to video file
            scenes: List of scene dicts
            force: Force recomputation even if cache exists

        Returns:
            Metadata dictionary (from cache or newly computed)
        """
        # Check cache first
        if not force:
            cached = self.load_metadata_cache(video_path)
            if cached:
                logger.info(f"   ✓ Using cached metadata for {os.path.basename(video_path)}")
                return cached

        # Compute metadata
        metadata = self.compute_video_metadata(video_path, scenes)

        # Save to cache
        self.save_metadata_cache(video_path, metadata)

        return metadata


def precompute_all_metadata(input_dir: str, cache_dir: Optional[str] = None,
                            force: bool = False) -> int:
    """
    Pre-compute metadata for all videos in a directory.

    Args:
        input_dir: Directory containing video files
        cache_dir: Directory to store cache files (default: same as videos)
        force: Force recomputation even if caches exist

    Returns:
        Number of videos processed
    """
    cache_manager = MetadataCache(cache_dir=cache_dir)

    # Find all video files
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mxf', '.mts', '.m2ts', '.ts')
    video_files = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))

    if not video_files:
        logger.warning(f"   ⚠️  No video files found in {input_dir}")
        return 0

    logger.info(f"\n   🔍 Pre-computing metadata for {len(video_files)} videos...")

    processed = 0
    skipped = 0

    for video_path in video_files:
        if cache_manager.is_cache_valid(video_path):
            skipped += 1
            logger.info(f"   ✓ Skipped (cached): {os.path.basename(video_path)}")
        else:
            # Note: This requires scene detection to be run first
            # In practice, this function will be called per-video after scene detection
            logger.info(f"   ℹ️  Metadata computation will be done during scene analysis for {os.path.basename(video_path)}")
            processed += 1

    logger.info(f"\n   ✅ Metadata pre-computation complete: {processed} computed, {skipped} from cache")

    return processed


# Singleton instance for easy access
_default_cache = None

def get_metadata_cache(cache_dir: Optional[str] = None) -> MetadataCache:
    """Get the default metadata cache instance."""
    global _default_cache
    if _default_cache is None:
        settings = get_settings()
        invalidation_hours = int(settings.cache.metadata_ttl_hours)
        # Check for explicit override; else use settings path
        if cache_dir is None:
            cache_dir = str(settings.paths.metadata_cache_dir)
        _default_cache = MetadataCache(cache_dir=cache_dir, invalidation_hours=invalidation_hours)
    return _default_cache
