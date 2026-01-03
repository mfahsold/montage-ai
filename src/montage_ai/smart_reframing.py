"""
Smart Reframing - AI-Powered Aspect Ratio Conversion

Converts landscape (16:9) video to vertical (9:16) or square (1:1)
by tracking the main subject (face/object) and keeping it centered.

Uses:
- MediaPipe for face detection (fast, CPU-friendly)
- OpenCV for object tracking (KCF/CSRT) if no face found
- Smooth camera motion simulation (damping) to avoid jitter

Usage:
    from montage_ai.smart_reframing import SmartReframer

    reframer = SmartReframer(target_aspect=9/16)
    crop_data = reframer.analyze("input.mp4")
    reframer.apply(crop_data, "input.mp4", "output_vertical.mp4")
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Mathematical optimization for smooth camera paths
try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .logger import logger
from .core.cmd_runner import run_command

# Try importing MediaPipe, but don't crash if missing (optional dependency)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logger.warning("MediaPipe not installed. Smart Reframing will fallback to center crop.")

@dataclass
class CropWindow:
    """A crop window at a specific timestamp."""
    time: float
    x: int
    y: int
    width: int
    height: int
    score: float  # Confidence score of the subject

class CinematicPathPlanner:
    """
    Solves for an optimal camera path using convex optimization (L2 regularization).
    Minimizes: Distance to Subject + Velocity (Shake) + Acceleration (Jerk).
    
    Cost Function:
    J(x) = Σ(x_t - c_t)² + λ1 * Σ(x_t - x_{t-1})² + λ2 * Σ(x_t - 2x_{t-1} + x_{t-2})²
    """
    
    def __init__(self, lambda_smooth: float = 500.0, lambda_trend: float = 50.0):
        self.lambda_smooth = lambda_smooth # Penalizes velocity (keeps camera still)
        self.lambda_trend = lambda_trend   # Penalizes acceleration (smooths starts/stops)

    def solve(self, raw_centers: List[int]) -> List[int]:
        """
        Solve the linear system to find the optimal path.
        """
        if not SCIPY_AVAILABLE:
            logger.warning("Scipy not available. Falling back to simple moving average.")
            return self._fallback_smooth(raw_centers)
            
        n = len(raw_centers)
        if n < 3:
            return raw_centers

        # Construct sparse matrices
        # Identity matrix (Data term)
        I = sparse.eye(n)
        
        # First difference matrix (Velocity term)
        # [ -1  1  0 ... ]
        # [  0 -1  1 ... ]
        D1 = sparse.diags([-1, 1], [0, 1], shape=(n-1, n))
        
        # Second difference matrix (Acceleration term)
        # [  1 -2  1  0 ... ]
        # [  0  1 -2  1 ... ]
        D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n))
        
        # Construct the linear system (A * x = b)
        # Derivative of Cost Function set to 0:
        # (I + λ1*D1.T*D1 + λ2*D2.T*D2) * x = raw_centers
        
        A = I + self.lambda_smooth * (D1.T @ D1) + self.lambda_trend * (D2.T @ D2)
        b = np.array(raw_centers)
        
        # Solve
        try:
            smoothed_path = spsolve(A, b)
            return [int(x) for x in smoothed_path]
        except Exception as e:
            logger.error(f"Optimization failed: {e}. Using fallback.")
            return self._fallback_smooth(raw_centers)

    def _fallback_smooth(self, data: List[int], window: int = 15) -> List[int]:
        """Simple moving average fallback."""
        return [int(x) for x in np.convolve(data, np.ones(window)/window, mode='same')]


class SmartReframer:
    """
    Intelligent video reframing engine.
    """

    def __init__(self, target_aspect: float = 9/16, smoothing_window: int = 15):
        self.target_aspect = target_aspect
        self.smoothing_window = smoothing_window
        self.path_planner = CinematicPathPlanner(lambda_smooth=100.0, lambda_trend=10.0)
        
        if MP_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1, # 1 = full range (better for video)
                min_detection_confidence=0.6
            )

    def analyze(self, video_path: str) -> List[CropWindow]:
        """
        Analyze video and calculate optimal crop windows per frame.
        Returns list of CropWindow objects.
        """
        if not MP_AVAILABLE:
            return self._fallback_center_crop(video_path)

        logger.info(f"Analyzing {Path(video_path).name} for smart reframing...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate target crop size
        target_width, target_height = self._calculate_crop_dims(width, height)
        
        crops = []
        raw_centers = []
        frame_idx = 0
        
        # Track last known good position to handle dropouts
        last_center_x = width // 2
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect face
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_frame)
            
            center_x = last_center_x # Default to last known
            score = 0.0
            
            if results.detections:
                # Find largest face
                largest_face = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                bbox = largest_face.location_data.relative_bounding_box
                
                # Calculate face center
                face_center_x = int((bbox.xmin + bbox.width / 2) * width)
                center_x = face_center_x
                score = largest_face.score[0]
                last_center_x = center_x # Update last known
            
            raw_centers.append(center_x)
            
            # Placeholder crop (will be updated after smoothing)
            crops.append(CropWindow(
                time=frame_idx / fps,
                x=0, # To be filled
                y=int((height - target_height) // 2),
                width=target_width,
                height=target_height,
                score=score
            ))
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.debug(f"Analyzed frame {frame_idx}/{total_frames}")

        cap.release()
        
        # Apply Cinematic Path Planning (Global Optimization)
        logger.info("Optimizing camera path...")
        smoothed_centers = self.path_planner.solve(raw_centers)
        
        # Update crops with smoothed centers
        for i, crop in enumerate(crops):
            # Clamp to frame boundaries
            center = smoothed_centers[i]
            x = max(0, min(width - target_width, center - target_width // 2))
            crop.x = int(x)
            
        return crops

    def _fallback_smooth(self, data: List[int], window: int = 15) -> List[int]:
        """Simple moving average fallback."""
        return [int(x) for x in np.convolve(data, np.ones(window)/window, mode='same')]

    def _calculate_crop_dims(self, src_w: int, src_h: int) -> Tuple[int, int]:
        """Calculate crop dimensions maintaining target aspect ratio."""
        # If source is already narrower than target, fit width
        if src_w / src_h < self.target_aspect:
            return src_w, int(src_w / self.target_aspect)
        
        # Otherwise fit height (standard for 16:9 -> 9:16)
        return int(src_h * self.target_aspect), src_h

    def _fallback_center_crop(self, video_path: str) -> List[CropWindow]:
        """Simple center crop if AI unavailable."""
        logger.info("Using fallback center crop...")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        target_w, target_h = self._calculate_crop_dims(width, height)
        x = (width - target_w) // 2
        y = (height - target_h) // 2
        
        return [
            CropWindow(time=i/fps, x=x, y=y, width=target_w, height=target_h, score=0.0)
            for i in range(total_frames)
        ]

    def _segment_crops(self, crops: List[CropWindow], min_segment_duration: float = 2.0) -> List[Dict]:
        """
        Group frame-level crops into stable segments.
        Returns list of dicts: {'start': float, 'end': float, 'x': int, 'y': int, 'w': int, 'h': int}
        """
        if not crops:
            return []

        segments = []
        current_start_idx = 0
        current_x_sum = 0
        current_count = 0
        
        width = crops[0].width
        height = crops[0].height
        
        # Threshold for creating a new cut (in pixels)
        # If the subject moves significantly (> 25% of width), we cut to a new angle
        threshold = width * 0.25
        
        for i, crop in enumerate(crops):
            # Check if we should cut
            duration = crop.time - crops[current_start_idx].time
            
            if duration >= min_segment_duration:
                # Calculate average of frames SO FAR (excluding current if it's the outlier)
                if current_count > 0:
                    avg_x = current_x_sum / current_count
                else:
                    avg_x = crop.x

                # If current frame deviates too much from segment average, start new segment
                if abs(crop.x - avg_x) > threshold:
                    # Finalize current segment (excluding current frame)
                    segments.append({
                        'start': crops[current_start_idx].time,
                        'end': crop.time,
                        'x': int(avg_x),
                        'y': crop.y,
                        'w': width,
                        'h': height
                    })
                    # Start new segment with current frame
                    current_start_idx = i
                    current_x_sum = crop.x 
                    current_count = 1
                    continue

            current_x_sum += crop.x
            current_count += 1

        # Add final segment
        if current_count > 0:
            avg_x = current_x_sum / current_count
            segments.append({
                'start': crops[current_start_idx].time,
                'end': crops[-1].time, # Use last frame time
                'x': int(avg_x),
                'y': crops[-1].y,
                'w': width,
                'h': height
            })
            
        return segments

    def apply(self, crops: List[CropWindow], input_path: str, output_path: str) -> None:
        """
        Apply the calculated crops using FFmpeg.
        Uses segmented cropping to simulate camera cuts/pans.
        """
        segments = self._segment_crops(crops)
        
        if not segments:
            logger.warning("No crop segments found.")
            return

        logger.info(f"Applying smart crop with {len(segments)} segments...")

        # If only one segment, simple crop
        if len(segments) == 1:
            seg = segments[0]
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", f"crop={seg['w']}:{seg['h']}:{seg['x']}:{seg['y']}",
                "-c:a", "copy",
                output_path
            ]
            run_command(cmd)
            return

        # Multiple segments: complex filter
        # [0:v]trim=start=0:end=2,setpts=PTS-STARTPTS,crop=...[v0];
        # [0:v]trim=start=2:end=5,setpts=PTS-STARTPTS,crop=...[v1];
        # [v0][v1]concat=n=2:v=1:a=0[outv]
        
        filter_complex = ""
        inputs = ""
        
        for i, seg in enumerate(segments):
            # Ensure we don't exceed video duration (handled by trim usually)
            # trim expects seconds
            filter_complex += f"[0:v]trim=start={seg['start']}:end={seg['end']},setpts=PTS-STARTPTS,crop={seg['w']}:{seg['h']}:{seg['x']}:{seg['y']}[v{i}];"
            inputs += f"[v{i}]"
            
        filter_complex += f"{inputs}concat=n={len(segments)}:v=1:a=0[outv]"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?", # Map audio if exists
            "-c:a", "copy",
            output_path
        ]
        
        run_command(cmd)

