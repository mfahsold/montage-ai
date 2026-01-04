"""
Auto Reframe Engine - AI-Powered Aspect Ratio Conversion

Converts landscape (16:9) video to vertical (9:16) or square (1:1)
by tracking the main subject (face/object) and keeping it centered.

Uses:
- MediaPipe for face detection (fast, CPU-friendly)
- OpenCV for object tracking (KCF/CSRT) if no face found
- Smooth camera motion simulation (damping) to avoid jitter

Usage:
    from montage_ai.smart_reframing import AutoReframeEngine

    reframer = AutoReframeEngine(target_aspect=9/16)
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
from .ffmpeg_utils import build_ffmpeg_cmd


# =============================================================================
# Kalman Filter for Subject Tracking Smoothing
# =============================================================================

class SubjectKalmanFilter:
    """
    1D Kalman filter for smoothing subject position tracking.

    State: [position, velocity]
    Measurement: position only

    Provides smoother transitions than moving average, especially when
    detections are lost (prediction step extrapolates motion).
    """

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        initial_position: float = 0.0
    ):
        # State: [position, velocity]
        self.x = np.array([initial_position, 0.0])

        # State covariance
        self.P = np.eye(2) * 100.0

        # State transition matrix (position += velocity * dt, dt=1 frame)
        self.F = np.array([[1, 1], [0, 1]])

        # Measurement matrix (we only observe position)
        self.H = np.array([[1, 0]])

        # Process noise covariance (how much we expect state to change)
        self.Q = np.array([
            [process_noise, 0],
            [0, process_noise * 0.1]  # velocity noise is smaller
        ])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

    def predict(self) -> float:
        """Predict next state (position)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]

    def update(self, measurement: float) -> float:
        """Update state with new measurement."""
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        y = np.array([measurement]) - self.H @ self.x  # Innovation
        self.x = self.x + (K @ y).flatten()

        # Update covariance
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        return self.x[0]

    @staticmethod
    def smooth_sequence(
        raw_centers: List[int],
        process_noise: float = 1.0,
        measurement_noise: float = 10.0
    ) -> List[int]:
        """
        Smooth a sequence of positions using forward-backward Kalman filtering.

        This achieves similar results to convex optimization but without scipy.
        """
        if len(raw_centers) < 2:
            return raw_centers

        # Forward pass
        forward_kf = SubjectKalmanFilter(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_position=raw_centers[0]
        )
        forward_estimates = []
        forward_covariances = []

        for z in raw_centers:
            forward_kf.predict()
            forward_kf.update(z)
            forward_estimates.append(forward_kf.x.copy())
            forward_covariances.append(forward_kf.P.copy())

        # Backward pass (RTS smoother)
        n = len(raw_centers)
        smoothed = [np.zeros(2) for _ in range(n)]
        smoothed[-1] = forward_estimates[-1]

        for i in range(n - 2, -1, -1):
            P_pred = forward_kf.F @ forward_covariances[i] @ forward_kf.F.T + forward_kf.Q
            C = forward_covariances[i] @ forward_kf.F.T @ np.linalg.inv(P_pred)
            smoothed[i] = forward_estimates[i] + C @ (smoothed[i + 1] - forward_kf.F @ forward_estimates[i])

        return [int(s[0]) for s in smoothed]

# Try importing MediaPipe, but don't crash if missing (optional dependency)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logger.warning("MediaPipe not installed. Auto Reframe will fallback to center crop.")

@dataclass
class Keyframe:
    """A manual override point for the camera path."""
    time: float
    center_x_norm: float # Normalized 0.0-1.0 relative to video width

@dataclass
class CropWindow:
    """A crop window at a specific timestamp."""
    time: float
    x: int
    y: int
    width: int
    height: int
    score: float  # Confidence score of the subject

class CameraMotionOptimizer:
    """
    Solves for an optimal camera path using convex optimization (L2 regularization).
    Minimizes: Distance to Subject + Velocity (Shake) + Acceleration (Jerk).
    
    Cost Function:
    J(x) = Σ(x_t - c_t)² + λ1 * Σ(x_t - x_{t-1})² + λ2 * Σ(x_t - 2x_{t-1} + x_{t-2})²
    """
    
    def __init__(self, lambda_smooth: float = 500.0, lambda_trend: float = 50.0):
        self.lambda_smooth = lambda_smooth # Penalizes velocity (keeps camera still)
        self.lambda_trend = lambda_trend   # Penalizes acceleration (smooths starts/stops)

    def solve(self, raw_centers: List[int], constraints: Dict[int, int] = None) -> List[int]:
        """
        Solve the linear system to find the optimal path.
        
        Args:
            raw_centers: List of detected subject centers per frame.
            constraints: Optional dict mapping frame_index -> target_center_x (pixels).
                         Used for manual keyframes.
        """
        if not SCIPY_AVAILABLE:
            logger.warning("Scipy not available. Falling back to simple moving average.")
            return self._fallback_smooth(raw_centers, constraints)
            
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
        
        # Apply constraints (Keyframes)
        if constraints:
            # Add penalty term: λ_constraint * Σ (x_i - target_i)^2
            # This modifies the diagonal of A and the vector b.
            lambda_constraint = 10000.0 # Strong pull towards keyframe
            
            constraint_weights = np.zeros(n)
            constraint_targets = np.zeros(n)
            
            for idx, target in constraints.items():
                if 0 <= idx < n:
                    constraint_weights[idx] = lambda_constraint
                    constraint_targets[idx] = target
            
            # Add diagonal matrix to A
            C = sparse.diags(constraint_weights, 0, shape=(n, n))
            A = A + C
            
            # Add to b
            b = b + constraint_weights * constraint_targets

        # Solve
        try:
            smoothed_path = spsolve(A, b)
            return [int(x) for x in smoothed_path]
        except Exception as e:
            logger.error(f"Optimization failed: {e}. Using fallback.")
            return self._fallback_smooth(raw_centers, constraints)

    def _fallback_smooth(self, data: List[int], constraints: Dict[int, int] = None, window: int = 15) -> List[int]:
        """Kalman filter fallback for smooth camera paths (when scipy unavailable)."""
        if not data:
            return []

        # Apply constraints by blending with original data
        working_data = list(data)
        if constraints:
            for idx, target in constraints.items():
                if 0 <= idx < len(working_data):
                    working_data[idx] = target

        # Use Kalman filter for smoother results than moving average
        return SubjectKalmanFilter.smooth_sequence(
            working_data,
            process_noise=1.0 / self.lambda_smooth,  # Lower λ = more smoothing
            measurement_noise=10.0
        )


class AutoReframeEngine:
    """
    Intelligent video reframing engine.
    """

    def __init__(self, target_aspect: float = 9/16, smoothing_window: int = 15):
        self.target_aspect = target_aspect
        self.smoothing_window = smoothing_window
        self.path_planner = CameraMotionOptimizer(lambda_smooth=100.0, lambda_trend=10.0)
        
        if MP_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1, # 1 = full range (better for video)
                min_detection_confidence=0.6
            )

    # Backward compatibility alias
    SmartReframer = None # Will be set after class definition if needed, but better to just rename usages.

    def _calculate_iou(self, box1, box2) -> float:
        """Calculate Intersection over Union (IoU) between two normalized bounding boxes."""
        # box: [xmin, ymin, width, height]
        x1_min, y1_min, w1, h1 = box1.xmin, box1.ymin, box1.width, box1.height
        x1_max, y1_max = x1_min + w1, y1_min + h1
        
        x2_min, y2_min, w2, h2 = box2.xmin, box2.ymin, box2.width, box2.height
        x2_max, y2_max = x2_min + w2, y2_min + h2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_w = max(0, xi_max - xi_min)
        inter_h = max(0, yi_max - yi_min)
        intersection = inter_w * inter_h
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def analyze(self, video_path: str, keyframes: List[Keyframe] = None) -> List[CropWindow]:
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
        
        # Tracking state
        last_center_x = width // 2
        active_subject_bbox = None
        frames_since_detection = 0
        MAX_LOST_FRAMES = int(fps * 1.0) # Reset tracking after 1 second of loss
        
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
            current_best_face = None
            
            if results.detections:
                if active_subject_bbox is None or frames_since_detection > MAX_LOST_FRAMES:
                    # No active subject or lost track -> Find largest face
                    current_best_face = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                    frames_since_detection = 0 # Reset counter
                else:
                    # We have an active subject -> Find best match (IoU)
                    best_iou = -1.0
                    for detection in results.detections:
                        iou = self._calculate_iou(active_subject_bbox, detection.location_data.relative_bounding_box)
                        if iou > best_iou:
                            best_iou = iou
                            current_best_face = detection
                    
                    # If match is too poor, consider it lost (or scene cut)
                    if best_iou < 0.1:
                        # Fallback to largest if tracking fails completely? 
                        # Or just hold position? Let's hold position for now, but check if there's a HUGE face we are ignoring.
                        # For stability, let's just increment lost counter.
                        current_best_face = None 
                    else:
                        frames_since_detection = 0

                if current_best_face:
                    bbox = current_best_face.location_data.relative_bounding_box
                    
                    # Update active subject
                    active_subject_bbox = bbox
                    
                    # Calculate face center
                    face_center_x = int((bbox.xmin + bbox.width / 2) * width)
                    center_x = face_center_x
                    score = current_best_face.score[0]
                    last_center_x = center_x # Update last known
                else:
                    frames_since_detection += 1
            else:
                frames_since_detection += 1
            
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
        
        # Convert keyframes to constraints
        constraints = {}
        if keyframes:
            for kf in keyframes:
                f_idx = int(kf.time * fps)
                f_idx = max(0, min(f_idx, len(raw_centers) - 1))
                pixel_x = int(kf.center_x_norm * width)
                constraints[f_idx] = pixel_x

        # Apply Cinematic Path Planning (Global Optimization)
        logger.info("Optimizing camera path...")
        smoothed_centers = self.path_planner.solve(raw_centers, constraints)
        
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

    def apply(self, crops: Optional[List[CropWindow]], input_path: str, output_path: str) -> None:
        """
        Apply the calculated crops using FFmpeg.
        Uses segmented cropping to simulate camera cuts/pans.
        If crops is None, applies a static center crop.
        """
        if crops is None:
            # Fallback to center crop
            logger.info("No crops provided. Applying center crop.")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"Could not open {input_path}")
                return
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            target_w, target_h = self._calculate_crop_dims(width, height)
            x = (width - target_w) // 2
            y = (height - target_h) // 2
            
            cmd = build_ffmpeg_cmd([
                "-i", input_path,
                "-vf", f"crop={target_w}:{target_h}:{x}:{y}",
                "-c:a", "copy",
                output_path
            ])
            run_command(cmd)
            return

        segments = self._segment_crops(crops)
        
        if not segments:
            logger.warning("No crop segments found.")
            return

        logger.info(f"Applying smart crop with {len(segments)} segments...")

        # If only one segment, simple crop
        if len(segments) == 1:
            seg = segments[0]
            cmd = build_ffmpeg_cmd([
                "-i", input_path,
                "-vf", f"crop={seg['w']}:{seg['h']}:{seg['x']}:{seg['y']}",
                "-c:a", "copy",
                output_path
            ])
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
        
        cmd = build_ffmpeg_cmd([
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",  # Map audio if exists
            "-c:a", "copy",
            output_path
        ])
        
        run_command(cmd)


# Backward compatibility aliases
SmartReframer = AutoReframeEngine
CinematicPathPlanner = CameraMotionOptimizer
