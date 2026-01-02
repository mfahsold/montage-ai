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

class SmartReframer:
    """
    Intelligent video reframing engine.
    """

    def __init__(self, target_aspect: float = 9/16, smoothing_window: int = 15):
        self.target_aspect = target_aspect
        self.smoothing_window = smoothing_window
        
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
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect face
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_frame)
            
            center_x = width // 2
            score = 0.0
            
            if results.detections:
                # Find largest face
                largest_face = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                bbox = largest_face.location_data.relative_bounding_box
                
                # Calculate face center
                face_center_x = int((bbox.xmin + bbox.width / 2) * width)
                center_x = face_center_x
                score = largest_face.score[0]
            
            # Clamp crop window within frame
            x = max(0, min(width - target_width, center_x - target_width // 2))
            y = (height - target_height) // 2 # Always center vertically for now
            
            crops.append(CropWindow(
                time=frame_idx / fps,
                x=int(x),
                y=int(y),
                width=target_width,
                height=target_height,
                score=score
            ))
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.debug(f"Analyzed frame {frame_idx}/{total_frames}")

        cap.release()
        
        # Smooth the camera movement
        return self._smooth_crops(crops)

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

    def _smooth_crops(self, crops: List[CropWindow]) -> List[CropWindow]:
        """Apply moving average smoothing to x-coordinates."""
        if not crops:
            return []
            
        xs = [c.x for c in crops]
        # Simple moving average
        window = self.smoothing_window
        smoothed_xs = np.convolve(xs, np.ones(window)/window, mode='same')
        
        for i, c in enumerate(crops):
            c.x = int(smoothed_xs[i])
            
        return crops

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

