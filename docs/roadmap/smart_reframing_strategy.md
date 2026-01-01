# Strategy: Smart Reframing & Face Detection (SOTA 2025)

> **Status**: Research Phase
> **Goal**: Implement "Smart Crop" (9:16) and "Face-Aware Editing" locally.

## The Challenge
To fulfill the "Smart Reframing" promise (P1 feature) and the "identifies faces" marketing claim, we need a robust, local-first solution that:
1.  Detects faces and "salient" action.
2.  Calculates a dynamic crop window (e.g., keeping the subject in the center of a 9:16 frame).
3.  Runs on consumer hardware (CPU/GPU) without cloud API costs.

## SOTA Candidates (2024/2025)

### 1. MediaPipe (Google)
*   **Pros**: Extremely fast, CPU-optimized, easy Python API (`mediapipe-python`).
*   **Cons**: Can struggle with extreme angles or occlusion compared to heavier models.
*   **Verdict**: **Best for MVP**. It fits the "Junior Developer" persona (easy to integrate) and the "Offline" promise.

### 2. YOLOv8-Face / YOLOv11
*   **Pros**: Higher accuracy, detects smaller faces, GPU-accelerated.
*   **Cons**: Heavier dependency (Ultralytics/PyTorch), slower on CPU.
*   **Verdict**: Good upgrade path if MediaPipe fails.

### 3. DeepFace
*   **Pros**: Wrapper for many models (VGG-Face, FaceNet). Good for *recognition* (who is it?).
*   **Cons**: Overkill for *detection* (where is it?). Slower.

## Implementation Plan

### Phase 1: The "Smart Crop" Module (`smart_crop.py`)
1.  **Input**: 16:9 Video.
2.  **Process**:
    *   Sample frames (e.g., 5fps).
    *   Run MediaPipe Face Detection.
    *   Calculate "Center of Interest" (centroid of faces).
    *   Apply temporal smoothing (Kalman filter or simple moving average) to prevent jittery camera movement.
3.  **Output**: FFmpeg `crop` filter command or a new video file.

### Phase 2: Integration
*   Add `--smart-crop` flag to `montage-ai.sh`.
*   Use detection data to score clips (more faces = higher score for "party" style).

## Recommendation
Start with **MediaPipe**. It has zero external model weights to download (bundled) and runs everywhere.

```python
# Prototype Idea
import mediapipe as mp
import cv2

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    # ... process frames ...
```
