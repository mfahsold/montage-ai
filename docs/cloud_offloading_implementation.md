# Cloud Offloading Implementation Plan

This document details the implementation for offloading CPU-intensive tasks to Cloud GPU using `cgpu`.

## Overview

We have extended the `CGPUJob` architecture to support three new job types:
1.  **Scene Detection**: Offloads `PySceneDetect` to cloud.
2.  **Beat Analysis**: Offloads `librosa` beat tracking to cloud.
3.  **FFmpeg Rendering**: Offloads video transcoding/rendering to cloud (with NVENC support).

## 1. Scene Detection (`SceneDetectionJob`)

**Goal**: Move `scenedetect` execution to the cloud to prevent local CPU exhaustion.

**Implementation**:
-   **Class**: `montage_ai.cgpu_jobs.SceneDetectionJob`
-   **Packaging**: Generates a temporary Python script (`detect_scenes.py`) that imports `scenedetect` and runs `ContentDetector`.
-   **Requirements**: `scenedetect[opencv]`, `opencv-python-headless`, `numpy`.
-   **Execution**: Uploads video and script, runs `python detect_scenes.py`.
-   **Result**: Downloads `scenes.json` containing start/end times and frame numbers.

**Usage**:
```python
from montage_ai.cgpu_jobs import SceneDetectionJob

job = SceneDetectionJob(input_path="data/input/video.mp4", threshold=30.0)
result = job.execute()

if result.success:
    print(f"Scenes saved to: {result.output_path}")
    # Load JSON and use in pipeline
```

## 2. Beat Analysis (`BeatAnalysisJob`)

**Goal**: Move `librosa` analysis to the cloud.

**Implementation**:
-   **Class**: `montage_ai.cgpu_jobs.BeatAnalysisJob`
-   **Packaging**: Generates a temporary Python script (`analyze_audio.py`) that uses `librosa` to extract beat times and energy profile.
-   **Requirements**: `librosa`, `numpy`, `soundfile`.
-   **Execution**: Uploads audio and script, runs `python analyze_audio.py`.
-   **Result**: Downloads `analysis.json` containing tempo, beat times, and energy RMS.

**Usage**:
```python
from montage_ai.cgpu_jobs import BeatAnalysisJob

job = BeatAnalysisJob(input_path="data/music/track.mp3")
result = job.execute()

if result.success:
    print(f"Analysis saved to: {result.output_path}")
```

## 3. Transcoding/Rendering (`FFmpegRenderJob`)

**Goal**: Use Cloud GPU (NVENC) for video encoding.

**Implementation**:
-   **Class**: `montage_ai.cgpu_jobs.FFmpegRenderJob`
-   **Packaging**: Uses the pre-installed `ffmpeg` on the cloud instance.
-   **Execution**: Uploads input files, runs the specified FFmpeg command.
-   **Result**: Downloads the output video file.

**NVENC Support**:
To use hardware acceleration, include `-c:v h264_nvenc` (or `hevc_nvenc`) in your command arguments.

**Usage**:
```python
from montage_ai.cgpu_jobs import FFmpegRenderJob

# Example: Transcode to H.264 using NVENC
input_file = "data/input/raw.mov"
output_file = "compressed.mp4"

# Helper to create NVENC command
cmd_args = FFmpegRenderJob.create_nvenc_command(
    input_file=Path(input_file).name, # Use basename for remote command
    output_file=output_file,
    bitrate="5M"
)

job = FFmpegRenderJob(
    input_paths=[input_file],
    command_args=cmd_args,
    output_filename=output_file
)
result = job.execute()
```

## Integration Strategy

To integrate this into the main pipeline (`montage-ai.sh` / `editor.py`):

1.  **Check Availability**: Use `montage_ai.cgpu_utils.is_cgpu_available()` to decide whether to run locally or remotely.
2.  **Fallback**: If `cgpu` is not configured or fails, fall back to the local CPU implementation.
3.  **Parallelism**: Use `CGPUJobManager` to submit multiple analysis jobs (e.g., for multiple clips) and process them in parallel if the cloud environment supports it (though Colab is usually single-session, `cgpu` might handle queuing).

## Error Handling

-   All jobs return a `JobResult` object.
-   Check `result.success` before proceeding.
-   Logs are printed to stdout/stderr during execution.
