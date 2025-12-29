# Cloud Pipeline Technical Specification (Phase 3)

## 1. Overview
This document defines the technical implementation details for the **Cloud Pipeline Refactoring**. We will transition from ad-hoc `cgpu` calls in scattered modules to a unified **Job-Based Architecture**. This ensures reliability, better error handling, and easier addition of new cloud capabilities (like Stabilization).

## 2. Class Architecture

### 2.1. Base Class: `CGPUJob`
Located in: `src/montage_ai/cgpu_jobs/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class JobResult:
    success: bool
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

class CGPUJob(ABC):
    def __init__(self, job_id: str = None):
        self.job_id = job_id or str(uuid.uuid4())
        self.remote_work_dir = f"/content/job_{self.job_id}"
        self.status = JobStatus.PENDING

    @abstractmethod
    def prepare_local(self) -> bool:
        """Validate inputs and prepare local assets."""
        pass

    @abstractmethod
    def get_requirements(self) -> list[str]:
        """Return list of pip packages to install remotely."""
        return []

    @abstractmethod
    def upload(self) -> bool:
        """Upload assets to remote_work_dir."""
        pass

    @abstractmethod
    def run_remote(self) -> bool:
        """Execute the main logic on cgpu."""
        pass

    @abstractmethod
    def download(self) -> JobResult:
        """Retrieve results and return JobResult."""
        pass

    def cleanup(self):
        """Remove remote and local temp files."""
        run_cgpu_command(f"rm -rf {self.remote_work_dir}")
```

### 2.2. Job Manager: `CGPUJobManager`
Located in: `src/montage_ai/cgpu_jobs/manager.py`

*   **Singleton Pattern:** Ensures only one manager exists.
*   **Queue:** `collections.deque` for FIFO execution.
*   **Session Management:** Tracks if the remote environment has been initialized (e.g., `apt-get update` run once).

```python
class CGPUJobManager:
    def submit(self, job: CGPUJob) -> str:
        """Add job to queue, return job_id."""
        pass
        
    def process_queue(self):
        """Process all pending jobs sequentially."""
        pass
```

## 3. Job Implementations

### 3.1. `TranscribeJob` (Refactor)
*   **Input:** Audio file path, model size.
*   **Requirements:** `openai-whisper`.
*   **Command:** `whisper input.wav --model medium --output_format srt`.
*   **Output:** `.srt` file path.

### 3.2. `UpscaleJob` (Refactor)
*   **Input:** Video/Image path, scale factor.
*   **Requirements:** `realesrgan-ncnn-vulkan` (binary) or python package.
*   **Command:** `./realesrgan-ncnn-vulkan -i input.mp4 -o output.mp4 -s 4`.
*   **Output:** Upscaled video path.

### 3.3. `StabilizeJob` (New)
*   **Input:** Video path, smoothing parameters.
*   **Requirements:** `ffmpeg` (usually pre-installed on Colab).
*   **Command:**
    1.  `ffmpeg -i input.mp4 -vf vidstabdetect=stepsize=32:shakiness=5:accuracy=15:result=transform.trf -f null -`
    2.  `ffmpeg -i input.mp4 -vf vidstabtransform=input=transform.trf:zoom=0:smoothing=10 output.mp4`
*   **Output:** Stabilized video path.

## 4. Directory Structure

```
src/montage_ai/
├── cgpu_jobs/
│   ├── __init__.py
│   ├── base.py          # Abstract Base Class
│   ├── manager.py       # Queue & Execution Logic
│   ├── transcribe.py    # Whisper Job
│   ├── upscale.py       # Real-ESRGAN Job
│   └── stabilize.py     # VidStab Job
```

## 5. Migration Plan

1.  **Create Infrastructure:** Implement `base.py` and `manager.py`.
2.  **Port Transcriber:** Rewrite `transcriber.py` to use `TranscribeJob` internally (maintaining API compatibility for now).
3.  **Port Upscaler:** Rewrite `cgpu_upscaler.py` to use `UpscaleJob`.
4.  **Add Stabilization:** Implement `StabilizeJob` and expose it via `manager`.
5.  **Update Editor:** Modify `editor.py` to submit jobs to the manager instead of calling functions directly.

## 6. Error Handling Strategy

*   **Connection Failures:** `CGPUJobManager` will retry `cgpu` commands up to 3 times with exponential backoff.
*   **Remote Errors:** Capture `stderr` from Colab. If a job fails, mark as `FAILED` and preserve logs.
*   **Timeouts:** Each job type defines a `timeout` (e.g., 10 mins for transcription, 30 mins for upscaling).

## 7. Future Optimizations

*   **Batching:** If multiple small clips need stabilization, upload them as a zip, process in a loop remotely, download as zip.
*   **Parallelism:** If the user has multiple Colab accounts/instances, the Manager could dispatch to multiple backends (future scope).
