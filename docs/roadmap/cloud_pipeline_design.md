# Cloud Pipeline Architecture (cgpu Jobs)

## 1. Executive Summary
To support the "Hybrid Workflow" (Local Intelligence + Cloud Muscle), we need a generalized system for offloading tasks to Google Colab/Kaggle via `cgpu`. Currently, `cgpu_upscaler.py` and `transcriber.py` have duplicated logic. We will unify this into a **Job-Based Pipeline**.

## 2. Architecture

### The `CGPUJob` Abstraction

We will define a base class `CGPUJob` that handles the lifecycle of a remote task:

1.  **Prepare:** Create local temp dir, gather assets.
2.  **Upload:** Transfer assets to Colab (`/content/job_id/`).
3.  **Execute:** Run shell command on Colab.
4.  **Download:** Retrieve results.
5.  **Cleanup:** Remove remote and local temp files.

```python
class CGPUJob(ABC):
    @abstractmethod
    def prepare(self): ...
    
    @abstractmethod
    def get_remote_command(self) -> str: ...
    
    def run(self):
        # Template method pattern handling the lifecycle
```

### Job Types

| Job Class | Input | Remote Tool | Output |
| :--- | :--- | :--- | :--- |
| `UpscaleJob` | Video/Image | Real-ESRGAN | Upscaled Video/Image |
| `TranscribeJob` | Audio | OpenAI Whisper | .srt/.vtt file |
| `StabilizeJob` | Video | FFmpeg `vidstab` | Stabilized Video |
| `AnalysisJob` | Video | CLIP/BLIP | JSON Metadata |

## 3. Queue Management

Since `cgpu` typically connects to a single Colab instance (which is single-threaded for shell commands), we need a simple **Job Queue**:

*   **Sequential Execution:** Jobs run one after another.
*   **Session Persistence:** The Colab session remains active between jobs (avoiding setup overhead).
*   **Environment Caching:** `pip install` only runs once per session if possible.

## 4. Implementation Roadmap

### Step 1: Refactoring
*   Create `src/montage_ai/cgpu_jobs/` package.
*   Move `cgpu_upscaler.py` logic into `UpscaleJob`.
*   Move `transcriber.py` logic into `TranscribeJob`.

### Step 2: New Features
*   Implement `StabilizeJob`:
    *   Command: `ffmpeg -i input.mp4 -vf vidstabdetect=stepsize=32:shakiness=10:accuracy=15:result=transform.trf -f null - && ffmpeg -i input.mp4 -vf vidstabtransform=input=transform.trf:zoom=0:smoothing=10 output.mp4`
    *   Benefit: Offloads heavy CPU analysis.

### Step 3: Integration
*   Update `editor.py` to use the `JobManager` instead of direct module calls.

## 5. Future Proofing
*   **Parallelism:** If we support multiple Colab instances in the future, `JobManager` can dispatch to the least busy instance.
*   **Cost Optimization:** Check if task is small enough for local execution before offloading.
