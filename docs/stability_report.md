# Stability & Performance Analysis: Memory & cgpu

**Date:** December 28, 2025
**Status:** Hardened

## 1. Physical RAM Protection

### Mechanism
The system employs a multi-layered approach to prevent Out-Of-Memory (OOM) crashes:

1.  **Adaptive Memory Manager (`src/montage_ai/memory_monitor.py`):**
    -   Continuously monitors RSS (Resident Set Size) and VMS (Virtual Memory Size).
    -   Triggers proactive garbage collection (`gc.collect()`) when usage exceeds warning thresholds (default 75%).
    -   Used by `editor.py` to dynamically adjust batch sizes during processing.

2.  **Job Admission Control (`src/montage_ai/web_ui/app.py`):**
    -   **`MIN_MEMORY_GB` Gatekeeper:** Before starting any new job, the system checks if at least 2GB of RAM is available. If not, the job is rejected/queued.
    -   **Concurrency Limit:** `MAX_CONCURRENT_JOBS` (default: 2) strictly limits the number of parallel processing threads.

### Assessment
The protection is **robust** for single-node deployments. The combination of admission control and runtime monitoring effectively prevents the "thundering herd" problem where multiple jobs exhaust RAM.

## 2. cgpu Stability & Heavy Load

### Mechanism
The `cgpu` integration offloads heavy upscaling tasks to Google Colab GPUs.

1.  **Retry Logic (`src/montage_ai/cgpu_utils.py`):**
    -   Implements automatic retries for transient network failures.
    -   Detects "session expired" errors and attempts to reconnect/re-authenticate automatically.

2.  **Fallback Strategy (`src/montage_ai/editor.py`):**
    -   If cgpu upscaling fails (returns `None`), the system gracefully falls back to local processing (Real-ESRGAN via Vulkan or FFmpeg CPU).

3.  **Race Condition Fix (v2.2.1):**
    -   **Issue:** Previously, all jobs used a shared remote directory (`/content/upscale_work/input.mp4`). Concurrent jobs would overwrite each other's files.
    -   **Fix:** Implemented unique job directories using UUIDs (`/content/upscale_work/{uuid}/`).
    -   **Result:** Multiple concurrent cgpu jobs can now run safely without collision (limited only by Colab's ability to handle parallel execution, which is usually single-threaded per session, but this prevents data corruption).

### Assessment
With the race condition fix, the cgpu integration is **stable** for heavy loads. The fallback mechanism ensures that even if the cloud connection is flaky, the montage generation will complete (albeit slower).

## Recommendations

1.  **Monitor Colab Limits:** Google Colab has usage limits. Heavy continuous usage might trigger timeouts or bans. The fallback logic handles this, but users should be aware.
2.  **Adjust Concurrency:** For systems with <16GB RAM, keep `MAX_CONCURRENT_JOBS=1`.
