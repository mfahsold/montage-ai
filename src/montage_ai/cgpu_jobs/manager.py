"""
CGPU Job Manager - Orchestrates job queue and execution.

Features:
    - Singleton pattern (one manager per process)
    - FIFO job queue with sequential processing
    - Session persistence (setup runs once per session)
    - Retry logic with exponential backoff
    - Comprehensive logging

Usage:
    manager = CGPUJobManager()
    manager.submit(TranscribeJob(audio_path="input.wav"))
    manager.submit(UpscaleJob(video_path="input.mp4"))
    results = manager.process_queue()
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from .base import CGPUJob, JobStatus, JobResult
from ..cgpu_utils import is_cgpu_available, run_cgpu_command, CGPU_MAX_CONCURRENCY


@dataclass
class JobEntry:
    """Internal job queue entry."""
    job: CGPUJob
    submitted_at: float = field(default_factory=time.time)
    result: Optional[JobResult] = None


class CGPUJobManager:
    """
    Singleton job manager for cgpu operations.

    Handles:
        - Job submission and queuing
        - Sequential job execution
        - Session management (one-time remote setup)
        - Retry logic with exponential backoff
    """

    _instance: Optional["CGPUJobManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "CGPUJobManager":
        """Singleton: return existing instance or create new one."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize manager (only once due to singleton)."""
        if self._initialized:
            return

        self._queue: deque[JobEntry] = deque()
        self._completed: Dict[str, JobEntry] = {}
        self._session_ready: bool = False
        self._processing: bool = False
        self._on_job_complete: Optional[Callable[[CGPUJob, JobResult], None]] = None

        self._initialized = True
        print("🔧 CGPUJobManager initialized")

    @property
    def queue_size(self) -> int:
        """Number of jobs waiting in queue."""
        return len(self._queue)

    @property
    def is_processing(self) -> bool:
        """Whether the manager is currently processing jobs."""
        return self._processing

    def set_callback(self, callback: Callable[[CGPUJob, JobResult], None]) -> None:
        """Set callback function called after each job completes."""
        self._on_job_complete = callback

    def submit(self, job: CGPUJob) -> str:
        """
        Add a job to the queue.

        Args:
            job: CGPUJob instance to queue

        Returns:
            Job ID for tracking
        """
        entry = JobEntry(job=job)
        self._queue.append(entry)
        print(f"📋 Job {job.job_id} ({job.job_type}) queued. Queue size: {len(self._queue)}")
        return job.job_id

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get status of a job by ID.

        Args:
            job_id: The job ID to look up

        Returns:
            JobStatus or None if not found
        """
        # Check queue
        for entry in self._queue:
            if entry.job.job_id == job_id:
                return entry.job.status

        # Check completed
        if job_id in self._completed:
            return self._completed[job_id].job.status

        return None

    def get_result(self, job_id: str) -> Optional[JobResult]:
        """
        Get result of a completed job.

        Args:
            job_id: The job ID to look up

        Returns:
            JobResult or None if not found/not completed
        """
        if job_id in self._completed:
            return self._completed[job_id].result
        return None

    def _ensure_session(self) -> bool:
        """
        Ensure cgpu session is ready (one-time setup).

        Performs:
            - Check cgpu availability
            - Run apt-get update (if needed)
            - Install common dependencies

        Returns:
            True if session is ready, False otherwise
        """
        if self._session_ready:
            return True

        print("🔌 Initializing cgpu session...")

        if not is_cgpu_available():
            print("❌ cgpu not available")
            return False

        # One-time system setup
        success, stdout, _ = run_cgpu_command(
            "apt-get update -qq && apt-get install -qq -y ffmpeg > /dev/null 2>&1; echo SESSION_READY",
            timeout=120
        )

        if success and "SESSION_READY" in stdout:
            self._session_ready = True
            print("✅ cgpu session ready")
            return True
        else:
            print("⚠️ Session setup incomplete, will retry per-job")
            return True  # Continue anyway, jobs may still work

    def _execute_with_retry(self, job: CGPUJob) -> JobResult:
        """
        Execute a job with retry logic.

        Uses exponential backoff: 1s, 2s, 4s, ...

        Args:
            job: The job to execute

        Returns:
            JobResult from the job
        """
        last_result: Optional[JobResult] = None

        for attempt in range(job.max_retries):
            if attempt > 0:
                wait_time = 2 ** (attempt - 1)  # 1, 2, 4, ...
                print(f"   ⏳ Retry {attempt}/{job.max_retries - 1} in {wait_time}s...")
                time.sleep(wait_time)

            try:
                result = job.execute()
                last_result = result

                if result.success:
                    return result

                # Don't retry certain errors
                if result.error and any(x in result.error.lower() for x in [
                    "file not found",
                    "invalid input",
                    "not available"
                ]):
                    return result

            except Exception as e:
                last_result = JobResult(success=False, error=str(e))

        return last_result or JobResult(success=False, error="Max retries exceeded")

    def process_queue(self) -> List[JobResult]:
        """
        Process all queued jobs sequentially.

        Returns:
            List of JobResults in order of completion
        """
        if self._processing:
            print("⚠️ Already processing queue")
            return []

        self._processing = True
        results: List[JobResult] = []

        try:
            # Ensure session is ready
            if not self._ensure_session():
                self._processing = False
                return [JobResult(success=False, error="Failed to initialize cgpu session")]

            total_jobs = len(self._queue)
            max_workers = max(1, int(CGPU_MAX_CONCURRENCY))

            if max_workers == 1 or total_jobs <= 1:
                print(f"🚀 Processing {total_jobs} job(s)...")

                job_num = 0
                while self._queue:
                    job_num += 1
                    entry = self._queue.popleft()
                    job = entry.job

                    print(f"\n[{job_num}/{total_jobs}] 🔄 {job.job_type}: {job.job_id}")

                    # Execute with retry
                    result = self._execute_with_retry(job)
                    entry.result = result
                    results.append(result)

                    # Store in completed
                    self._completed[job.job_id] = entry

                    # Status emoji
                    status_emoji = "✅" if result.success else "❌"
                    print(f"   {status_emoji} {job.job_type} completed in {result.duration_seconds:.1f}s")

                    if result.error:
                        print(f"   ⚠️ Error: {result.error}")

                    # Callback
                    if self._on_job_complete:
                        try:
                            self._on_job_complete(job, result)
                        except Exception as e:
                            print(f"   ⚠️ Callback error: {e}")
            else:
                entries = list(self._queue)
                self._queue.clear()
                print(f"🚀 Processing {total_jobs} job(s) with {max_workers} workers...")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_entry = {}
                    for entry in entries:
                        job = entry.job
                        print(f"📤 Queued {job.job_type}: {job.job_id}")
                        future = executor.submit(self._execute_with_retry, job)
                        future_to_entry[future] = entry

                    for future in as_completed(future_to_entry):
                        entry = future_to_entry[future]
                        job = entry.job
                        try:
                            result = future.result()
                        except Exception as e:
                            result = JobResult(success=False, error=str(e))

                        entry.result = result
                        results.append(result)
                        self._completed[job.job_id] = entry

                        status_emoji = "✅" if result.success else "❌"
                        print(f"   {status_emoji} {job.job_type} completed in {result.duration_seconds:.1f}s")
                        if result.error:
                            print(f"   ⚠️ Error: {result.error}")

                        if self._on_job_complete:
                            try:
                                self._on_job_complete(job, result)
                            except Exception as e:
                                print(f"   ⚠️ Callback error: {e}")

            print(f"\n📊 Queue complete: {sum(1 for r in results if r.success)}/{len(results)} succeeded")

        finally:
            self._processing = False

        return results

    def process_one(self) -> Optional[JobResult]:
        """
        Process a single job from the queue.

        Returns:
            JobResult or None if queue is empty
        """
        if not self._queue:
            return None

        if self._processing:
            print("⚠️ Already processing")
            return None

        self._processing = True

        try:
            if not self._ensure_session():
                return JobResult(success=False, error="Failed to initialize cgpu session")

            entry = self._queue.popleft()
            job = entry.job

            print(f"🔄 Processing {job.job_type}: {job.job_id}")

            result = self._execute_with_retry(job)
            entry.result = result
            self._completed[job.job_id] = entry

            if self._on_job_complete:
                try:
                    self._on_job_complete(job, result)
                except Exception:
                    pass

            return result

        finally:
            self._processing = False

    def clear_queue(self) -> int:
        """
        Clear all pending jobs from the queue.

        Returns:
            Number of jobs cleared
        """
        count = len(self._queue)
        self._queue.clear()
        print(f"🗑️ Cleared {count} job(s) from queue")
        return count

    def clear_completed(self) -> int:
        """
        Clear completed job history.

        Returns:
            Number of entries cleared
        """
        count = len(self._completed)
        self._completed.clear()
        return count

    def reset_session(self) -> None:
        """Reset session state (forces re-initialization)."""
        self._session_ready = False
        print("🔄 Session reset")

    def stats(self) -> Dict[str, int]:
        """
        Get manager statistics.

        Returns:
            Dict with queue_size, completed_count, success_count, failed_count
        """
        completed = list(self._completed.values())
        return {
            "queue_size": len(self._queue),
            "completed_count": len(completed),
            "success_count": sum(1 for e in completed if e.result and e.result.success),
            "failed_count": sum(1 for e in completed if e.result and not e.result.success),
        }


# Convenience function
def get_manager() -> CGPUJobManager:
    """Get the singleton CGPUJobManager instance."""
    return CGPUJobManager()
