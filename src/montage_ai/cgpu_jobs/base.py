"""
CGPU Job Base Class - Abstract interface for cloud GPU jobs.

All cgpu jobs follow the lifecycle:
    1. prepare_local() - Validate inputs, prepare local assets
    2. upload()        - Transfer assets to remote_work_dir
    3. run_remote()    - Execute computation on cgpu
    4. download()      - Retrieve results
    5. cleanup()       - Remove temp files (local + remote)

Example implementation:
    class MyJob(CGPUJob):
        timeout = 300  # 5 minutes

        def __init__(self, input_path: str):
            super().__init__()
            self.input_path = input_path

        def prepare_local(self) -> bool:
            return Path(self.input_path).exists()

        def get_requirements(self) -> list[str]:
            return ["numpy", "opencv-python"]

        # ... implement other abstract methods
"""

import uuid
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..logger import logger
from ..utils import file_size_mb
from ..cgpu_utils import (
    is_cgpu_available,
    run_cgpu_command,
    copy_to_remote,
    download_via_base64,
)


class JobStatus(Enum):
    """Status of a CGPU job."""
    PENDING = "pending"
    PREPARING = "preparing"
    UPLOADING = "uploading"
    RUNNING = "running"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Result of a completed CGPU job."""
    success: bool
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def __bool__(self) -> bool:
        return self.success


class CGPUJob(ABC):
    """
    Abstract base class for all CGPU jobs.

    Subclasses must implement:
        - prepare_local(): Validate inputs
        - get_requirements(): List pip packages
        - upload(): Transfer files to remote
        - run_remote(): Execute main logic
        - download(): Retrieve results

    Class attributes (override in subclasses):
        - timeout: Max execution time in seconds (default: 600)
        - max_retries: Number of retry attempts (default: 3)
        - job_type: Human-readable job type name
    """

    # Override in subclasses
    timeout: int = 600  # 10 minutes default
    max_retries: int = 3
    job_type: str = "generic"
    requires_gpu: bool = True  # Whether this job requires a GPU

    def __init__(self, job_id: Optional[str] = None):
        """
        Initialize a new CGPU job.

        Args:
            job_id: Optional custom job ID. Auto-generated if not provided.
        """
        self.job_id = job_id or uuid.uuid4().hex[:12]
        self.remote_work_dir = f"/content/cgpu_job_{self.job_id}"
        self.status = JobStatus.PENDING
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._error: Optional[str] = None
        self._installed_requirements: bool = False

    def expected_output_path(self) -> Optional[Path]:
        """Return expected local output path if known before execution."""
        output_path = getattr(self, "output_path", None)
        if isinstance(output_path, Path):
            return output_path
        if isinstance(output_path, str):
            return Path(output_path)
        return None

    def _skip_if_output_exists(self) -> Optional[JobResult]:
        """Return a success result if output already exists and reuse is enabled."""
        settings = get_settings()
        output_path = self.expected_output_path()
        if settings.processing.should_skip_output(output_path):
            self.status = JobStatus.COMPLETED
            self._end_time = time.time()
            return JobResult(
                success=True,
                output_path=str(output_path),
                metadata={"skipped": True},
                duration_seconds=self.elapsed_seconds,
            )
        return None

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time since job started."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.time()
        return end - self._start_time

    @abstractmethod
    def prepare_local(self) -> bool:
        """
        Validate inputs and prepare local assets before upload.

        Returns:
            True if preparation succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def get_requirements(self) -> List[str]:
        """
        Return list of pip packages required on the remote.

        Returns:
            List of package names (e.g., ["numpy", "opencv-python"])
        """
        return []

    @abstractmethod
    def upload(self) -> bool:
        """
        Upload local assets to remote_work_dir.

        Use copy_to_remote() for file transfers.

        Returns:
            True if upload succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def run_remote(self) -> bool:
        """
        Execute the main computation on cgpu.

        Use run_cgpu_command() for remote execution.

        Returns:
            True if execution succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def download(self) -> JobResult:
        """
        Download results from remote and return JobResult.

        Use download_via_base64() for file transfers.

        Returns:
            JobResult with success status and output_path.
        """
        pass

    def setup_remote_env(self) -> bool:
        """
        Create remote work directory and install requirements.

        Returns:
            True if setup succeeded, False otherwise.
        """
        # Create work directory
        success, _, stderr = run_cgpu_command(f"mkdir -p {self.remote_work_dir}")
        if not success:
            self._error = f"Failed to create remote directory: {stderr}"
            return False

        # Install requirements if not already done
        requirements = self.get_requirements()
        if requirements and not self._installed_requirements:
            packages = " ".join(requirements)
            success, _, stderr = run_cgpu_command(
                f"pip install -q {packages}",
                timeout=120
            )
            if not success:
                self._error = f"Failed to install requirements: {stderr}"
                return False
            self._installed_requirements = True

        return True

    def cleanup(self) -> None:
        """Remove remote work directory and local temp files."""
        try:
            run_cgpu_command(f"rm -rf {self.remote_work_dir}", timeout=30)
        except Exception:
            pass  # Best effort cleanup

    def log_upload_start(self, path: Path) -> None:
        """Log upload start message with file size."""
        size = file_size_mb(path)
        logger.info(f"Uploading {path.name} ({size:.1f} MB)...")

    def log_upload_complete(self) -> None:
        """Log upload completion."""
        logger.info("Upload complete")

    def log_download_start(self, description: str = "result") -> None:
        """Log download start message."""
        logger.info(f"Downloading {description}...")

    def log_output_size(self, path: Path) -> None:
        """Log output file size."""
        if path.exists():
            size = file_size_mb(path)
            logger.info(f"Output: {size:.1f} MB")

    def warn_large_file(self, path: Path, warn_mb: float = 500.0, very_large_mb: float = 1000.0) -> None:
        """
        Warn if file is large.

        Args:
            path: Path to check.
            warn_mb: Size in MB to trigger warning.
            very_large_mb: Size in MB to trigger strong warning.
        """
        size = file_size_mb(path)
        if size > very_large_mb:
            logger.warning(f"Very large file ({size:.1f} MB) - consider splitting into smaller segments")
        elif size > warn_mb:
            logger.warning(f"Large file ({size:.1f} MB) - processing may take a long time")

    def execute(self) -> JobResult:
        """
        Execute the full job lifecycle with error handling.

        Lifecycle:
            1. Check cgpu availability
            2. prepare_local()
            3. setup_remote_env()
            4. upload()
            5. run_remote()
            6. download()
            7. cleanup()

        Returns:
            JobResult with success status and output.
        """
        self._start_time = time.time()

        skip_result = self._skip_if_output_exists()
        if skip_result:
            return skip_result

        try:
            # Check cgpu availability
            if not is_cgpu_available(require_gpu=self.requires_gpu):
                self.status = JobStatus.FAILED
                return JobResult(
                    success=False,
                    error="cgpu not available",
                    duration_seconds=self.elapsed_seconds
                )

            # Phase 1: Prepare
            self.status = JobStatus.PREPARING
            if not self.prepare_local():
                self.status = JobStatus.FAILED
                return JobResult(
                    success=False,
                    error=self._error or "Local preparation failed",
                    duration_seconds=self.elapsed_seconds
                )

            # Phase 2: Setup remote environment
            if not self.setup_remote_env():
                self.status = JobStatus.FAILED
                return JobResult(
                    success=False,
                    error=self._error or "Remote setup failed",
                    duration_seconds=self.elapsed_seconds
                )

            # Phase 3: Upload
            self.status = JobStatus.UPLOADING
            if not self.upload():
                self.status = JobStatus.FAILED
                return JobResult(
                    success=False,
                    error=self._error or "Upload failed",
                    duration_seconds=self.elapsed_seconds
                )

            # Phase 4: Run
            self.status = JobStatus.RUNNING
            if not self.run_remote():
                self.status = JobStatus.FAILED
                return JobResult(
                    success=False,
                    error=self._error or "Remote execution failed",
                    duration_seconds=self.elapsed_seconds
                )

            # Phase 5: Download
            self.status = JobStatus.DOWNLOADING
            result = self.download()
            result.duration_seconds = self.elapsed_seconds

            if result.success:
                self.status = JobStatus.COMPLETED
            else:
                self.status = JobStatus.FAILED

            return result

        except Exception as e:
            self.status = JobStatus.FAILED
            self._error = str(e)
            return JobResult(
                success=False,
                error=f"Job execution error: {e}",
                duration_seconds=self.elapsed_seconds
            )
        finally:
            self._end_time = time.time()
            self.cleanup()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.job_id} status={self.status.value}>"
