"""
Tests for cgpu_jobs module - Cloud GPU Job Architecture.

Tests cover:
    - CGPUJob base class and lifecycle
    - JobStatus and JobResult dataclasses
    - CGPUJobManager singleton and queue processing
    - TranscribeJob, UpscaleJob, StabilizeJob implementations
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.montage_ai.cgpu_jobs import (
    CGPUJob,
    JobStatus,
    JobResult,
    CGPUJobManager,
)
from src.montage_ai.cgpu_jobs.base import CGPUJob as BaseCGPUJob
from src.montage_ai.cgpu_jobs.transcribe import TranscribeJob
from src.montage_ai.cgpu_jobs.upscale import UpscaleJob
from src.montage_ai.cgpu_jobs.stabilize import StabilizeJob, stabilize_video


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_successful_result(self):
        """JobResult with success=True is truthy."""
        result = JobResult(success=True, output_path="/tmp/output.mp4")
        assert result.success is True
        assert result.output_path == "/tmp/output.mp4"
        assert bool(result) is True

    def test_failed_result(self):
        """JobResult with success=False is falsy."""
        result = JobResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert bool(result) is False

    def test_result_with_metadata(self):
        """JobResult stores metadata correctly."""
        result = JobResult(
            success=True,
            output_path="/tmp/out.srt",
            metadata={"model": "medium", "language": "en"},
            duration_seconds=45.5
        )
        assert result.metadata["model"] == "medium"
        assert result.duration_seconds == 45.5


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_all_statuses_exist(self):
        """All expected statuses are defined."""
        expected = ["PENDING", "PREPARING", "UPLOADING", "RUNNING",
                    "DOWNLOADING", "COMPLETED", "FAILED", "CANCELLED"]
        for status_name in expected:
            assert hasattr(JobStatus, status_name)

    def test_status_values(self):
        """Status values are lowercase strings."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"


class TestCGPUJobBase:
    """Tests for CGPUJob abstract base class."""

    def test_job_id_generation(self):
        """Jobs get unique IDs when not provided."""
        # Create a concrete implementation for testing
        class DummyJob(BaseCGPUJob):
            def prepare_local(self): return True
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self): return JobResult(success=True)

        job1 = DummyJob()
        job2 = DummyJob()
        assert job1.job_id != job2.job_id
        assert len(job1.job_id) == 12

    def test_custom_job_id(self):
        """Jobs accept custom IDs."""
        class DummyJob(BaseCGPUJob):
            def prepare_local(self): return True
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self): return JobResult(success=True)

        job = DummyJob(job_id="custom123")
        assert job.job_id == "custom123"

    def test_initial_status(self):
        """Jobs start in PENDING status."""
        class DummyJob(BaseCGPUJob):
            def prepare_local(self): return True
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self): return JobResult(success=True)

        job = DummyJob()
        assert job.status == JobStatus.PENDING

    def test_remote_work_dir(self):
        """Remote work directory is set correctly."""
        class DummyJob(BaseCGPUJob):
            def prepare_local(self): return True
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self): return JobResult(success=True)

        job = DummyJob(job_id="test123")
        assert job.remote_work_dir == "/content/cgpu_job_test123"


class TestTranscribeJob:
    """Tests for TranscribeJob implementation."""

    def test_init_default_params(self):
        """TranscribeJob initializes with defaults."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            job = TranscribeJob(audio_path=f.name)
            assert job.model == "medium"
            assert job.output_format == "srt"
            assert job.language is None
            assert job.job_type == "transcribe"
            assert job.timeout == 600

    def test_init_custom_params(self):
        """TranscribeJob accepts custom parameters."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            job = TranscribeJob(
                audio_path=f.name,
                model="large-v3",
                output_format="vtt",
                language="de"
            )
            assert job.model == "large-v3"
            assert job.output_format == "vtt"
            assert job.language == "de"

    def test_invalid_model_fallback(self):
        """Invalid model falls back to medium."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            job = TranscribeJob(audio_path=f.name, model="invalid_model")
            assert job.model == "medium"

    def test_invalid_format_fallback(self):
        """Invalid format falls back to srt."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            job = TranscribeJob(audio_path=f.name, output_format="invalid")
            assert job.output_format == "srt"

    def test_prepare_local_file_exists(self):
        """prepare_local succeeds when file exists."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            f.flush()
            job = TranscribeJob(audio_path=f.name)
            assert job.prepare_local() is True

    def test_prepare_local_file_missing(self):
        """prepare_local fails when file doesn't exist."""
        job = TranscribeJob(audio_path="/nonexistent/audio.wav")
        assert job.prepare_local() is False

    def test_get_requirements(self):
        """TranscribeJob requires openai-whisper."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            job = TranscribeJob(audio_path=f.name)
            reqs = job.get_requirements()
            assert "openai-whisper" in reqs


class TestUpscaleJob:
    """Tests for UpscaleJob implementation."""

    def test_init_video(self):
        """UpscaleJob detects video files."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job = UpscaleJob(input_path=f.name)
            assert job.is_video is True
            assert job.job_type == "upscale"
            assert job.timeout == 1800

    def test_init_image(self):
        """UpscaleJob detects image files."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            job = UpscaleJob(input_path=f.name)
            assert job.is_video is False

    def test_scale_clamping(self):
        """Scale is clamped to 2-4."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job1 = UpscaleJob(input_path=f.name, scale=1)
            assert job1.scale == 2

            job2 = UpscaleJob(input_path=f.name, scale=10)
            assert job2.scale == 4

            job3 = UpscaleJob(input_path=f.name, scale=3)
            assert job3.scale == 3

    def test_model_validation(self):
        """Invalid model falls back to realesrgan-x4plus."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job = UpscaleJob(input_path=f.name, model="invalid")
            assert job.model == "realesrgan-x4plus"

    def test_valid_models(self):
        """Valid models are accepted."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            for model in UpscaleJob.VALID_MODELS:
                job = UpscaleJob(input_path=f.name, model=model)
                assert job.model == model

    def test_model_normalization(self):
        """Model names are normalized correctly."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            # anime keyword triggers animevideov3
            job1 = UpscaleJob(input_path=f.name, model="anime")
            assert job1.model == "realesr-animevideov3"

            # unknown model falls back to x4plus
            job2 = UpscaleJob(input_path=f.name, model="unknown")
            assert job2.model == "realesrgan-x4plus"

    def test_output_path_auto_generated(self):
        """Output path is auto-generated with _upscaled suffix."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job = UpscaleJob(input_path=f.name)
            assert "_upscaled.mp4" in str(job.output_path)

    def test_denoise_strength_clamping(self):
        """Denoise strength is clamped to 0-1."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job1 = UpscaleJob(input_path=f.name, denoise_strength=-0.5)
            assert job1.denoise_strength == 0.0

            job2 = UpscaleJob(input_path=f.name, denoise_strength=1.5)
            assert job2.denoise_strength == 1.0


class TestStabilizeJob:
    """Tests for StabilizeJob implementation."""

    def test_init_defaults(self):
        """StabilizeJob initializes with defaults."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job = StabilizeJob(video_path=f.name)
            assert job.smoothing == 10
            assert job.shakiness == 5
            assert job.accuracy == 15
            assert job.stepsize == 6
            assert job.zoom == 0.0
            assert job.optzoom == 1
            assert job.crop == "black"
            assert job.job_type == "stabilize"
            assert job.timeout == 900

    def test_parameter_clamping(self):
        """Parameters are clamped to valid ranges."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job = StabilizeJob(
                video_path=f.name,
                smoothing=100,  # max 30
                shakiness=20,   # max 10
                accuracy=50,    # max 15
                stepsize=100,   # max 32
                zoom=200.0,     # max 100
                optzoom=5,      # max 2
            )
            assert job.smoothing == 30
            assert job.shakiness == 10
            assert job.accuracy == 15
            assert job.stepsize == 32
            assert job.zoom == 100.0
            assert job.optzoom == 2

    def test_crop_validation(self):
        """Invalid crop values fall back to black."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job = StabilizeJob(video_path=f.name, crop="invalid")
            assert job.crop == "black"

    def test_valid_crop_values(self):
        """Valid crop values are accepted."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job1 = StabilizeJob(video_path=f.name, crop="black")
            assert job1.crop == "black"

            job2 = StabilizeJob(video_path=f.name, crop="keep")
            assert job2.crop == "keep"

    def test_output_path_auto_generated(self):
        """Output path is auto-generated with _stabilized suffix."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job = StabilizeJob(video_path=f.name)
            assert "_stabilized.mp4" in str(job.output_path)

    def test_prepare_local_valid_video(self):
        """prepare_local succeeds for valid video files."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            f.flush()
            job = StabilizeJob(video_path=f.name)
            assert job.prepare_local() is True

    def test_prepare_local_invalid_extension(self):
        """prepare_local fails for invalid file extensions."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a video")
            f.flush()
            job = StabilizeJob(video_path=f.name)
            assert job.prepare_local() is False

    def test_get_requirements(self):
        """StabilizeJob has no pip requirements (uses system ffmpeg)."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            job = StabilizeJob(video_path=f.name)
            assert job.get_requirements() == []


class TestCGPUJobManager:
    """Tests for CGPUJobManager singleton."""

    def setup_method(self):
        """Reset singleton for each test."""
        CGPUJobManager._instance = None

    def test_singleton_pattern(self):
        """CGPUJobManager is a singleton."""
        manager1 = CGPUJobManager()
        manager2 = CGPUJobManager()
        assert manager1 is manager2

    def test_submit_job(self):
        """Jobs can be submitted to queue."""
        manager = CGPUJobManager()

        class DummyJob(BaseCGPUJob):
            job_type = "dummy"
            def prepare_local(self): return True
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self): return JobResult(success=True)

        job = DummyJob()
        job_id = manager.submit(job)

        assert manager.queue_size == 1
        assert job_id == job.job_id

    def test_get_job_status(self):
        """Job status can be queried."""
        manager = CGPUJobManager()

        class DummyJob(BaseCGPUJob):
            job_type = "dummy"
            def prepare_local(self): return True
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self): return JobResult(success=True)

        job = DummyJob()
        manager.submit(job)

        status = manager.get_job_status(job.job_id)
        assert status == JobStatus.PENDING

    def test_clear_queue(self):
        """Queue can be cleared."""
        manager = CGPUJobManager()

        class DummyJob(BaseCGPUJob):
            job_type = "dummy"
            def prepare_local(self): return True
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self): return JobResult(success=True)

        manager.submit(DummyJob())
        manager.submit(DummyJob())
        assert manager.queue_size == 2

        cleared = manager.clear_queue()
        assert cleared == 2
        assert manager.queue_size == 0

    def test_stats(self):
        """Manager provides statistics."""
        manager = CGPUJobManager()
        stats = manager.stats()

        assert "queue_size" in stats
        assert "completed_count" in stats
        assert "success_count" in stats
        assert "failed_count" in stats

    @patch('src.montage_ai.cgpu_jobs.base.run_cgpu_command')
    @patch('src.montage_ai.cgpu_jobs.base.is_cgpu_available')
    @patch('src.montage_ai.cgpu_jobs.manager.run_cgpu_command')
    @patch('src.montage_ai.cgpu_jobs.manager.is_cgpu_available')
    def test_process_queue_execution(self, mock_mgr_avail, mock_mgr_run, mock_base_avail, mock_base_run):
        """Manager processes jobs in queue."""
        mock_mgr_avail.return_value = True
        mock_base_avail.return_value = True
        mock_mgr_run.return_value = (True, "SESSION_READY", "")
        mock_base_run.return_value = (True, "", "")

        manager = CGPUJobManager()
        manager.clear_queue()

        # Create a mock job that tracks execution
        class MockJob(CGPUJob):
            job_type = "mock"
            def __init__(self):
                super().__init__()
                self.executed = False

            def prepare_local(self): return True
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self):
                self.executed = True
                return JobResult(success=True)

        job = MockJob()
        manager.submit(job)

        # Process the queue
        results = manager.process_queue()

        assert len(results) == 1
        assert results[0].success is True
        assert job.executed is True
        assert manager.queue_size == 0

    def test_callback_registration(self):
        """Callbacks can be registered."""
        manager = CGPUJobManager()
        callback_called = []

        def my_callback(job, result):
            callback_called.append((job, result))

        manager.set_callback(my_callback)
        # Callback will be invoked during process_queue


class TestJobExecution:
    """Integration tests for job execution lifecycle."""

    @patch('src.montage_ai.cgpu_jobs.base.is_cgpu_available')
    def test_execute_cgpu_not_available(self, mock_available):
        """Jobs fail gracefully when cgpu is unavailable."""
        mock_available.return_value = False

        class DummyJob(BaseCGPUJob):
            def prepare_local(self): return True
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self): return JobResult(success=True)

        job = DummyJob()
        result = job.execute()

        assert result.success is False
        assert "not available" in result.error

    @patch('src.montage_ai.cgpu_jobs.base.is_cgpu_available')
    @patch('src.montage_ai.cgpu_jobs.base.run_cgpu_command')
    def test_execute_prepare_fails(self, mock_run, mock_available):
        """Jobs handle prepare_local failure."""
        mock_available.return_value = True

        class FailingPrepareJob(BaseCGPUJob):
            def prepare_local(self):
                self._error = "Preparation failed"
                return False
            def get_requirements(self): return []
            def upload(self): return True
            def run_remote(self): return True
            def download(self): return JobResult(success=True)

        job = FailingPrepareJob()
        result = job.execute()

        assert result.success is False
        assert job.status == JobStatus.FAILED

    @patch('src.montage_ai.cgpu_jobs.base.is_cgpu_available')
    @patch('src.montage_ai.cgpu_jobs.base.run_cgpu_command')
    def test_execute_full_lifecycle(self, mock_run, mock_available):
        """Successful job goes through full lifecycle."""
        mock_available.return_value = True
        mock_run.return_value = (True, "OK", "")

        lifecycle_steps = []

        class TrackingJob(BaseCGPUJob):
            def prepare_local(self):
                lifecycle_steps.append("prepare")
                return True
            def get_requirements(self):
                return []
            def upload(self):
                lifecycle_steps.append("upload")
                return True
            def run_remote(self):
                lifecycle_steps.append("run")
                return True
            def download(self):
                lifecycle_steps.append("download")
                return JobResult(success=True, output_path="/tmp/out")

        job = TrackingJob()
        result = job.execute()

        assert result.success is True
        assert lifecycle_steps == ["prepare", "upload", "run", "download"]
        assert job.status == JobStatus.COMPLETED


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @patch('src.montage_ai.cgpu_jobs.stabilize.StabilizeJob.execute')
    def test_stabilize_video_success(self, mock_execute):
        """stabilize_video returns path on success."""
        mock_execute.return_value = JobResult(
            success=True,
            output_path="/tmp/stabilized.mp4"
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"video data")
            f.flush()
            result = stabilize_video(f.name)
            assert result == "/tmp/stabilized.mp4"

    @patch('src.montage_ai.cgpu_jobs.stabilize.StabilizeJob.execute')
    def test_stabilize_video_failure(self, mock_execute):
        """stabilize_video returns None on failure."""
        mock_execute.return_value = JobResult(success=False, error="Failed")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"video data")
            f.flush()
            result = stabilize_video(f.name)
            assert result is None


class TestModuleImports:
    """Tests for module-level imports and exports."""

    def test_lazy_imports(self):
        """Job classes can be imported lazily."""
        from src.montage_ai.cgpu_jobs import TranscribeJob
        from src.montage_ai.cgpu_jobs import UpscaleJob
        from src.montage_ai.cgpu_jobs import StabilizeJob

        assert TranscribeJob is not None
        assert UpscaleJob is not None
        assert StabilizeJob is not None

    def test_all_exports(self):
        """__all__ contains expected exports."""
        import src.montage_ai.cgpu_jobs as cgpu_jobs

        expected = ["CGPUJob", "JobStatus", "JobResult", "CGPUJobManager",
                    "TranscribeJob", "UpscaleJob", "StabilizeJob"]
        for name in expected:
            assert name in cgpu_jobs.__all__
