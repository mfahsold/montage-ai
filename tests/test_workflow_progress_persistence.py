import pytest
from unittest.mock import Mock

from montage_ai.core.workflow import VideoWorkflow, WorkflowOptions, WorkflowPhase


class DummyWorkflow(VideoWorkflow):
    @property
    def workflow_name(self) -> str:
        return "dummy"

    @property
    def workflow_type(self) -> str:
        return "dummy"

    def initialize(self) -> None:
        pass

    def validate(self) -> None:
        pass

    def analyze(self):
        return {}

    def process(self, analysis_result):
        return {}

    def render(self, processing_result):
        return {}

    def export(self, render_result) -> str:
        return "/tmp/out.mp4"


def make_workflow_with_mock_store():
    opts = WorkflowOptions(input_path="/tmp/in.mp4", output_dir="/tmp", job_id="job-123")
    wf = DummyWorkflow(opts)
    mock_store = Mock()
    mock_store.update_job = Mock(return_value=True)
    mock_store.update_job_with_retry = Mock(return_value=True)
    wf.job_store = mock_store
    return wf, mock_store


def test_update_progress_uses_best_effort_for_small_percent():
    wf, store = make_workflow_with_mock_store()

    wf._update_progress(percent=30, message="processing clip 1")

    store.update_job.assert_called_once()
    store.update_job_with_retry.assert_not_called()


def test_update_progress_uses_retry_for_high_percent():
    wf, store = make_workflow_with_mock_store()

    wf._update_progress(percent=95, message="finalizing")

    store.update_job_with_retry.assert_called_once()
    # verify the call contained the expected primary args and used the small retry budget
    called_args, called_kwargs = store.update_job_with_retry.call_args
    assert called_args[0] == wf.options.job_id
    assert called_args[1] == {"progress_percent": 95, "message": "finalizing"}
    assert called_kwargs.get("retries", None) == 2
    # Ensure best-effort path not used for high-value update
    store.update_job.assert_not_called()
