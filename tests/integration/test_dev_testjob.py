import os
import time
import requests
import pytest

pytestmark = pytest.mark.scale


@pytest.mark.skipif(os.environ.get("ENABLE_DEV_ENDPOINTS") != "true", reason="dev endpoints disabled")
@pytest.mark.skipif(os.environ.get("RUN_SCALE_TESTS") != "1", reason="scale tests disabled")
def test_dev_testjob_enqueues_and_completes():
    """Dev-only: POST to /api/internal/testjob to generate lightweight jobs and assert completion."""
    base = os.environ.get("TEST_BASE_URL", "http://localhost:8080")
    r = requests.post(f"{base}/api/internal/testjob", json={"count": 3, "duration": 2}, timeout=10)
    assert r.status_code in (200, 201)
    job_ids = r.json().get("job_ids", [])
    assert job_ids and len(job_ids) == 3

    deadline = time.time() + 60
    completed = set()
    while time.time() < deadline and len(completed) < len(job_ids):
        for jid in job_ids:
            if jid in completed:
                continue
            r = requests.get(f"{base}/api/jobs/{jid}", timeout=5)
            assert r.status_code == 200
            st = r.json().get("status")
            if st in ("finished", "completed", "done"):
                completed.add(jid)
        time.sleep(2)

    assert len(completed) == len(job_ids), f"Not all dev testjobs finished in time (finished={len(completed)})"
