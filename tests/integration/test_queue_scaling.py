import os
import time
import requests
import pytest

BASE = os.environ.get("TEST_BASE_URL", "http://localhost:8080")

pytestmark = pytest.mark.scale


@pytest.mark.skipif(os.environ.get("RUN_SCALE_TESTS") != "1", reason="scale tests disabled")
def test_preview_jobs_scale_workers_and_complete():
    """Smoke: submit 3 preview jobs and assert they complete within timeout.

    - Skipped by default (RUN_SCALE_TESTS=1 to enable)
    - Non‑destructive: uses preview quality (fast path)
    """
    job_ids = []
    for _ in range(3):
        r = requests.post(f"{BASE}/api/jobs", json={"style": "dynamic", "options": {}, "quality_profile": "preview"}, timeout=10)
        assert r.status_code in (200, 201)
        data = r.json()
        job_ids.append(data.get("job_id") or data.get("id") or data.get("jobId"))

    assert job_ids, "no jobs created"

    deadline = time.time() + 180
    completed = set()
    while time.time() < deadline and len(completed) < len(job_ids):
        for jid in job_ids:
            if jid in completed:
                continue
            r = requests.get(f"{BASE}/api/jobs/{jid}", timeout=5)
            if r.status_code == 200:
                st = r.json().get("status")
                if st in ("finished", "completed", "done"):
                    completed.add(jid)
        time.sleep(3)

    assert len(completed) == len(job_ids), f"Not all jobs finished in time (finished={len(completed)})"
