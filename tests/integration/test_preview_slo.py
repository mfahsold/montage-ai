import os
import time
import requests
import pytest

BASE = os.environ.get("TEST_BASE_URL", "http://localhost:8080")

pytestmark = pytest.mark.scale


@pytest.mark.skipif(os.environ.get("RUN_SCALE_TESTS") != "1", reason="scale tests disabled")
def test_preview_slo_end_to_end():
    """On-cluster smoke: small preview should complete within SLOs.

    - Requires cluster-accessible web + worker + PVC with small clips present
    - Non-destructive and skipped by default
    """
    slo_p50 = float(os.environ.get("SLO_P50", "8"))
    slo_p95 = float(os.environ.get("SLO_P95", "30"))

    # Create job pointing at the small test clips (these should exist on the test PVC)
    r = requests.post(f"{BASE}/api/jobs", json={"style": "dynamic", "options": {"video_files": ["benchmark_clip_a.mp4", "benchmark_clip_b.mp4"]}, "quality_profile": "preview"}, timeout=10)
    assert r.status_code in (200, 201)
    jid = r.json().get("id") or r.json().get("job_id")
    assert jid

    t0 = time.time()
    deadline = t0 + 90
    finished = False
    while time.time() < deadline:
        r = requests.get(f"{BASE}/api/jobs/{jid}", timeout=5)
        assert r.status_code == 200
        st = r.json().get("status")
        if st in ("finished", "completed", "done"):
            finished = True
            break
        time.sleep(1)

    assert finished, "preview job did not finish in 90s"
    elapsed = time.time() - t0

    # Single-run assertion — CI guard will enforce SLOs over several runs
    assert elapsed <= slo_p95, f"preview exceeded p95 SLO: {elapsed:.1f}s > {slo_p95}s"