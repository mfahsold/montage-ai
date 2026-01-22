import os
import time
import requests
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv('RUN_DEV_E2E', 'false').lower() != 'true', reason='Dev e2e tests are opt-in')
def test_preview_smoke_and_metrics(dev_base_url):
    """Opt-in smoke: enqueue 3 preview requests and assert metrics appear."""
    base = dev_base_url.rstrip('/')
    runs = 3
    job_ids = []
    for _ in range(runs):
        resp = requests.post(f"{base}/api/jobs", json={"video_files": ["benchmark_clip_a.mp4"], "quality_profile": "preview"}, timeout=10)
        assert resp.status_code in (200, 201)
        job = resp.json()
        job_ids.append(job['id'])
        time.sleep(0.5)

    # wait for jobs to complete (best-effort)
    finished = 0
    deadline = time.time() + 60
    while time.time() < deadline and finished < runs:
        finished = 0
        for jid in job_ids:
            st = requests.get(f"{base}/api/jobs/{jid}", timeout=5).json().get('status')
            if st in ('finished', 'completed'):
                finished += 1
        time.sleep(1)

    assert finished >= 1, 'At least one preview should complete in smoke'

    # Check metrics
    metrics = requests.get(f"{base}/metrics", timeout=5).text
    assert 'montage_time_to_preview_seconds_count' in metrics
    assert 'montage_proxy_cache_hit_total' in metrics
    assert 'montage_proxy_cache_request_total' in metrics
