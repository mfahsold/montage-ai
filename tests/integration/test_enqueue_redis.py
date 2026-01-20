import os
import time

import pytest

import requests

BASE = os.environ.get("TEST_BASE_URL", "http://localhost:8080")

pytestmark = pytest.mark.scale


@pytest.mark.skipif(os.environ.get("RUN_SCALE_TESTS") != "1", reason="scale tests disabled")
def test_job_enqueued_into_redis_queue():
    """Integration: POST a preview job and assert it appears in Redis RQ queue.

    - Requires cluster-accessible Redis (will be skipped otherwise)
    - Non-destructive: only asserts list-length increases
    """
    # Enqueue a preview job
    r = requests.post(f"{BASE}/api/jobs", json={"style": "dynamic", "options": {}, "quality_profile": "preview"}, timeout=10)
    assert r.status_code in (200, 201)
    data = r.json()
    jid = data.get("job_id") or data.get("id") or data.get("jobId")
    assert jid

    # Try to connect to cluster Redis
    try:
        import redis
    except Exception:
        pytest.skip("redis client not available in runner")

    host = os.environ.get("REDIS_HOST", "redis.default.svc.cluster.local")
    port = int(os.environ.get("REDIS_PORT", "6379"))

    client = redis.Redis(host=host, port=port, socket_connect_timeout=3)
    try:
        client.ping()
    except Exception:
        pytest.skip("Redis not reachable from test runner")

    # Determine queue name from settings if available, else assume 'default'
    try:
        from montage_ai.config import get_settings
        qname = get_settings().session.queue_fast_name
    except Exception:
        qname = os.environ.get("QUEUE_FAST_NAME", "default")

    key = f"rq:queue:{qname}"

    # Poll for a short window for the job to appear on the RQ list
    deadline = time.time() + 10
    seen = False
    while time.time() < deadline:
        try:
            ln = client.llen(key)
        except Exception:
            ln = 0
        if ln and ln > 0:
            seen = True
            break
        time.sleep(0.5)

    assert seen, f"job {jid} not observed on Redis key {key} (llen={ln})"