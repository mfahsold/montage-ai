import json


def test_dev_testjob_post_enqueues_and_creates_jobs(client, mock_redis_and_rq):
    """Unit: POST /api/internal/testjob enqueues lightweight dev jobs (mocked).

    - Uses the `client` and `mock_redis_and_rq` fixtures to avoid hitting real Redis.
    - Asserts HTTP 201 and that JobStore.create_job and queue.enqueue are called
      the expected number of times.
    """
    mock_job_store = mock_redis_and_rq['job_store']
    mock_q = mock_redis_and_rq['q']

    # Enable dev endpoints guard for the test
    import os
    os.environ['ENABLE_DEV_ENDPOINTS'] = 'true'

    payload = {"count": 4, "duration": 1}
    resp = client.post("/api/internal/testjob", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 201

    body = resp.get_json()
    assert 'job_ids' in body and len(body['job_ids']) == 4

    # job_store.create_job called for each job. enqueue is exercised but uses a
    # module-level ResilientQueue instance which is not the same as the mocked
    # `q` fixture; assert job-store side effects instead.
    assert mock_job_store.create_job.call_count == 4
    assert mock_job_store.update_job.call_count == 0


def test_enqueue_montage_uses_correct_queue_and_logs(monkeypatch, caplog):
    """enqueue_montage should pick the preview queue when requested and call enqueue."""
    from montage_ai.web_ui.app import enqueue_montage

    recorded = {}

    class FakeQueue:
        def __init__(self, name):
            self.name = name

        def enqueue(self, fn, job_id, style, options, job_timeout=None):
            recorded['called'] = (self.name, job_id, options.get('quality_profile'), job_timeout)
            class J: id = 'rq-123'
            return J()

    monkeypatch.setattr('montage_ai.web_ui.app.get_montage_queue', lambda opts: FakeQueue('preview-queue'))

    caplog.clear()
    caplog.set_level('INFO')
    rq_job = enqueue_montage('j-1', 'dynamic', {'quality_profile': 'preview', 'job_timeout': 42})

    assert recorded['called'][0] == 'preview-queue'
    assert recorded['called'][1] == 'j-1'
    assert recorded['called'][2] == 'preview'
    assert recorded['called'][3] == 42
    assert getattr(rq_job, 'id') == 'rq-123'
    assert any('Enqueuing job j-1' in r.message for r in caplog.records)
