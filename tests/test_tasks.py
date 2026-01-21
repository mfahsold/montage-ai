import time

import pytest

from montage_ai import tasks


def test_run_test_job_sleeps_and_updates_jobstore(monkeypatch):
    calls = {"sleep": 0, "updates": []}

    # Stub out sleep so test is fast
    def fake_sleep(s):
        calls["sleep"] += 1
        assert s <= 1

    monkeypatch.setattr(tasks, "time", type("T", (), {"sleep": fake_sleep}))

    class DummyStore:
        def update_job(self, job_id, payload):
            calls["updates"].append((job_id, payload))
        # Backwards-compatible shim for newer JobStore API used by tasks
        def update_job_with_retry(self, job_id, updates, retries=1, backoff_base=0.01):
            return self.update_job(job_id, updates) or True

    monkeypatch.setattr(tasks, "JobStore", lambda: DummyStore())
    monkeypatch.setattr(tasks.telemetry, "record_event", lambda *a, **k: None)

    tasks.run_test_job("devtest-xyz", duration=2)

    assert calls["sleep"] >= 1
    assert any(u[1]["status"] == "started" for u in calls["updates"])
    assert any(u[1]["status"] == "finished" for u in calls["updates"])
