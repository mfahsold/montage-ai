import os
import json

from montage_ai import telemetry


def test_get_prometheus_metrics_counts(tmp_path, monkeypatch):
    # Point telemetry collector to a temp dir
    # Create a fresh TelemetryCollector pointed at tmp_path and patch the global getter
    coll = telemetry.TelemetryCollector(storage_path=tmp_path)
    # Persist a small aggregate so the collector picks it up
    coll._aggregate.total_jobs = 3
    coll._aggregate.successful_jobs = 2
    coll._aggregate.failed_jobs = 1
    coll._aggregate.avg_time_to_preview = 1.23
    # Write events.jsonl with a couple of known events
    ev_lines = [
        json.dumps({"type": "proxy_cache_hit", "data": {}}),
        json.dumps({"type": "proxy_cache_miss", "data": {}}),
        json.dumps({"type": "scene_detection_timeout", "data": {}}),
    ]
    with open(tmp_path / "events.jsonl", "w") as f:
        f.write("\n".join(ev_lines) + "\n")

    # Install our test collector as the global collector for the duration of the test
    orig = getattr(telemetry, '_collector', None)
    telemetry._collector = coll

    try:
        out = telemetry.get_prometheus_metrics()
    finally:
        telemetry._collector = orig
    assert "montage_jobs_total 3" in out
    assert "montage_time_to_preview_seconds_avg 1.230" in out
    assert "montage_proxy_cache_hit_total 1" in out
    assert "montage_proxy_cache_miss_total 1" in out
    assert "montage_scene_detection_timeout_total 1" in out
    assert "montage_proxy_cache_hit_rate" in out
