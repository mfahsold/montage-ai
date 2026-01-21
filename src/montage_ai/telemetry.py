"""
Telemetry Module for Montage AI

Collects and stores operational metrics for monitoring and optimization.
Uses local JSON storage (KISS principle - no external dependencies).

Metrics collected:
- Time-to-Preview: How long until first preview is available
- Export Success Rate: Percentage of successful exports
- Render Time: Total render duration
- Job Durations: Per-phase timing breakdowns

Usage:
    from montage_ai.telemetry import Telemetry

    with Telemetry.job_context("job_123", "montage") as t:
        t.phase_start("analyzing")
        # ... do analysis ...
        t.phase_end("analyzing")

        t.phase_start("rendering")
        # ... do rendering ...
        t.phase_end("rendering")

        t.record_success()  # or t.record_failure("error message")
"""

import json
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from .logger import logger
from .config import get_settings


@dataclass
class JobMetrics:
    """Metrics for a single job execution."""
    job_id: str
    job_type: str  # "montage", "shorts", "transcript"
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None

    # Phase timings (phase_name -> duration_seconds)
    phase_durations: Dict[str, float] = field(default_factory=dict)

    # Key metrics
    time_to_preview: Optional[float] = None  # Seconds until first preview
    render_duration: Optional[float] = None  # Total render time
    export_duration: Optional[float] = None  # Export phase time

    # Job details
    clip_count: int = 0
    output_duration: Optional[float] = None  # Output video duration
    output_size_mb: Optional[float] = None  # Output file size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": self.end_time - self.start_time if self.end_time else None,
            "success": self.success,
            "error_message": self.error_message,
            "phase_durations": self.phase_durations,
            "time_to_preview": self.time_to_preview,
            "render_duration": self.render_duration,
            "export_duration": self.export_duration,
            "clip_count": self.clip_count,
            "output_duration": self.output_duration,
            "output_size_mb": self.output_size_mb,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all jobs."""
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0

    # Averages
    avg_time_to_preview: Optional[float] = None
    avg_render_duration: Optional[float] = None
    avg_export_duration: Optional[float] = None

    # By job type
    jobs_by_type: Dict[str, int] = field(default_factory=dict)
    success_by_type: Dict[str, int] = field(default_factory=dict)

    # Recent errors
    recent_errors: List[Dict[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_jobs == 0:
            return 0.0
        return self.successful_jobs / self.total_jobs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_jobs": self.total_jobs,
            "successful_jobs": self.successful_jobs,
            "failed_jobs": self.failed_jobs,
            "success_rate": round(self.success_rate * 100, 1),
            "avg_time_to_preview_seconds": round(self.avg_time_to_preview, 2) if self.avg_time_to_preview else None,
            "avg_render_duration_seconds": round(self.avg_render_duration, 2) if self.avg_render_duration else None,
            "avg_export_duration_seconds": round(self.avg_export_duration, 2) if self.avg_export_duration else None,
            "jobs_by_type": self.jobs_by_type,
            "success_by_type": self.success_by_type,
            "recent_errors": self.recent_errors[-10:],  # Last 10 errors
        }


class TelemetryCollector:
    """
    Collects and persists telemetry data.

    Thread-safe and writes to JSON files in the output directory.
    """

    MAX_HISTORY = 1000  # Max jobs to keep in history

    def __init__(self, storage_path: Optional[Path] = None):
        # Use /tmp for telemetry to avoid permission issues with /data/output PVC
        if storage_path:
            default_storage = Path(storage_path)
        else:
            telemetry_base = Path("/tmp/montage_telemetry")
            default_storage = telemetry_base
        self.storage_path = default_storage
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._current_jobs: Dict[str, JobMetrics] = {}
        self._phase_starts: Dict[str, Dict[str, float]] = {}  # job_id -> {phase -> start_time}

        # Load existing aggregate metrics
        self._aggregate = self._load_aggregate()
        self._history: List[JobMetrics] = []

    def _load_aggregate(self) -> AggregateMetrics:
        """Load aggregate metrics from disk."""
        agg_file = self.storage_path / "aggregate.json"
        if agg_file.exists():
            try:
                with open(agg_file) as f:
                    data = json.load(f)
                return AggregateMetrics(
                    total_jobs=data.get("total_jobs", 0),
                    successful_jobs=data.get("successful_jobs", 0),
                    failed_jobs=data.get("failed_jobs", 0),
                    avg_time_to_preview=data.get("avg_time_to_preview_seconds"),
                    avg_render_duration=data.get("avg_render_duration_seconds"),
                    avg_export_duration=data.get("avg_export_duration_seconds"),
                    jobs_by_type=data.get("jobs_by_type", {}),
                    success_by_type=data.get("success_by_type", {}),
                    recent_errors=data.get("recent_errors", []),
                )
            except Exception as e:
                logger.warning(f"Failed to load telemetry aggregate: {e}")
        return AggregateMetrics()

    def _save_aggregate(self) -> None:
        """Save aggregate metrics to disk."""
        agg_file = self.storage_path / "aggregate.json"
        try:
            with open(agg_file, "w") as f:
                json.dump(self._aggregate.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save telemetry aggregate: {e}")

    def _save_job(self, metrics: JobMetrics) -> None:
        """Save individual job metrics to disk."""
        job_file = self.storage_path / f"job_{metrics.job_id}.json"
        try:
            with open(job_file, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save job telemetry: {e}")

    def start_job(self, job_id: str, job_type: str) -> JobMetrics:
        """Start tracking a new job."""
        with self._lock:
            metrics = JobMetrics(
                job_id=job_id,
                job_type=job_type,
                start_time=time.time()
            )
            self._current_jobs[job_id] = metrics
            self._phase_starts[job_id] = {}
            logger.debug(f"Telemetry: Started tracking job {job_id}")
            return metrics

    def phase_start(self, job_id: str, phase: str) -> None:
        """Record start of a phase."""
        with self._lock:
            if job_id in self._phase_starts:
                self._phase_starts[job_id][phase] = time.time()

    def phase_end(self, job_id: str, phase: str) -> Optional[float]:
        """Record end of a phase, return duration."""
        with self._lock:
            if job_id not in self._phase_starts:
                return None

            start = self._phase_starts[job_id].pop(phase, None)
            if start is None:
                return None

            duration = time.time() - start

            if job_id in self._current_jobs:
                self._current_jobs[job_id].phase_durations[phase] = duration

                # Track specific phases
                if phase == "preview":
                    self._current_jobs[job_id].time_to_preview = duration
                elif phase == "rendering":
                    self._current_jobs[job_id].render_duration = duration
                elif phase == "exporting":
                    self._current_jobs[job_id].export_duration = duration

            return duration

    def update_job(self, job_id: str, **kwargs) -> None:
        """Update job metrics."""
        with self._lock:
            if job_id in self._current_jobs:
                metrics = self._current_jobs[job_id]
                for key, value in kwargs.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)

    def end_job(self, job_id: str, success: bool, error_message: Optional[str] = None) -> Optional[JobMetrics]:
        """End job tracking and persist metrics."""
        with self._lock:
            if job_id not in self._current_jobs:
                return None

            metrics = self._current_jobs.pop(job_id)
            self._phase_starts.pop(job_id, None)

            metrics.end_time = time.time()
            metrics.success = success
            metrics.error_message = error_message

            # Update aggregate
            self._aggregate.total_jobs += 1
            if success:
                self._aggregate.successful_jobs += 1
            else:
                self._aggregate.failed_jobs += 1
                if error_message:
                    self._aggregate.recent_errors.append({
                        "job_id": job_id,
                        "time": datetime.now().isoformat(),
                        "error": error_message[:200]  # Truncate long errors
                    })

            # Update by type
            jt = metrics.job_type
            self._aggregate.jobs_by_type[jt] = self._aggregate.jobs_by_type.get(jt, 0) + 1
            if success:
                self._aggregate.success_by_type[jt] = self._aggregate.success_by_type.get(jt, 0) + 1

            # Update averages (running average)
            n = self._aggregate.successful_jobs
            if n > 0 and success:
                if metrics.time_to_preview:
                    old = self._aggregate.avg_time_to_preview or 0
                    self._aggregate.avg_time_to_preview = old + (metrics.time_to_preview - old) / n

                if metrics.render_duration:
                    old = self._aggregate.avg_render_duration or 0
                    self._aggregate.avg_render_duration = old + (metrics.render_duration - old) / n

                if metrics.export_duration:
                    old = self._aggregate.avg_export_duration or 0
                    self._aggregate.avg_export_duration = old + (metrics.export_duration - old) / n

            # Persist
            self._save_job(metrics)
            self._save_aggregate()

            logger.info(f"Telemetry: Job {job_id} completed (success={success}, duration={metrics.end_time - metrics.start_time:.1f}s)")

            return metrics

    def get_aggregate(self) -> AggregateMetrics:
        """Get current aggregate metrics."""
        with self._lock:
            return self._aggregate

    def get_job_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific job."""
        with self._lock:
            if job_id in self._current_jobs:
                return self._current_jobs[job_id].to_dict()

        # Try loading from disk
        job_file = self.storage_path / f"job_{job_id}.json"
        if job_file.exists():
            try:
                with open(job_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return None


class TelemetryContext:
    """Context manager for tracking a single job."""

    def __init__(self, collector: TelemetryCollector, job_id: str, job_type: str):
        self._collector = collector
        self._job_id = job_id
        self._job_type = job_type
        self._success = False
        self._error: Optional[str] = None

    def phase_start(self, phase: str) -> None:
        """Start tracking a phase."""
        self._collector.phase_start(self._job_id, phase)

    def phase_end(self, phase: str) -> Optional[float]:
        """End tracking a phase, return duration."""
        return self._collector.phase_end(self._job_id, phase)

    def update(self, **kwargs) -> None:
        """Update job metrics."""
        self._collector.update_job(self._job_id, **kwargs)

    def record_success(self) -> None:
        """Mark job as successful."""
        self._success = True

    def record_failure(self, error: str) -> None:
        """Mark job as failed with error message."""
        self._success = False
        self._error = error

    def __enter__(self) -> "TelemetryContext":
        self._collector.start_job(self._job_id, self._job_type)
        return self

    def __exit__(self, exc_type, exc_val, _exc_tb) -> None:
        if exc_type is not None:
            self._error = str(exc_val) if exc_val else str(exc_type)
            self._success = False
        self._collector.end_job(self._job_id, self._success, self._error)


# Global collector instance
_collector: Optional[TelemetryCollector] = None


def get_collector() -> TelemetryCollector:
    """Get or create the global telemetry collector."""
    global _collector
    if _collector is None:
        _collector = TelemetryCollector()
    return _collector


@contextmanager
def job_context(job_id: str, job_type: str):
    """Context manager for tracking a job."""
    ctx = TelemetryContext(get_collector(), job_id, job_type)
    with ctx:
        yield ctx


def record_event(event_type: str, data: Dict[str, Any]) -> None:
    """Record a standalone event (not tied to a job)."""
    collector = get_collector()
    event_file = collector.storage_path / "events.jsonl"
    try:
        with open(event_file, "a") as f:
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "data": data
            }
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning(f"Failed to record event: {e}")


# Convenience functions
def start_job(job_id: str, job_type: str) -> JobMetrics:
    """Start tracking a job."""
    return get_collector().start_job(job_id, job_type)


def phase_start(job_id: str, phase: str) -> None:
    """Start tracking a phase."""
    get_collector().phase_start(job_id, phase)


def phase_end(job_id: str, phase: str) -> Optional[float]:
    """End tracking a phase."""
    return get_collector().phase_end(job_id, phase)


def end_job(job_id: str, success: bool, error: Optional[str] = None) -> Optional[JobMetrics]:
    """End tracking a job."""
    return get_collector().end_job(job_id, success, error)


def get_aggregate() -> AggregateMetrics:
    """Get aggregate metrics."""
    return get_collector().get_aggregate()


def get_kpis() -> Dict[str, Any]:
    """Get KPIs in a format suitable for display."""
    agg = get_aggregate()
    from .config import get_settings
    _settings = get_settings()
    preview_target = int(_settings.monitoring.preview_target_seconds)
    return {
        "success_rate_percent": round(agg.success_rate * 100, 1),
        "total_jobs": agg.total_jobs,
        "avg_time_to_preview_seconds": round(agg.avg_time_to_preview, 1) if agg.avg_time_to_preview else None,
        "avg_render_duration_seconds": round(agg.avg_render_duration, 1) if agg.avg_render_duration else None,
        "kpi_targets": {
            "time_to_preview_target": preview_target,  # Configurable target (seconds)
            "success_rate_target": 95,  # 95%
        },
        "kpi_status": {
            "time_to_preview": "ok" if (agg.avg_time_to_preview or 0) < preview_target else "warning",
            "success_rate": "ok" if agg.success_rate >= 0.95 else "warning",
        }
    }


def get_prometheus_metrics() -> str:
    """Render a small Prometheus exposition text payload from local telemetry.

    This is intentionally lightweight (no prometheus_client dependency) and
    suitable for scraping by a sidecar or basic Prometheus job. It aggregates:
    - aggregate job KPIs (counts / averages)
    - event-based counters (scanned from events.jsonl)

    The implementation is optimized for low-volume scraping and is best-effort.
    """
    agg = get_aggregate()
    lines: list[str] = []

    # Basic job-level metrics
    lines.append(f"# HELP montage_jobs_total Total jobs tracked by Montage AI")
    lines.append(f"# TYPE montage_jobs_total counter")
    lines.append(f"montage_jobs_total {int(agg.total_jobs)}")

    lines.append(f"# HELP montage_time_to_preview_seconds_avg Average time-to-preview (successes only)")
    lines.append(f"# TYPE montage_time_to_preview_seconds_avg gauge")
    if agg.avg_time_to_preview:
        lines.append(f"montage_time_to_preview_seconds_avg {agg.avg_time_to_preview:.3f}")
    else:
        lines.append("montage_time_to_preview_seconds_avg NaN")

    # Success rate
    lines.append(f"# HELP montage_success_rate_percent Success rate percent (0-100)")
    lines.append(f"# TYPE montage_success_rate_percent gauge")
    lines.append(f"montage_success_rate_percent {round(agg.success_rate * 100, 2)}")

    # Zero-initialized counters we will populate from events.jsonl
    counters = {
        'proxy_cache_hit': 0,
        'proxy_cache_miss': 0,
        'proxy_cache_evicted': 0,
        'jobstore_update_attempt': 0,
        'jobstore_update_failed': 0,
        'scene_detection_timeout': 0,
        'scene_detection_failed': 0,
    }

    try:
        evfile = get_collector().storage_path / 'events.jsonl'
        if evfile.exists():
            with open(evfile, 'r') as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                        et = ev.get('type')
                        if et in counters:
                            counters[et] += 1
                    except Exception:
                        continue
    except Exception:
        # Best-effort: do not fail scraping
        pass

    # Emit counters
    for k, v in counters.items():
        metric_name = f"montage_{k}_total"
        lines.append(f"# HELP {metric_name} Event counter for {k}")
        lines.append(f"# TYPE {metric_name} counter")
        lines.append(f"{metric_name} {int(v)}")

    # Proxy cache hit rate (derived)
    hits = counters.get('proxy_cache_hit', 0)
    misses = counters.get('proxy_cache_miss', 0)
    total = hits + misses
    hit_rate = (hits / total) if total > 0 else 0.0
    lines.append(f"# HELP montage_proxy_cache_hit_rate Proxy cache hit ratio (0-1)")
    lines.append(f"# TYPE montage_proxy_cache_hit_rate gauge")
    lines.append(f"montage_proxy_cache_hit_rate {hit_rate:.3f}")

    return "\n".join(lines) + "\n"
