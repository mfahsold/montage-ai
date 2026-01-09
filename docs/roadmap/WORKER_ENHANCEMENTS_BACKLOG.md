# Worker Enhancements Backlog

**Status:** 🟢 Phase 1 Complete (Production-Ready Connection Handling)  
**Next:** Phase 2-5 (Advanced fluxibri_core Patterns)

---

## 🔵 Phase 0: Docker Build Optimization & Version Validation

**Goal:** Implement intelligent cache invalidation and frontend version tracking for deployments.

**Context:** Discovered Docker layer caching (instruction-based, not file-content hashing) prevents fresh code inclusion during rebuilds. Need systematic approach to bust cache AND validate frontend/backend version alignment.

**Pseudocode:**

```dockerfile
# Dockerfile

ARG BUILD_VERSION=dev
ARG BUILD_DATE=unknown
ARG BUILD_COMMIT=unknown

# Layer 1: Base + APT (rarely changes)
FROM python:3.10-slim-bookworm as base
RUN apt-get update && apt-get install -y ...

# Layer 2: Python dependencies (changes when requirements.txt changes)
FROM base as deps
COPY requirements.txt .
RUN pip install --default-timeout=600 --retries 5 -r requirements.txt

# Layer 3: Source code (changes frequently - bust cache with BUILD_ARGS)
FROM deps as src
ARG BUILD_VERSION
ARG BUILD_DATE
ARG BUILD_COMMIT
ENV MONTAGE_VERSION=${BUILD_VERSION} \
    BUILD_DATE=${BUILD_DATE} \
    BUILD_COMMIT=${BUILD_COMMIT}
COPY . /app
RUN echo "{\"version\": \"${BUILD_VERSION}\", \"date\": \"${BUILD_DATE}\", \"commit\": \"${BUILD_COMMIT}\"}" > /app/build-info.json

# Layer 4: Final (user setup, etc.)
FROM src as final
RUN useradd -m montage && chown -R montage:montage /app
USER montage
WORKDIR /app/src
CMD ["python", "-m", "montage_ai.worker"]
```

```bash
# Build script: scripts/build-docker.sh

#!/bin/bash
set -e

VERSION=$(git describe --tags --always 2>/dev/null || echo "dev")
DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

echo "🐳 Building with VERSION=$VERSION DATE=$DATE COMMIT=$COMMIT"

docker build \
  --build-arg BUILD_VERSION="$VERSION" \
  --build-arg BUILD_DATE="$DATE" \
  --build-arg BUILD_COMMIT="$COMMIT" \
  -t ghcr.io/mfahsold/montage-ai:${VERSION} \
  -t ghcr.io/mfahsold/montage-ai:latest \
  .

# Verify version in image
echo "✅ Verifying build info..."
docker run --rm ghcr.io/mfahsold/montage-ai:${VERSION} \
  cat /app/build-info.json | jq .
```

```python
# src/montage_ai/version.py

import json
import os
from pathlib import Path

def get_build_info():
    """
    Return build-time environment info.
    Used by frontend (polling /api/status) and worker (logging startup).
    """
    build_info_path = Path(__file__).parent.parent / "build-info.json"
    
    if build_info_path.exists():
        with open(build_info_path) as f:
            return json.load(f)
    
    return {
        "version": os.getenv("MONTAGE_VERSION", "dev"),
        "date": os.getenv("BUILD_DATE", "unknown"),
        "commit": os.getenv("BUILD_COMMIT", "unknown"),
    }

# Usage in worker.py startup:
# build_info = get_build_info()
# logger.info(f"🚀 Starting Worker v{build_info['version']} ({build_info['commit'][:7]})")
```

```javascript
// src/montage_ai/web_ui/static/js/version-check.js

// Frontend version validation
async function validateBuildVersion() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    const backendVersion = data.build_info.version;
    const frontendVersion = window.MONTAGE_BUILD_VERSION || 'dev';
    
    if (backendVersion !== frontendVersion) {
      console.warn(`⚠️  Version mismatch: Frontend=${frontendVersion} Backend=${backendVersion}`);
      // Optionally show user notification or force refresh
    }
  } catch (e) {
    console.debug('Version check skipped:', e);
  }
}

// Call on page load
document.addEventListener('DOMContentLoaded', validateBuildVersion);
```

**Implementation Checklist:**
1. Add `ARG BUILD_VERSION/DATE/COMMIT` to Dockerfile
2. Create layered COPY strategy (Layer 2: deps, Layer 3: src with args)
3. Generate `build-info.json` during build
4. Create `scripts/build-docker.sh` for local builds
5. Update `src/montage_ai/version.py` to read build info
6. Add `/api/status` endpoint returning build_info
7. Add `version-check.js` to frontend for validation
8. Update CI/CD to pass build args on all deployments

**Benefits:**
- ✅ Cache busting through build args (fresh code each deploy)
- ✅ Version tracking across frontend/backend
- ✅ Deployment audit trail (git commit, date)
- ✅ Mismatch detection (useful for multi-service deployments)

**Estimated Effort:** 3 hours

---

## ✅ Phase 1: Production-Ready Connection Handling (COMPLETED)

**Implemented:**
- ✅ Exponential backoff (2s → 4s → 8s → 16s → 32s)
- ✅ Redis connection retry (5 attempts)
- ✅ Connection testing with `ping()`
- ✅ Named workers (`montage-worker-{pid}`)
- ✅ Socket timeouts + retry_on_timeout
- ✅ Max jobs limit (1000)
- ✅ Graceful shutdown (120s terminationGracePeriodSeconds)

**Files Modified:**
- `src/montage_ai/worker.py` (65 lines, production-ready)
- `deploy/k3s/app/worker-deployment.yaml` (full manifest)

---

## 🔵 Phase 2: XAUTOCLAIM - Crash Recovery

**Goal:** Automatically reclaim jobs stuck in "processing" after worker crashes.

**Pseudocode:**
```python
# In src/montage_ai/worker.py

def setup_crash_recovery(redis_conn, queue_name: str, consumer_group: str):
    """
    Setup XAUTOCLAIM for automatic recovery of jobs from crashed workers.
    
    Pattern from fluxibri_core:
    - Check for pending jobs older than 30s
    - Reclaim them to this worker
    - Re-process or move to DLQ
    """
    
    # 1. Ensure consumer group exists
    try:
        redis_conn.xgroup_create(
            name=f"queue:{queue_name}",
            groupname=consumer_group,
            id='0',
            mkstream=True
        )
        logger.info(f"📦 Consumer Group '{consumer_group}' created")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
    
    # 2. XAUTOCLAIM loop (run in background thread)
    while True:
        try:
            # Claim jobs pending for > 30s
            claimed = redis_conn.xautoclaim(
                name=f"queue:{queue_name}",
                groupname=consumer_group,
                consumername=f"montage-worker-{os.getpid()}",
                min_idle_time=30000,  # 30 seconds
                start_id='0-0',
                count=10
            )
            
            if claimed[1]:  # If any jobs reclaimed
                logger.warning(f"♻️ Reclaimed {len(claimed[1])} crashed jobs")
                for job_id, job_data in claimed[1]:
                    # Re-enqueue or send to DLQ based on retry count
                    retry_count = int(job_data.get(b'retry_count', 0))
                    if retry_count < 3:
                        # Re-enqueue with incremented retry
                        enqueue_job_with_retry(job_data, retry_count + 1)
                    else:
                        # Move to Dead Letter Queue
                        move_to_dlq(job_id, job_data, "Max retries exceeded")
            
            time.sleep(10)  # Check every 10s
            
        except Exception as e:
            logger.error(f"❌ XAUTOCLAIM failed: {e}")
            time.sleep(30)

# Usage in start_worker():
# threading.Thread(target=setup_crash_recovery, args=(redis_conn, 'default', 'montage-workers'), daemon=True).start()
```

**Implementation Steps:**
1. Add `xgroup_create` in `start_worker()` to ensure consumer group exists
2. Create background thread running `xautoclaim` loop
3. Add retry counter to job metadata
4. Test crash scenario: kill worker mid-job, verify reclaim

**Estimated Effort:** 4 hours

---

## 🔵 Phase 3: Job Status Publishing (Redis Pub/Sub)

**Goal:** Real-time UI updates for job progress.

**Pseudocode:**
```python
# In src/montage_ai/worker.py

class JobStatusPublisher:
    """
    Publish job status updates via Redis Pub/Sub for real-time UI.
    
    Pattern from fluxibri_core/pkg/jobstatus/jobstatus.go:
    - Publish to `job:status:{job_id}` channel
    - Status: QUEUED → CLAIMED → PROCESSING → COMPLETED/FAILED
    """
    
    def __init__(self, redis_conn):
        self.redis = redis_conn
    
    def publish_status(self, job_id: str, status: str, progress: int = 0, message: str = ""):
        """
        Publish job status update.
        
        Args:
            job_id: Unique job identifier
            status: QUEUED, CLAIMED, PROCESSING, COMPLETED, FAILED
            progress: 0-100
            message: Optional status message
        """
        payload = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "timestamp": time.time(),
            "worker": f"montage-worker-{os.getpid()}"
        }
        
        channel = f"job:status:{job_id}"
        self.redis.publish(channel, json.dumps(payload))
        logger.debug(f"📡 Published {status} for job {job_id[:8]}...")

# In montage job processing:
def process_montage_job(job):
    publisher = JobStatusPublisher(redis_conn)
    
    try:
        publisher.publish_status(job.id, "CLAIMED", 0, "Job claimed by worker")
        
        # Beat detection
        publisher.publish_status(job.id, "PROCESSING", 10, "Analyzing audio...")
        beats = detect_beats(audio_file)
        
        # Scene detection
        publisher.publish_status(job.id, "PROCESSING", 30, "Detecting scenes...")
        scenes = detect_scenes(video_file)
        
        # Assembly
        publisher.publish_status(job.id, "PROCESSING", 50, "Assembling montage...")
        assemble_clips(beats, scenes)
        
        # Rendering
        publisher.publish_status(job.id, "PROCESSING", 70, "Rendering video...")
        render_video()
        
        publisher.publish_status(job.id, "COMPLETED", 100, "Montage ready!")
        
    except Exception as e:
        publisher.publish_status(job.id, "FAILED", 0, str(e))
        raise
```

**Frontend Integration (web_ui/templates/jobs.html):**
```javascript
// Subscribe to job status updates
const eventSource = new EventSource(`/api/jobs/${jobId}/stream`);

eventSource.onmessage = (event) => {
    const status = JSON.parse(event.data);
    
    // Update progress bar
    document.getElementById('progress').style.width = `${status.progress}%`;
    
    // Update status text
    document.getElementById('status-message').textContent = status.message;
    
    // Update status badge
    const badge = document.getElementById('status-badge');
    badge.className = `badge badge-${status.status.toLowerCase()}`;
    badge.textContent = status.status;
};
```

**Implementation Steps:**
1. Add `JobStatusPublisher` class to `worker.py`
2. Integrate `publish_status()` calls in `MontageBuilder.build_montage()`
3. Create Flask SSE endpoint `/api/jobs/<id>/stream`
4. Add JavaScript event listener in `jobs.html`

**Estimated Effort:** 6 hours

---

## 🔵 Phase 4: Dead Letter Queue (DLQ)

**Goal:** Capture and inspect failed jobs after max retries.

**Pseudocode:**
```python
# In src/montage_ai/worker.py

def move_to_dlq(job_id: str, job_data: dict, reason: str):
    """
    Move failed job to Dead Letter Queue.
    
    Pattern from fluxibri_core:
    - Stream name: `jobs:dead`
    - Store job data + failure metadata
    - Allow manual inspection/retry
    """
    
    dlq_payload = {
        "job_id": job_id,
        "original_data": job_data,
        "failure_reason": reason,
        "failed_at": time.time(),
        "worker": f"montage-worker-{os.getpid()}",
        "retry_count": job_data.get('retry_count', 0)
    }
    
    # Add to DLQ stream
    redis_conn.xadd(
        name="jobs:dead",
        fields=dlq_payload,
        maxlen=1000  # Keep last 1000 failed jobs
    )
    
    logger.error(f"💀 Job {job_id[:8]} moved to DLQ: {reason}")

# In job processing error handler:
except Exception as e:
    retry_count = job.meta.get('retry_count', 0)
    
    if retry_count >= 3:
        move_to_dlq(job.id, job.data, str(e))
    else:
        # Re-enqueue with backoff
        retry_delay = 2 ** retry_count  # 2s, 4s, 8s
        queue.enqueue_in(timedelta(seconds=retry_delay), process_montage_job, job.data, meta={'retry_count': retry_count + 1})
```

**DLQ Inspection Tool (CLI):**
```python
# scripts/inspect_dlq.py

def list_dead_jobs():
    """List all jobs in Dead Letter Queue."""
    
    jobs = redis_conn.xrange("jobs:dead", count=100)
    
    for job_id, data in jobs:
        print(f"Job: {data[b'job_id'].decode()}")
        print(f"  Failed: {datetime.fromtimestamp(float(data[b'failed_at']))}")
        print(f"  Reason: {data[b'failure_reason'].decode()}")
        print(f"  Retries: {data[b'retry_count'].decode()}")
        print()

def retry_dead_job(job_id: str):
    """Manually retry a failed job from DLQ."""
    
    # Find job in DLQ
    jobs = redis_conn.xrange("jobs:dead")
    for stream_id, data in jobs:
        if data[b'job_id'].decode() == job_id:
            # Re-enqueue with fresh retry count
            original_data = json.loads(data[b'original_data'])
            queue.enqueue(process_montage_job, original_data, meta={'retry_count': 0})
            
            # Remove from DLQ
            redis_conn.xdel("jobs:dead", stream_id)
            
            print(f"✅ Job {job_id} requeued")
            return
    
    print(f"❌ Job {job_id} not found in DLQ")

# Usage:
# python scripts/inspect_dlq.py list
# python scripts/inspect_dlq.py retry <job_id>
```

**Implementation Steps:**
1. Add `move_to_dlq()` function to `worker.py`
2. Integrate DLQ logic in exception handlers
3. Create `scripts/inspect_dlq.py` CLI tool
4. Add DLQ viewer in Web UI (`/admin/dlq`)

**Estimated Effort:** 5 hours

---

## 🔵 Phase 5: Prometheus Metrics

**Goal:** Monitor worker health and performance.

**Pseudocode:**
```python
# In src/montage_ai/metrics.py

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Metrics
jobs_processed = Counter('montage_jobs_processed_total', 'Total jobs processed', ['status'])
jobs_duration = Histogram('montage_job_duration_seconds', 'Job processing duration')
worker_queue_size = Gauge('montage_queue_size', 'Number of jobs in queue')
redis_pool_size = Gauge('montage_redis_pool_size', 'Redis connection pool size')
redis_pool_in_use = Gauge('montage_redis_pool_in_use', 'Active Redis connections')

# In worker.py:
def process_montage_job_instrumented(job):
    """Instrumented version with metrics."""
    
    start_time = time.time()
    
    try:
        result = process_montage_job(job)
        jobs_processed.labels(status='success').inc()
        return result
        
    except Exception as e:
        jobs_processed.labels(status='failed').inc()
        raise
        
    finally:
        duration = time.time() - start_time
        jobs_duration.observe(duration)

def collect_metrics_loop():
    """Background thread collecting metrics."""
    
    while True:
        try:
            # Queue size
            queue_size = len(Queue(connection=redis_conn))
            worker_queue_size.set(queue_size)
            
            # Redis pool stats
            pool_info = redis_conn.connection_pool.get_connection('ping').connection_pool
            redis_pool_size.set(pool_info.max_connections)
            redis_pool_in_use.set(pool_info._in_use_connections)
            
        except Exception as e:
            logger.error(f"❌ Metrics collection failed: {e}")
        
        time.sleep(15)  # Collect every 15s

# In start_worker():
# Start Prometheus exporter on :9090
start_http_server(9090)
threading.Thread(target=collect_metrics_loop, daemon=True).start()
```

**Grafana Dashboard Config:**
```yaml
# deploy/k3s/monitoring/grafana-dashboard.yaml

panels:
  - title: "Job Success Rate"
    query: "rate(montage_jobs_processed_total{status='success'}[5m])"
  
  - title: "Job Duration (p95)"
    query: "histogram_quantile(0.95, montage_job_duration_seconds_bucket)"
  
  - title: "Queue Size"
    query: "montage_queue_size"
  
  - title: "Redis Pool Usage"
    query: "montage_redis_pool_in_use / montage_redis_pool_size * 100"
```

**Implementation Steps:**
1. Add `prometheus_client` to `requirements.txt`
2. Create `src/montage_ai/metrics.py`
3. Integrate metrics in `worker.py`
4. Expose `:9090/metrics` in worker Deployment
5. Create Grafana dashboard YAML

**Estimated Effort:** 6 hours

---

## 📊 Priority Matrix

| Phase | Impact | Effort | Priority |
|-------|--------|--------|----------|
| Phase 2: XAUTOCLAIM | 🔥 High | 4h | **P1** |
| Phase 3: Status Publishing | 🔥 High | 6h | **P1** |
| Phase 4: DLQ | 🟡 Medium | 5h | **P2** |
| Phase 5: Metrics | 🟡 Medium | 6h | **P2** |

**Recommendation:** Start with Phase 2 (crash recovery) and Phase 3 (real-time status) as they provide immediate production value.

---

## 🔗 References

**fluxibri_core Implementations:**
- Redis Streams: `pkg/redisstream/consumer.go`
- Worker Pattern: `archive/legacy_v1_cleanup_20251219/llamacpp/simpleworker/`
- Job Status: `pkg/jobstatus/jobstatus.go`
- Metrics: `pkg/metrics/prometheus.go`

**Documentation:**
- [Redis Streams Consumer Groups](https://redis.io/docs/data-types/streams-tutorial/)
- [XAUTOCLAIM Command](https://redis.io/commands/xautoclaim/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
