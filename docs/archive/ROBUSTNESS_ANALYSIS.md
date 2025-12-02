# Montage AI - Robustness Analysis & Recommendations

**Date:** 2025-12-02
**System:** montage-ai Web UI
**Analyst:** Claude Code

## Executive Summary

The web interface implementation is **functionally complete** and properly integrated with backend features. However, there are critical robustness gaps that could lead to crashes and resource exhaustion, particularly when using cgpu or processing multiple large videos.

**Status:** ✅ Frontend Complete | ⚠️ Robustness Needs Improvement

---

## 1. Frontend Feature Review

### ✅ What's Working Well

| Feature | Frontend | Backend | Status |
|---------|----------|---------|--------|
| File Upload (Videos/Music) | ✅ | ✅ | Working |
| Style Selection (7 styles) | ✅ | ✅ | All mapped correctly |
| Creative Prompt | ✅ | ✅ | Working |
| Enhancement Options | ✅ | ✅ | All options available |
| Job Creation & Tracking | ✅ | ✅ | Working |
| Auto-polling Status | ✅ | N/A | Working (3s intervals) |
| File Downloads | ✅ | ✅ | Working |
| Timeline Export (OTIO/EDL/CSV) | ✅ | ✅ | Working |

### Frontend Code Quality

**Strengths:**
- Clean, minimal design following KISS principle
- Vanilla JavaScript (no framework bloat)
- Responsive design
- Clear separation of concerns
- Good error handling in API calls

**Test Coverage:**
- Basic API tests exist
- File validation tested
- Job creation/status tested
- ✅ Tests pass (when dependencies installed)

---

## 2. Crash & OOM Analysis

### System Resources (Current)

```
Memory: 19Gi available / 29Gi total
Disk:   272G available / 937G total
Swap:   44Gi available (unused)
```

**Conclusion:** Hardware is not the bottleneck. The issue is **lack of resource constraints** in containers.

### Crash Investigation Findings

**Last Container Run:**
- ✅ Completed successfully (33 cuts, 211s video)
- ✅ No OOM errors in logs
- ⚠️ Some unrelated containers restarting (worker_small, worker_micro, worker_big)

**Potential OOM Triggers:**
1. **CGPU Processing:** LLM inference can use 4-8GB RAM
2. **Video Processing:** MoviePy loads entire clips into memory
3. **Multiple Jobs:** No queue limit, jobs run in threads
4. **Upscaling:** Real-ESRGAN uses 2-4GB RAM per frame
5. **Beat Detection:** librosa loads entire audio file

---

## 3. Critical Robustness Issues

### 🔴 HIGH SEVERITY

#### Issue #1: No Memory Limits in Docker
**Impact:** Container can consume all system RAM, causing OOM killer to terminate processes

**Current State:**
```yaml
# docker-compose.web.yml - NO LIMITS SET
services:
  web-ui:
    build: .
    # ❌ No memory limits
    # ❌ No CPU limits
```

**Evidence:**
```bash
docker stats
# MEM LIMIT shows "0B" = unlimited
```

#### Issue #2: Unbounded Job Queue
**Impact:** Multiple concurrent jobs can exhaust memory

**Code Location:** `src/montage_ai/web_ui/app.py:50-119`

```python
def run_montage(job_id: str, style: str, options: dict):
    # ❌ No memory monitoring
    # ❌ No resource checks before starting
    # ❌ Runs in daemon thread (no cleanup on shutdown)
    thread = threading.Thread(target=run_montage, ...)
    thread.daemon = True  # Dies on crash, no cleanup
    thread.start()
```

#### Issue #3: No Video Memory Management
**Impact:** Large videos (>1GB) loaded entirely into RAM

**Code Location:** `src/montage_ai/editor.py` (MoviePy usage)

```python
# MoviePy loads entire clip into memory
clip = VideoFileClip(path)  # ❌ No streaming
```

#### Issue #4: Non-Persistent Job State
**Impact:** All job history lost on crash/restart

```python
jobs = {}  # ❌ In-memory only
job_lock = threading.Lock()
```

### 🟡 MEDIUM SEVERITY

#### Issue #5: No Graceful Degradation
- No fallback when memory is low
- No automatic quality reduction
- No proxy generation for large files

#### Issue #6: CGPU Errors Not Handled
- Network failures not caught
- API timeout not configurable
- No fallback to local processing

#### Issue #7: Long-Running Job Timeout
- Fixed 1-hour timeout
- No progress reporting during processing
- User has no way to cancel job

---

## 4. Recommended Solutions

### Priority 1: Container Resource Limits

**Implementation:**

```yaml
# docker-compose.web.yml
services:
  web-ui:
    deploy:
      resources:
        limits:
          memory: 8G      # Prevent OOM
          cpus: '4.0'     # Prevent CPU saturation
        reservations:
          memory: 2G      # Guarantee minimum
          cpus: '1.0'
```

**Kubernetes:**

```yaml
resources:
  limits:
    memory: "8Gi"
    cpu: "4000m"
  requests:
    memory: "2Gi"
    cpu: "1000m"
```

### Priority 2: Memory Monitoring

**Add to `app.py`:**

```python
import psutil

def check_memory_available(required_gb=4):
    """Check if enough memory available before starting job."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    return available_gb >= required_gb

def run_montage(job_id, style, options):
    # Check memory before starting
    if not check_memory_available():
        with job_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Insufficient memory. Try again later."
        return
    # ... proceed with processing
```

### Priority 3: Job Queue with Limits

**Add to `app.py`:**

```python
from collections import deque

MAX_CONCURRENT_JOBS = 2
job_queue = deque()
active_jobs = 0

def job_worker():
    """Background worker that processes job queue."""
    global active_jobs
    while True:
        if active_jobs < MAX_CONCURRENT_JOBS and job_queue:
            job = job_queue.popleft()
            active_jobs += 1
            try:
                run_montage(**job)
            finally:
                active_jobs -= 1
        time.sleep(1)

# Start worker thread
worker = threading.Thread(target=job_worker, daemon=False)
worker.start()
```

### Priority 4: Persistent Job State

**Use SQLite for job storage:**

```python
import sqlite3

def init_db():
    conn = sqlite3.connect('jobs.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT,
            created_at TEXT,
            options TEXT
        )
    ''')
    conn.commit()
    return conn

# Load jobs from DB on startup
def load_jobs():
    conn = sqlite3.connect('jobs.db')
    rows = conn.execute('SELECT * FROM jobs').fetchall()
    for row in rows:
        jobs[row[0]] = json.loads(row[3])
```

### Priority 5: Video Streaming & Proxy Generation

**Add automatic proxy generation for large files:**

```python
def should_use_proxy(video_path):
    """Check if video is too large for direct processing."""
    size_mb = Path(video_path).stat().st_size / (1024**2)
    return size_mb > 500  # 500MB threshold

def create_proxy(video_path):
    """Create low-res proxy for large videos."""
    proxy_path = video_path.with_suffix('.proxy.mp4')
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', 'scale=960:-2',  # 960p width
        '-c:v', 'libx264', '-crf', '28',
        '-preset', 'veryfast',
        str(proxy_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return proxy_path
```

### Priority 6: Enhanced Error Handling

**Add retry mechanism:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_cgpu_api(prompt):
    """Call cgpu API with retry logic."""
    response = requests.post(CGPU_ENDPOINT, json={"prompt": prompt}, timeout=30)
    response.raise_for_status()
    return response.json()
```

---

## 5. Testing Recommendations

### Load Testing

```bash
# Test concurrent job submission
for i in {1..5}; do
  curl -X POST http://localhost:5000/api/jobs \
    -H "Content-Type: application/json" \
    -d '{"style":"dynamic","enhance":true}'
done

# Monitor memory usage
watch -n 1 docker stats
```

### OOM Testing

```bash
# Set strict memory limit to test behavior
docker run --memory=2g montage-ai:latest

# Try to trigger OOM
# Submit job with large videos + upscale + cgpu
```

### Recovery Testing

```bash
# Kill container during job
docker kill <container-id>

# Restart and check job recovery
docker-compose -f docker-compose.web.yml up

# Verify jobs can resume or show proper state
```

---

## 6. Deployment Best Practices

### Local Docker

```yaml
version: '3.8'
services:
  web-ui:
    build: .
    restart: unless-stopped  # Auto-restart on crash
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
    volumes:
      - ./data:/data
      - ./jobs.db:/app/jobs.db  # Persist job state
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: montage-ai-web
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: web-ui
        resources:
          limits:
            memory: "8Gi"
            cpu: "4000m"
          requests:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/status
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/status
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## 7. Monitoring & Alerts

### Prometheus Metrics (Future Enhancement)

```python
from prometheus_client import Counter, Gauge, Histogram

job_counter = Counter('montage_jobs_total', 'Total jobs created')
job_duration = Histogram('montage_job_duration_seconds', 'Job processing time')
memory_usage = Gauge('montage_memory_bytes', 'Current memory usage')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

### Log Aggregation

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/logs/montage-ai.log'),
        logging.StreamHandler()
    ]
)
```

---

## 8. Implementation Roadmap

### Week 1: Critical Fixes
- [ ] Add Docker memory limits
- [ ] Implement memory checking before job start
- [ ] Add job queue with concurrency limits
- [ ] Test under constrained resources

### Week 2: Persistence & Recovery
- [ ] Add SQLite job persistence
- [ ] Implement job recovery on restart
- [ ] Add health checks
- [ ] Test crash recovery

### Week 3: Graceful Degradation
- [ ] Auto-generate proxies for large videos
- [ ] Add retry logic for CGPU
- [ ] Implement fallback to local processing
- [ ] Add progress reporting

### Week 4: Monitoring & Testing
- [ ] Add Prometheus metrics
- [ ] Set up log aggregation
- [ ] Load testing
- [ ] Documentation updates

---

## 9. Conclusion

The Montage AI web interface is **well-designed and functional**, but needs **robustness hardening** for production use. The main risks are:

1. **OOM crashes** from unbounded memory usage
2. **Lost job state** from in-memory storage
3. **No graceful degradation** under resource pressure

**Recommended Action:** Implement Priority 1-3 solutions before production deployment.

**Estimated Effort:** 2-3 days for critical fixes, 1 week for full robustness suite.

---

## Appendix: Related Files

- `src/montage_ai/web_ui/app.py` - Main Flask application
- `src/montage_ai/web_ui/templates/index.html` - Frontend HTML
- `src/montage_ai/web_ui/static/app.js` - Frontend JavaScript
- `docker-compose.web.yml` - Docker Compose configuration
- `Dockerfile` - Container build instructions
- `src/montage_ai/editor.py` - Video processing engine

---

**Report Generated:** 2025-12-02
**Review Status:** Complete
**Next Review:** After implementing Priority 1-3 fixes
