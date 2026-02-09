# Phase 2: Go Worker Canary Deployment

**Status:** ✅ Ready for Phase 2  
**Binary:** `go/montage-worker` (6.5 MB, statically-linked)  
**Latest Commit:** `0f7ac27` (build fixes + go.sum lock file)

---

## What's Complete (Phase 1 ✅)

- ✅ Go worker scaffold (11 source files, 559 LoC)
- ✅ RQ-compatible Redis queue integration
- ✅ Goroutine pool orchestration (configurable concurrency)
- ✅ Python subprocess hooks (for Creative Director calls)
- ✅ Graceful shutdown (SIGTERM + context cancellation)
- ✅ Structured logging with colored output
- ✅ Multi-arch Dockerfile (amd64, arm64)
- ✅ Binary compiles successfully (no build errors)
- ✅ K8s canary deployment manifest
- ✅ Migration strategy document (MIGRATION.md)

---

## Phase 2: Canary Deployment Checklist

### 1. Build & Push Multi-Arch Docker Image

**Prerequisites:**
- Docker with buildx support (`docker buildx version`)
- Registry credentials configured (`~/.docker/config.json`)
- Target registry set in `go/build-and-push.sh` (currently: `registry.registry.svc.cluster.local:5000`)

**Command:**
```bash
cd go
./build-and-push.sh
```

**What this does:**
- Builds Go binary optimized for amd64 + arm64
- Creates Alpine container (38 MB base)
- Pushes to `registry.registry.svc.cluster.local:5000/montage-ai-worker:latest`

**Expected output:**
```
✅ Building image for amd64...
✅ Building image for arm64...
✅ Pushing multi-arch manifest...
📦 Image ready: registry.registry.svc.cluster.local:5000/montage-ai-worker:latest
```

---

### 2. Deploy Canary to Kubernetes

**Deploy 1 Go replica alongside Python workers:**

```bash
kubectl apply -f deploy/k3s/overlays/cluster/worker-go-canary.yaml
```

**Verify deployment:**
```bash
# Check pod status
kubectl get pods -n montage-ai -l app.kubernetes.io/component=worker-go

# Watch logs
kubectl logs -f -l app.kubernetes.io/component=worker-go -n montage-ai

# Check resource usage
kubectl top pods -n montage-ai -l app.kubernetes.io/component=worker-go
```

**Expected behavior:**
- 1 Go pod running alongside ~5 Python workers
- Memory usage: ~50-100 MB (vs. ~200 MB Python)
- CPU idle: <5%, under load: 50-100% (proportional to jobs processed)
- Logs show: `✅ Worker initialized with X goroutines`

---

### 3. Monitor Canary Metrics (24-48 Hours)

**Key metrics to track:**

| Metric | Target | Tool |
|--------|--------|------|
| **Memory per pod** | <100 MB | `kubectl top`, Prometheus |
| **Job success rate** | >99% | Redis logs, K8s events |
| **Job latency (p50)** | <5 sec | Custom metrics in Go worker |
| **Job latency (p99)** | <15 sec | Same |
| **Pod restarts** | 0 | `kubectl describe pod` |
| **CPU utilization** | 40-80% under load | `kubectl top` |

**Prometheus queries (if available):**

```promql
# Memory usage per pod
container_memory_usage_bytes{pod="montage-ai-worker-go-*"}

# CPU usage per pod
rate(container_cpu_usage_seconds_total{pod="montage-ai-worker-go-*"}[5m])

# Job queue depth (should shrink with Go worker)
redis_connected_clients{job="redis"}
```

**Manual testing:**

```bash
# Upload test video from Web UI
# Check job processing in Redis
redis-cli -h redis.montage-ai.svc.cluster.local

> KEYS rq:queue:*
> LLEN rq:queue:default
> LLEN rq:queue:high-priority

# Check job completions in /data/output/
kubectl exec -it montage-ai-web-xxx -c web -- \
  ls -lh /data/output/ | tail -10
```

---

### 4. Validation Gates (Decision Points)

**✅ PROCEED to Phase 3 if:**
- [ ] Pod runs for 48 hours without crash or restart
- [ ] Memory usage stays <150 MB
- [ ] Job success rate ≥ 99%
- [ ] No error patterns in logs (grep for "ERROR\|panic\|fatal")
- [ ] Latency comparable to Python worker (within ±10%)

**⚠️ INVESTIGATE if:**
- [ ] Memory > 200 MB (goroutine leak or job leak)
- [ ] Job success <95% (job parsing, subprocess issue)
- [ ] Restarts >0 (signal handling issue)
- [ ] CPU constantly >95% (pool too small, increase WORKER_CPUS)

**🛑 ROLLBACK if:**
- [ ] Pod crashes repeatedly (logs show panic or fatal error)
- [ ] Jobs processed but not completed in /data/output/
- [ ] Queue backing up (jobs accumulating in Redis)

---

## Phase 2 Rollback Plan

**If issues detected:**

```bash
# 1. Delete canary
kubectl delete -f deploy/k3s/overlays/cluster/worker-go-canary.yaml

# 2. Verify queue is draining (Python workers continue)
redis-cli LLEN rq:queue:default

# 3. Check logs for root cause
kubectl logs -n montage-ai-render montage-ai-render-xxx

# 4. Fix issue (e.g., update config, rebuild image)
# Edit: go/internal/config/config.go or go/pkg/worker/pool.go

# 5. Rebuild and retry
cd go && ./build-and-push.sh
kubectl apply -f deploy/k3s/overlays/cluster/worker-go-canary.yaml
```

**Zero data loss:**
- All jobs remain in Redis queue
- Python workers continue processing
- No deployments affected

---

## Configuration for Phase 2

**Environment variables (set in K8s deployment):**

```yaml
env:
  - name: REDIS_HOST
    value: redis.montage-ai.svc.cluster.local
  - name: REDIS_PORT
    value: "6379"
  - name: REDIS_PASSWORD
    value: ""  # Set if Redis requires auth
  - name: WORKER_CPUS
    value: "4"  # 4 CPUs × 250 goroutines = 1000 concurrent jobs
  - name: WORKER_QUEUES
    value: "default,high-priority,gpu-tasks"
  - name: WORKER_LOG_LEVEL
    value: "info"  # debug, info, warn, error
```

**Kubernetes resource limits:**

```yaml
resources:
  requests:
    memory: 64Mi
    cpu: 100m
  limits:
    memory: 512Mi    # Increase if goroutine pool scales
    cpu: 4000m       # 4 CPUs per WORKER_CPUS env
```

---

## Phase 2 Success Criteria

After 48 hours of canary monitoring:

**Performance:**
- [ ] Processing jobs from Redis queue
- [ ] Memory stable <150 MB
- [ ] CPU utilization proportional to job load
- [ ] Job latency within 10% of Python worker

**Reliability:**
- [ ] Zero restarts / crashes
- [ ] No error logs (grep for ERROR, panic, fatal)
- [ ] Output files correctly written to /data/output/

**Operations:**
- [ ] Graceful SIGTERM (drains in-flight jobs)
- [ ] Scaling up: Ready for Phase 3 (2-5 replicas)

**Then Proceed to Phase 3:**
- Scale Go from 1 → 5 → 10 replicas
- Monitor queue depth and pod density
- Prepare Python worker sunsetting

---

## Quick Reference: Phase 2 Commands

```bash
# Build & push image
cd go && ./build-and-push.sh

# Deploy canary
kubectl apply -f deploy/k3s/overlays/cluster/worker-go-canary.yaml

# Monitor
kubectl logs -f -l app.kubernetes.io/component=worker-go -n montage-ai
kubectl top pods -n montage-ai

# Test (from Web UI)
# 1. Upload video
# 2. Check /data/output/ for processed file
# 3. Verify logs: kubectl logs -f <go-worker-pod>

# Rollback
kubectl delete -f deploy/k3s/overlays/cluster/worker-go-canary.yaml

# Investigate issues
kubectl describe pod <pod-name> -n montage-ai
kubectl debug pod <pod-name> -n montage-ai
```

---

## Success Example: Phase 2 Complete

After 48 hours:

```
✅ Go worker canary running for 48h:
   - Pod: montage-ai-worker-go-canary-xyz (Ready 1/1)
   - Memory: 87 MB (↓ 56% vs Python)
   - CPU: 42% under load (equivalent latency)
   - Restarts: 0
   - Jobs processed: 1,247
   - Success rate: 99.8%

➡️  Ready to proceed to Phase 3: Scale to 5 replicas

Expected benefit:
   - 5× more job throughput
   - Pod density improves: 8 replicas/server vs 4 (Python only)
   - Cluster autoscaling becomes more efficient
```

---

## Next: Phase 3 (Week 2-3)

Once canary succeeds, scale Go worker:

```bash
kubectl scale deployment montage-ai-worker-go-canary \
  --replicas=5 -n montage-ai
```

Then monitor for 1 week before final Phase 4 (Python sunsetting).

See full roadmap in `go/MIGRATION.md`.

---

**Questions? Check:**
- `go/README.md` — Architecture & module structure
- `go/MIGRATION.md` — Full 4-phase migration plan
- `deploy/k3s/overlays/cluster/worker-go-canary.yaml` — K8s manifest

**Build & deploy:** `cd go && ./build-and-push.sh`
