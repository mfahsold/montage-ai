# Go Worker Migration Guide

## Executive Summary

This is **Phase 1** of a gradual migration from Python RQ Worker → Go Worker. The go binary will run alongside the Python worker, processing jobs from shared Redis queue.

**No breaking changes.** Python API, CLI, and config stay the same.

---

## Architecture Change

### Before (Python Only)
```
Web UI → Upload → FastAPI → Redis Queue
                    ↓
                RQ Worker (Python)
                    ↓
                Output /data/
```

### After (Hybrid)
```
Web UI → Upload → FastAPI → Redis Queue
                    ↓
        ┌───────────┴────────────┐
        ↓                        ↓
    RQ Worker            Go Worker Goroutines
    (Python)             (Concurrency X10)
        ↓                        ↓
        └───────────┬────────────┘
                    ↓
            Output /data/
```

---

## Migration Phases

### ✅ Phase 1: POC (Completed)
- [x] Go worker scaffold
- [x] Redis queue integration
- [x] Python subprocess calls (for Creative Director)
- [x] Kubernetes canary deployment

### 🟡 Phase 2: Canary Deployment (Next)
- [ ] Build Go image
- [ ] Deploy 1 replica alongside Python worker
- [ ] Monitor metrics (CPU, memory, latency, job success rate)
- [ ] Route 10% of jobs via queue priority

### 🟡 Phase 3: Gradual Rollout (Week 2-3)
- [ ] Scale Go (2 → 5 → 10 replicas)
- [ ] Keep Python as fallback
- [ ] Monitor for issues
- [ ] Adjust resource limits

### 🟡 Phase 4: Sunsetting (Week 4)
- [ ] Delete Python RQ worker
- [ ] Update documentation
- [ ] Archive old code

---

## Quick Start

### Build Locally

```bash
cd go
go mod download
go build -o montage-worker ./cmd/worker

# Test locally (requires Redis running)
REDIS_HOST=localhost ./montage-worker
```

### Build Docker Image

```bash
cd go
./build-and-push.sh
# Output: registry.example.com/montage-ai-worker:go-v1-canary
```

### Deploy Canary (1 Replica)

```bash
kubectl apply -f deploy/k3s/overlays/cluster/worker-go-canary.yaml

# Verify
kubectl get pods -n montage-ai -l app.kubernetes.io/component=worker-go
kubectl logs -n montage-ai -l app.kubernetes.io/component=worker-go -f
```

---

## Monitoring the Migration

### Key Metrics to Watch

| Metric | Python Baseline | Go Target | Success Criteria |
|--------|-------------|----------|---|
| Memory per pod | ~200 MB | ~50 MB | ✅ -75% |
| Max concurrent jobs | ~50 (16GB server) | ~500 (same) | ✅ X10 |
| FFmpeg startup latency | 200-300 ms | 10-20 ms | ✅ X15 |
| Average job duration | ~45 sec (video) | ~45 sec (same) | ✅ No regression |
| Job success rate | 99.2% | 99.2% | ✅ No regression |

### Monitor Commands

```bash
# Watch pod memory usage
kubectl top pods -n montage-ai -l app.kubernetes.io/component=worker-go

# Watch job queue length
kubectl exec -n montage-ai redis-0 -- redis-cli LLEN "rq:queue:default"

# Get error logs
kubectl logs -n montage-ai -l app.kubernetes.io/component=worker-go --tail=100 | grep ERROR

# Check job status
python3 -m montage_ai.cli jobs list --status pending
```

---

## Rollback Plan

If issues arise:

```bash
# Immediately revert to Python-only
kubectl delete deployment montage-ai-worker-go -n montage-ai

# Scale up Python worker
kubectl scale deployment montage-ai-worker -n montage-ai --replicas=5

# Investigate logs
kubectl logs -n montage-ai -l app.kubernetes.io/component=worker -f
```

---

## Known Limitations (Phase 1)

- ❌ **GPU task scheduling** — Not yet implemented
- ❌ **Metrics export** — Use `kubectl top` for now; Prometheus integration in Phase 2
- ❌ **Graceful shutdown** — Jobs may interrupt on pod termination (TODO: SIGTERM handler)
- ❌ **Job retry logic** — Basic exponential backoff (no circuit breaker)

These will be added in Phase 2+ as needed.

---

## Next Steps

1. **Build & Deploy Canary:** `./build-and-push.sh` + `kubectl apply`
2. **Monitor:** Watch metrics for 24-48 hours
3. **Scale:** If stable, `kubectl scale --replicas=3`
4. **Validate:** Confirm job success rate, latency, resource usage
5. **Gradual Rollout:** Move to 100% Go workers over 1 week

---

## Troubleshooting

### Pod won't start
```bash
kubectl describe pod <POD_NAME> -n montage-ai
# Check events section for image pull errors or mount issues
```

### Jobs not processing
```bash
# Check Redis connection
kubectl exec montage-ai-worker-go-abc123 -c montage-ai-worker-go -- montage-worker --health-check

# Check queue
kubectl exec redis-0 -- redis-cli LLEN "rq:queue:default"
```

### High memory usage
```bash
# Reduce worker goroutines
kubectl set env deployment/montage-ai-worker-go WORKER_CPUS=2 -n montage-ai
```

### Slow job processing
```bash
# Check if Python analyzer is the bottleneck
kubectl logs montage-ai-worker-go-abc123 -n montage-ai | grep -i "python\|analyzer"
```

---

## References

- [Go concurrency patterns](https://go.dev/blog/pipelines)
- [Redis streams](https://redis.io/commands/xread/)
- [Kubernetes deployment best practices](https://kubernetes.io/docs/concepts/configuration/overview/)
