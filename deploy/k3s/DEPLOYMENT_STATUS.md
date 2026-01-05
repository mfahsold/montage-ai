# Deployment Status & Implementation Plan

**Last Updated:** January 5, 2026, 16:13 UTC

## ✅ Successfully Deployed

**Status:** 🟢 LIVE at http://montage-ai.fluxibri.lan

### Deployment Details

```yaml
Namespace:        montage-ai
Image:            192.168.1.12:5000/montage-ai:latest
Registry:         Cluster HTTP Registry (192.168.1.12:5000)
Pods:             1/1 Running (AMD64 only)
Storage:          local-path (160Gi total: 50Gi input + 100Gi output + 10Gi music)
Ingress:          http://montage-ai.fluxibri.lan (Traefik, HTTP entrypoint)
Monitoring:       ServiceMonitor configured for Prometheus
Architecture:     AMD64 only (ARM64 pending Phase 2)
Pod Security:     privileged (hardening in Phase 6)
```

### What Was Fixed (Phase 1 - Service Stability)

**Problem:** Container was running `montage_ai.editor` (batch job) instead of web server.

**Solution:**
1. ✅ Changed Dockerfile CMD from `ENTRYPOINT ["python", "-u", "-m", "montage_ai.editor"]` to `CMD ["python", "-u", "-m", "montage_ai.web_ui.app"]`
2. ✅ Added `/health` endpoint in `app.py` for Kubernetes liveness/readiness probes
3. ✅ Changed default port from 5000 to 8080 (Kubernetes standard)
4. ✅ Updated HEALTHCHECK to use curl instead of Python check
5. ✅ Fixed Ingress to use HTTP (`web` entrypoint) instead of HTTPS (`websecure`)

---

## 📋 Strategic Implementation Plan

### Phase 1: Service Stability ✅ COMPLETE

**Goal:** Make web UI run as persistent service, not one-shot job.

- [x] Update Dockerfile CMD to start Flask server
- [x] Add `/health` endpoint implementation  
- [x] Test readiness/liveness probes
- [x] Verify Ingress routing to web UI
- [x] Fix HTTP/HTTPS routing

**Status:** ✅ Complete (Jan 5, 2026)

---

### Phase 2: Multi-Architecture Support 🔄 NEXT

**Goal:** Support ARM64 nodes (Raspberry Pi, Jetson) in addition to AMD64.

**Priority:** HIGH  
**ETA:** Week 2 (Jan 12, 2026)

**Tasks:**
- [ ] Set up Docker buildx for multi-arch (`linux/amd64,linux/arm64`)
- [ ] Update Tekton pipeline with multi-arch kaniko build
- [ ] Test montage-ai on ARM64 nodes (raspillm8850, pi-worker-1, codeaijetson-desktop)
- [ ] Remove AMD64-only nodeSelector from deployment
- [ ] Verify performance on ARM vs AMD64

**Why:** Cluster has 4 ARM64 nodes (58% of total capacity) sitting idle. Multi-arch enables full cluster utilization.

**Implementation Notes:**
- Use buildx with `--platform linux/amd64,linux/arm64`
- May need ARM64-specific optimizations for librosa/numba
- Consider separate images for CPU-intensive vs web workloads

---

### Phase 3: Monitoring & Observability 📊

**Goal:** Full Grafana dashboard integration with custom metrics.

**Priority:** MEDIUM  
**ETA:** Week 3 (Jan 19, 2026)

**Tasks:**
- [ ] Create Grafana dashboard ConfigMap (similar to fluxibri_core pattern)
- [ ] Add custom Prometheus metrics:
  - Processing time per video (histogram)
  - Queue depth (gauge)
  - Active render jobs (gauge)
  - Error rate by phase (counter)
  - Resource usage (CPU/Memory/GPU) per job
- [ ] Set up AlertManager rules:
  - Job failure threshold
  - High memory usage
  - Disk space warnings
- [ ] Document metrics in README with example queries

**Metrics to Expose:**
```python
from prometheus_client import Counter, Histogram, Gauge

job_duration = Histogram('montage_job_duration_seconds', 'Time to complete montage job')
active_jobs = Gauge('montage_active_jobs', 'Number of active rendering jobs')
job_errors = Counter('montage_job_errors_total', 'Total job failures', ['phase'])
```

---

### Phase 4: Storage Optimization 💾

**Goal:** Efficient storage for distributed workloads.

**Priority:** MEDIUM  
**ETA:** Week 4 (Jan 26, 2026)

**Tasks:**
- [ ] Benchmark local-path vs NFS for video workloads
- [ ] Implement storage tiering:
  - Hot: Active jobs (local SSD)
  - Warm: Recent outputs (NFS)
  - Cold: Archive (MinIO/S3)
- [ ] Add cleanup policies:
  - Auto-delete temp files after 24h
  - Move completed jobs to warm storage after 7d
  - Archive to cold storage after 30d
- [ ] Consider MinIO for object storage (S3-compatible)
- [ ] Add PVC monitoring (usage alerts at 80%)

**Current Storage Usage:**
```
/data/input  → 50Gi  (local-path, WaitForFirstConsumer)
/data/output → 100Gi (local-path, WaitForFirstConsumer)
/data/music  → 10Gi  (local-path, WaitForFirstConsumer)
```

**Proposed Tiering:**
```
Hot:  local-path on NVMe nodes (codeai-fluxibriserver)
Warm: NFS on shared storage (192.168.1.x:/exports/montage)
Cold: MinIO bucket (s3://montage-ai-archive/)
```

---

### Phase 5: Auto-Scaling 📈

**Goal:** Scale pods based on workload (queue depth, CPU, memory).

**Priority:** LOW  
**ETA:** Week 5-6 (Feb 2-9, 2026)

**Tasks:**
- [ ] Implement HPA (Horizontal Pod Autoscaler):
  ```yaml
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
  ```
- [ ] Add KEDA for queue-based scaling (if job queue implemented)
- [ ] Test scaling behavior under load
- [ ] Document scaling policies and tune thresholds
- [ ] Consider GPU-aware scheduling (if GPU support added)

**Scaling Strategy:**
- Scale up: Queue depth > 3 OR CPU > 70%
- Scale down: Idle for 5min AND queue depth = 0
- Max replicas: 5 (limited by PVC RWO constraint)

**Note:** May need ReadWriteMany (RWX) PVCs for multi-pod scaling. Consider NFS or CephFS.

---

### Phase 6: Security Hardening 🔒

**Goal:** Move from `privileged` PodSecurity to `restricted`.

**Priority:** LOW  
**ETA:** Week 7-8 (Feb 16-23, 2026)

**Tasks:**
- [ ] Audit container capabilities (run as non-root?)
- [ ] Remove unnecessary privileges (allowPrivilegeEscalation: false)
- [ ] Implement NetworkPolicies:
  - Deny all ingress by default
  - Allow only from Ingress Controller
  - Allow egress to cluster DNS and external APIs
- [ ] Add PodSecurityPolicy migration path
- [ ] Run security scans (trivy, grype) in CI
- [ ] Document security best practices

**Current Issues:**
- Namespace requires `pod-security.kubernetes.io/enforce=privileged`
- Container may need root for FFmpeg GPU access
- No NetworkPolicies defined

---

## 📊 Implementation Timeline

| Phase | Priority | ETA | Status |
|-------|----------|-----|--------|
| **Phase 1: Service Stability** | 🔴 HIGH | Week 1 | ✅ **COMPLETE** |
| **Phase 2: Multi-Arch** | 🔴 HIGH | Week 2 | 🔄 In Progress |
| **Phase 3: Monitoring** | 🟡 MEDIUM | Week 3 | 📋 Planned |
| **Phase 4: Storage** | 🟡 MEDIUM | Week 4 | 📋 Planned |
| **Phase 5: Auto-Scaling** | 🟢 LOW | Week 5-6 | 📋 Planned |
| **Phase 6: Security** | 🟢 LOW | Week 7-8 | 📋 Planned |

---

## 🔍 Known Limitations

1. **AMD64 Only:** Cannot utilize 4 ARM64 nodes (raspillm8850, pi-worker-1, codeaijetson-desktop, codeai-thinkpad-t14s-gen-6)
2. **No Grafana Dashboard:** ServiceMonitor configured but no visualization
3. **Single Replica:** No horizontal scaling (PVC is ReadWriteOnce)
4. **Privileged Namespace:** Security can be hardened
5. **HTTP Only:** No TLS termination (consider Let's Encrypt later)
6. **No GPU Support:** Not using AMD RX 7900 XTX or Jetson GPU

---

## 🎯 Success Metrics

**Phase 1 (Complete):**
- ✅ Pod status: Running (not Completed/CrashLoopBackOff)
- ✅ Readiness probe: Passing
- ✅ Ingress: HTTP 200 on http://montage-ai.fluxibri.lan
- ✅ Health endpoint: Returns JSON with status=healthy

**Phase 2 (Target):**
- [ ] ARM64 image builds successfully
- [ ] ARM64 pod starts without errors
- [ ] Performance within 20% of AMD64

**Phase 3 (Target):**
- [ ] Grafana dashboard shows real-time metrics
- [ ] Prometheus scraping successful
- [ ] At least 5 custom metrics exposed

---

## 📝 Deployment Checklist (for Future Updates)

1. **Code Changes:**
   - [ ] Update code
   - [ ] Run tests: `make test`
   - [ ] Update CHANGELOG.md

2. **Build & Push:**
   - [ ] Build image: `docker buildx build --platform linux/amd64 -t 192.168.1.12:5000/montage-ai:latest --load .`
   - [ ] Push to registry: `docker push 192.168.1.12:5000/montage-ai:latest`

3. **Deploy:**
   - [ ] Apply manifests: `kubectl apply -k deploy/k3s/app/`
   - [ ] Restart deployment: `kubectl rollout restart deployment montage-ai-web -n montage-ai`
   - [ ] Verify pods: `kubectl get pods -n montage-ai`
   - [ ] Check logs: `kubectl logs -f -n montage-ai -l app=montage-ai`

4. **Verify:**
   - [ ] Health check: `curl http://montage-ai.fluxibri.lan/health`
   - [ ] Web UI accessible: http://montage-ai.fluxibri.lan
   - [ ] Metrics endpoint: `curl http://montage-ai-web:8080/metrics`

---

## 🔗 Quick Links

- **Web UI:** http://montage-ai.fluxibri.lan
- **Health Check:** http://montage-ai.fluxibri.lan/health
- **Grafana:** http://grafana.fluxibri.lan (dashboard pending Phase 3)
- **Prometheus:** http://prometheus.fluxibri.lan (metrics already collected)
- **GitHub Repo:** https://github.com/mfahsold/montage-ai
- **Docker Registry:** 192.168.1.12:5000/montage-ai:latest

---

## 💡 Next Actions

**Immediate (This Week):**
1. Commit and push changes to GitHub
2. Update main README.md with deployment status
3. Start Phase 2: Multi-arch build setup

**Short-term (Next 2 Weeks):**
1. Complete Phase 2 (Multi-Arch)
2. Begin Phase 3 (Monitoring dashboard)

**Long-term (1-2 Months):**
1. Storage optimization
2. Auto-scaling
3. Security hardening
