# Cluster Rollout Checklist (MoE-Ready)

**Version:** 1.0  
**Date:** 2026-03-09  
**Purpose:** Pre-deployment validation for Montage AI clusters with MoE support

**Canonical sign-off artifact:** [MOE_GO_LIVE_SIGNOFF.md](MOE_GO_LIVE_SIGNOFF.md)

---

## **Pre-Flight Gates** ✅

Before any deployment, verify these hard gates:

- [ ] `kubectl`, `kustomize`, `make` available and functional
- [ ] Cluster reachable: `kubectl cluster-info`
- [ ] All placeholders in `deploy/k3s/config-global.yaml` replaced (no `<PLACEHOLDER>` strings)
- [ ] Registry/image accessible from all nodes
- [ ] Image tag explicitly versioned (not `latest`)
- [ ] Architecture match: Nodes match config (amd64/arm64)
- [ ] **Node pools separated:** control, web, cpu-worker, gpu-worker pools exist
- [ ] **GPU device plugin installed** (NVIDIA/Intel) and GPUs schedulable
- [ ] **RWX StorageClass** available with sufficient IOPS (>3000)
- [ ] **Registry pull secrets** configured for private registries
- [ ] **Secrets created** (not in git): litellm-auth, cgpu-credentials
- [ ] **NetworkPolicies** applied (default deny + explicit allows)
- [ ] **Observability stack** ready: Prometheus, Grafana, central logging

**Quick Check:**
```bash
cd deploy/k3s
make pre-flight

# Verify node pools
kubectl get nodes --label-columns=node-role.kubernetes.io/worker-cpu,node-role.kubernetes.io/worker-gpu

# Verify GPU support
kubectl get nodes -o yaml | grep -A5 "nvidia.com/gpu"

# Verify storage
kubectl get storageclass
kubectl get pvc -n <namespace>
```

---

## **Storage Requirements** 💾

### Multi-Node Clusters
- [ ] RWX-capable StorageClass available (NFS/Longhorn/Ceph/EFS)
- [ ] PVC strategy defined and documented
- [ ] Storage backend health verified: `kubectl get sc,pvc -A`

If existing cluster PVCs are still `RWO`, execute:
- [RWX_PVC_MIGRATION_RUNBOOK.md](RWX_PVC_MIGRATION_RUNBOOK.md)

### Bootstrap Verification
- [ ] `.ready` marker file created after first deployment
- [ ] Data directories initialized:
  - `/data/input/`
  - `/data/output/`
  - `/data/music/`
  - `/tmp/montage_*`

**Verification:**
```bash
kubectl exec -n <namespace> deploy/montage-ai -- ls -la /data/
```

---

## **Configuration Management** ⚙️

- [ ] `deploy/config.env` complete with all required values
- [ ] `deploy/k3s/config-global.yaml` rendered: `make config`
- [ ] No hardcoded values in manifests (all via config)
- [ ] Resource limits configured (CPU/memory per node type)
- [ ] Autoscaling parameters set (min/max replicas)
- [ ] Cache and LLM/CGPU endpoints configured

**Critical Environment Variables:**
```bash
# Storage
STORAGE_CLASS=<rwx-class>
PVC_SIZE=100Gi

# Resources
WEB_REPLICAS=2
WORKER_REPLICAS=3
MAX_WORKERS_PER_NODE=4

# External Services
OLLAMA_HOST=http://ollama:11434
FFMPEG_MCP_ENDPOINT=http://ffmpeg-mcp:8080
CGPU_ENABLED=false

# MoE Settings (NEW)
MOE_ENABLED=true
MOE_HUMAN_REVIEW=true
MOE_AUTO_APPLY=true
MOE_AUTO_THRESHOLD=0.8
MOE_TIMEOUT=30.0
MOE_FALLBACK=true
```

---

## **MoE-Specific Requirements** 🧠

### Production Readiness
- [ ] MoE integrated into real pipeline (not just CLI mock)
- [ ] Real media context passed to experts (beats, scenes, audio)
- [ ] Human decision audit logging configured
- [ ] Expert weights externally configurable via env vars:
  - `MOE_WEIGHT_RHYTHM=1.2`
  - `MOE_WEIGHT_NARRATIVE=1.0`
  - `MOE_WEIGHT_AUDIO=1.1`

### Timeout Enforcement
- [ ] `MOE_TIMEOUT` enforced in code (not just config)
- [ ] Expert execution cancels after timeout
- [ ] Partial results handled gracefully

### Fallback Strategy
- [ ] Fallback path tested: MoE error → classic pipeline
- [ ] `MOE_FALLBACK=true` verified working
- [ ] No data loss during fallback

### Conflict Resolution
- [ ] Human review queue accessible (Web UI or CLI)
- [ ] Conflict notification mechanism active
- [ ] `max_conflicts_before_human` triggers correctly

---

## **Standard Rollout Sequence** 🚀

Execute in this exact order:

```bash
cd deploy/k3s

# 1. Render config
make config

# 2. Pre-flight checks
make pre-flight

# 3. Validate manifests
make validate

# 4. Build and push (multi-arch if needed)
make build push

# 5. Deploy
make deploy-cluster

# 6. Bootstrap
./bootstrap.sh

# 7. Verify
kubectl get all -n <namespace>
kubectl logs -n <namespace> deploy/montage-ai-web -f
```

---

## **Go-Live Verification** 🎯

### Basic Health
- [ ] All pods `Ready`: `kubectl get pods -n <namespace>`
- [ ] No PVC pending: `kubectl get pvc -n <namespace>`
- [ ] Services responding: `curl http://<service>/api/status`

### Performance SLOs
- [ ] Preview generation p50 < 8s
- [ ] Preview generation p95 < 30s
- [ ] Cache hit rate > 70%

**Test Command:**
```bash
# Run preview SLO tests
pytest tests/test_preview_slo.py -v
```

### MoE Metrics
- [ ] Expert execution count visible in logs
- [ ] Conflict detection working (test with conflicting styles)
- [ ] Auto-apply rate > 60% (for low-impact deltas)
- [ ] Human review queue accessible

**Verify MoE:**
```bash
# Check MoE status
kubectl exec -n <namespace> deploy/montage-ai -- \
  python -c "from montage_ai.moe import MoEControlPlane; \
  m = MoEControlPlane(); print(m.get_status())"
```

---

## **Rollback Readiness** ↩️

Before going live, ensure:

- [ ] `kubectl rollout history deploy/montage-ai-web` shows history
- [ ] Rollback tested: `kubectl rollout undo deploy/montage-ai-web`
- [ ] Last known-good image tag documented
- [ ] Database/config backups current

### Rollback Triggers (Auto-rollback recommended)
- SLO violation (p95 > 60s for 5 minutes)
- MoE conflict rate > 80% (indicates expert misconfiguration)
- Worker crashloop > 3 restarts in 10 minutes
- Storage PVC failures

**Emergency Rollback:**
```bash
# Quick rollback to previous version
kubectl rollout undo deploy/montage-ai-web -n <namespace>
kubectl rollout undo deploy/montage-ai-worker -n <namespace>
```

---

## **Post-Deployment Monitoring** 📊

Set up within 24h of deployment:

- [ ] Prometheus metrics scraping configured
- [ ] Grafana dashboards imported:
  - Preview generation latency
  - MoE expert execution times
  - Conflict resolution rates
  - Storage utilization
- [ ] Alertmanager rules active:
  - High error rate (> 5%)
  - High queue depth (> 100 jobs)
  - Storage > 80% full
  - MoE timeout rate > 10%

---

## **Go/No-Go Criteria** 🚦

**Minimum for GO:**
1. ✅ `make pre-flight` passes
2. ✅ RWX storage healthy (no pending PVCs)
3. ✅ Bootstrap completed (`.ready` marker exists)
4. ✅ MoE real-input pipeline active (not mock)
5. ✅ Preview SLOs pass (p95 < 30s)

**Auto NO-GO:**
- Any hard gate fails
- Storage class not RWX-capable on multi-node
- MoE integration only mock (no real media context)
- Rollback mechanism untested

---

## **Sign-Off** ✍️

Record dated approvals and evidence in [MOE_GO_LIVE_SIGNOFF.md](MOE_GO_LIVE_SIGNOFF.md).

---

## **Quick Reference Commands**

```bash
# Full status check
kubectl get all,pvc,ingress -n <namespace>

# Check MoE specifically
kubectl exec -n <namespace> deploy/montage-ai -- \
  montage-ai moe --style documentary --show-deltas

# View recent logs
kubectl logs -n <namespace> --selector=app=montage-ai --tail=100 -f

# Scale workers
kubectl scale deploy/montage-ai-worker -n <namespace> --replicas=5

# Emergency stop (keep data)
kubectl scale deploy/montage-ai-web deploy/montage-ai-worker \
  -n <namespace> --replicas=0
```

---

## **Go-Live Abnahmekriterien** ✅

Before production activation, verify:

### Load Testing
- [ ] Parallel render jobs (10+) complete successfully
- [ ] Preview generation p95 < 30s under load
- [ ] No OOM kills during 1h stress test
- [ ] Storage I/O not saturated (<80% bandwidth)

### Resilience Testing
- [ ] Pod restart: Jobs resume without data loss
- [ ] Node failure: Workload rescheduled within 60s
- [ ] GPU node failure: Fallback to CPU encoding works
- [ ] Storage failover: PVCs remain accessible

### Recovery Testing
- [ ] Output data recovery time < 5 minutes
- [ ] Queue state recovery (Redis persistence)
- [ ] Rollback tested: `kubectl rollout undo`

### Observability Verification
- [ ] Dashboards accessible: Grafana, Kibana/Loki
- [ ] Alerts configured and tested:
  - "Job hängt" (>10 min no progress)
  - "GPU unavailable" (nvidia.com/gpu = 0 allocatable)
  - "Storage saturated" (>85% PVC usage)
  - "Error rate spike" (>5% failed jobs)

**Sign-Off Required:**
Use [MOE_GO_LIVE_SIGNOFF.md](MOE_GO_LIVE_SIGNOFF.md) as the single approval table (owner, status, evidence, date).

---

**Last Updated:** 2026-03-09  
**Next Review:** 2026-04-09
