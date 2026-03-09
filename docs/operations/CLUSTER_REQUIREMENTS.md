# Cluster Infrastructure Requirements

**Version:** 1.0  
**Date:** 2026-03-09  
**Purpose:** Technical requirements for production-ready Montage AI clusters

---

## Executive Summary

Montage AI is a CPU-, GPU-, and I/O-intensive video processing application. It combines FFmpeg-based rendering, audio/video analysis, and segment-based disk writing. Storage and scheduling are as critical as raw compute performance.

---

## Muss-Anforderungen (Go-Live Critical)

### 1. Kubernetes Cluster Architecture

| Requirement | Specification |
|-------------|---------------|
| **Node Pools** | Minimum 4 pools: control, web, cpu-worker, gpu-worker |
| **Control Plane** | 3+ nodes, no user workloads |
| **Web/API Pool** | 2+ nodes, general workloads |
| **CPU Workers** | 2+ nodes, CPU-intensive encoding |
| **GPU Workers** | 1+ nodes with NVIDIA/Intel GPUs |

**Node Labels:**
```yaml
node-role.kubernetes.io/worker-cpu: "true"
node-role.kubernetes.io/worker-gpu: "true"
```

### 2. GPU Support

| Component | Requirement |
|-----------|-------------|
| **Device Plugin** | NVIDIA Device Plugin OR Intel GPU Plugin |
| **GPU Scheduling** | `nvidia.com/gpu` or `gpu.intel.com/i915` resources |
| **Drivers** | Latest stable (NVIDIA 545+, Intel 23.3+) |
| **Reproducibility** | GPU requests per Pod (e.g., `nvidia.com/gpu: 1`) |

### 3. Storage Requirements

| Tier | Type | Access Mode | Performance |
|------|------|-------------|-------------|
| **Persistent Data** | NFS/Longhorn/EFS | ReadWriteMany | >3000 IOPS, >200 MB/s write |
| **Ephemeral Scratch** | emptyDir | Pod-scoped | Local SSD preferred |
| **Cache** | Redis | In-memory | Sub-ms latency |

**PVCs Required:**
- `montage-ai-input` (footage, RWX)
- `montage-ai-output` (renders, RWX)
- `montage-ai-music` (audio library, RWX)
- `montage-ai-assets` (templates, RWX)

### 4. Storage Isolation

**Critical:** Ephemeral scratch and persistent output must be separated:

```yaml
volumes:
  - name: data-output
    persistentVolumeClaim:
      claimName: montage-ai-output  # Survives pod restart
  - name: tmp-segments
    emptyDir:
      sizeLimit: 20Gi               # Pod-scoped, auto-cleanup
```

### 5. Container Registry

| Requirement | Specification |
|-------------|---------------|
| **Accessibility** | Cluster-internal reachable |
| **Pull Secrets** | `imagePullSecrets` configured |
| **Version Pinning** | Explicit tags or digests (no `latest`) |
| **Multi-arch** | AMD64 and ARM64 images |

### 6. Secret Management

**Never use plain environment variables for secrets.**

Required approach:
- Kubernetes Secrets (encrypted at rest)
- External Secret Operators (Vault, AWS Secrets Manager)
- Sealed Secrets for GitOps

**Required Secrets:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: montage-ai-secrets
type: Opaque
stringData:
  LITELLM_API_KEY: "sk-..."
  CGPU_CREDENTIALS: '{"refresh_token": "..."}'
```

### 7. Observability

| Component | Tool | Metrics |
|-----------|------|---------|
| **Logs** | Loki / ELK / CloudWatch | All container logs |
| **Metrics** | Prometheus + Grafana | Pod/node/GPU metrics |
| **Tracing** | Jaeger (optional) | Request flows |

**Key Alerts:**
- Job hangs (>10 min no progress)
- GPU unavailable (allocatable = 0)
- Storage saturated (>85% PVC usage)
- Error rate spike (>5% failed jobs)

### 8. Job Scheduling

| Feature | Requirement |
|---------|-------------|
| **Retry Strategy** | Exponential backoff, max 3 retries |
| **Timeouts** | Job-level timeout (default: 1h) |
| **Resource Limits** | Per-job CPU/memory limits |
| **Preemption** | Low-priority jobs can be preempted |

### 9. Network Security

**NetworkPolicies Required:**
```yaml
# Default deny
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
```

**Explicit Allows:**
- Web UI → API (port 8080)
- Workers → Redis (port 6379)
- Workers → Storage (NFS ports)
- All → Registry (port 443)

**TLS:**
- Ingress TLS termination
- Optional: mTLS between services

### 10. Configuration Management

**Centralized via Environment/Settings:**
- `deploy/config.env` for deployment defaults
- `deploy/k3s/config-global.yaml` for cluster-specific values
- ConfigMaps for runtime configuration

**No hardcoded values in:**
- Deployment manifests
- Application code
- Scripts

---

## Soll-Anforderungen (Strongly Recommended)

### 11. Queue-Based Decoupling

**Message Queue for API/Worker separation:**
- Redis Queue or RabbitMQ
- Async job processing
- Queue depth monitoring

### 12. Horizontal Pod Autoscaling

| Workload | Scaling Metric | Range |
|----------|----------------|-------|
| Web | CPU utilization | 2-10 replicas |
| Workers | Queue length (KEDA) | 3-20 replicas |
| GPU Workers | GPU utilization | 1-5 replicas |

### 13. Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: montage-ai-quota
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 400Gi
    requests.nvidia.com/gpu: 4
```

### 14. Object Storage

**For archives and exports:**
- S3-compatible storage (MinIO, AWS S3, GCS)
- Lifecycle rules (transition to cold storage)
- Signed URLs for secure access

### 15. Rolling Updates

- Zero-downtime for Web/API components
- Rolling update strategy for workers
- Canary deployments (optional)

### 16. Node Affinity

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: node-role.kubernetes.io/worker-gpu
              operator: Exists
```

---

## Kann-Anforderungen (Optimization / MoE-Ready)

### 17. Specialized Worker Types

Separate workers for:
- **Analysis** (scene detection, audio analysis)
- **Transcription** (speech-to-text)
- **Rendering** (FFmpeg encoding)
- **Upscaling** (AI super-resolution)

### 18. Spot/Preemptible Instances

For non-critical batch jobs:
- Spot instance pools
- Checkpoint/resume capability
- 70-90% cost savings

### 19. Multi-Cluster / Burst

- Primary cluster + burst clusters
- Federation or multi-cluster ingress
- Disaster recovery

### 20. Performance Benchmarks

Standardized benchmarks per build:
- Render time per minute of output
- Throughput (jobs/hour)
- Cost per minute of output

---

## Abnahmekriterien (Acceptance Criteria)

### Before Production Activation

| Test | Criteria |
|------|----------|
| **Load Test** | 10+ parallel render jobs, p95 < 30s |
| **Resilience** | Pod/node failures auto-recover without data loss |
| **Recovery** | Output data recovery < 5 min, queue state recoverable |
| **Observability** | All alerts tested and firing correctly |

### Sign-Off Required

| Role | Responsibility |
|------|----------------|
| Deploy Engineer | Infrastructure deployment |
| QA Lead | Load and resilience testing |
| SRE/Ops | Monitoring and alerting |
| Product Owner | Business sign-off |

Operational sign-off and dated evidence are tracked in:
- [MOE_GO_LIVE_SIGNOFF.md](MOE_GO_LIVE_SIGNOFF.md)

---

## Quick Reference

### Verification Commands

```bash
# Node pools
kubectl get nodes --label-columns=node-role.kubernetes.io/worker-cpu,node-role.kubernetes.io/worker-gpu

# GPU availability
kubectl get nodes -o yaml | grep "nvidia.com/gpu"

# Storage
kubectl get storageclass
kubectl get pvc -n montage-ai

# Secrets
kubectl get secrets -n montage-ai

# Network policies
kubectl get networkpolicies -n montage-ai
```

### Resource Requirements Summary

| Component | CPU | Memory | GPU | Storage |
|-----------|-----|--------|-----|---------|
| Web UI | 250m-1 | 512Mi-1Gi | - | - |
| CPU Worker | 1-16 | 2Gi-24Gi | - | 20Gi ephemeral |
| GPU Worker | 4-16 | 8Gi-32Gi | 1 | 50Gi ephemeral |
| Redis | 500m-2 | 1Gi-4Gi | - | Persistent |

---

**Last Updated:** 2026-03-09  
**Next Review:** 2026-04-09  
**Owner:** Infrastructure Team
