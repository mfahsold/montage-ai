# Infrastructure Team Feedback: SOTA Cluster-Agnostic Architecture

**Date**: 2026-01-29
**Context**: Montage AI Distributed Job Processing Refactor
**Status**: Recommendations for Cluster-Agnostic, DRY Deployment

---

## Executive Summary

This document provides SOTA (State-of-the-Art) recommendations for transitioning Montage AI to a cluster-agnostic, two-mode deployment architecture (Local vs Cluster). The recommendations are based on:

1. **Official Kubernetes Documentation** (kubernetes.io)
2. **CNCF Project Best Practices** (KEDA, Kueue, Volcano, JobSet)
3. **Recent Academic Research** (OSDI 2024, JNCA 2025, arXiv 2025)
4. **Industry Standards** for MLOps and AI workload orchestration

---

## 1. Recommended Architecture: Two Deployment Modes

### Mode 1: Local Development
```
DEPLOYMENT_MODE=local
```
- Docker Compose or direct CLI (`./montage-ai.sh web`)
- No Kubernetes dependencies
- Local file paths, no shared storage requirements
- Direct job execution (no queue)

### Mode 2: Cluster Deployment
```
DEPLOYMENT_MODE=cluster
```
- Single canonical Kustomize overlay: `overlays/cluster`
- API + Workers + Jobs via Kubernetes primitives
- Queue-based job distribution (Redis)
- Autoscaling via KEDA (ScaledObject/ScaledJob)

---

## 2. Kubernetes Primitives: SOTA Recommendations

### 2.1 Batch Workloads: Use Kubernetes Jobs + JobSet

**Current State**: Direct `batch/v1.Job` submission via `JobSubmitter`

**Recommended Enhancement**: Adopt [JobSet](https://jobset.sigs.k8s.io/) for distributed workloads

| Feature | Jobs (Current) | JobSet (Recommended) |
|---------|---------------|---------------------|
| Coordinated scheduling | No | Yes |
| Multi-role workloads | Manual | Native (leader/worker) |
| Topology-aware placement | Manual | Built-in |
| Failure handling | Per-job | Cohesive restart |

**Key Benefits**:
- Gang scheduling for distributed render tasks
- Automatic headless service for pod-to-pod communication
- Topology constraints to minimize network latency between workers

**Source**: [Introducing JobSet - Kubernetes Blog (March 2025)](https://kubernetes.io/blog/2025/03/23/introducing-jobset/)

### 2.2 Queue-Based Scaling: KEDA

**Current State**: HPA for web, manual worker scaling

**Recommended**:
```yaml
# ScaledObject for worker Deployments
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: montage-ai-worker
spec:
  scaleTargetRef:
    name: montage-ai-worker
  triggers:
    - type: redis-cluster
      metadata:
        address: redis:6379
        listName: default
        listLength: "5"
  minReplicaCount: 0  # Scale-to-zero when idle
  maxReplicaCount: 10
  cooldownPeriod: 300
```

**Best Practices** (from [keda.sh](https://keda.sh/)):
- Use `ScaledObject` for Deployments (workers)
- Use `ScaledJob` for ephemeral job pods
- Set `cooldownPeriod` to prevent thrashing (300s recommended)
- Choose queue-depth metrics over CPU for event-driven workloads

### 2.3 Advanced Batch Scheduling: Kueue

For complex multi-tenant or quota-constrained environments, consider [Kueue](https://kueue.sigs.k8s.io/):

| Feature | Description |
|---------|-------------|
| ClusterQueue | Define resource quotas per workload type |
| LocalQueue | Namespace-scoped job submission |
| MultiKueue | Cross-cluster job dispatching |
| Fair sharing | Weighted resource allocation |

**Use Case**: When multiple users/teams share GPU resources

---

## 3. Scheduling Policies: Cluster-Agnostic Design

### 3.1 Remove Hardcoded Node Selectors

**Anti-pattern** (current in some legacy overlays):
```yaml
nodeSelector:
  kubernetes.io/hostname: specific-node-name
```

**Recommended**:
```yaml
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
            - key: gpu.type
              operator: In
              values: ["nvidia", "amd"]
```

### 3.2 Use Taints/Tolerations for GPU Tiers

```yaml
# Node configuration (cluster admin)
taints:
  - key: "gpu"
    value: "required"
    effect: "NoSchedule"

# Pod configuration
tolerations:
  - key: "gpu"
    operator: "Equal"
    value: "required"
    effect: "NoSchedule"
```

**Source**: [Taints and Tolerations - Kubernetes Docs](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/)

### 3.3 Topology Spread Constraints

For high availability across nodes/zones:

```yaml
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
    labelSelector:
      matchLabels:
        app.kubernetes.io/name: montage-ai
```

**Source**: [Pod Topology Spread - Kubernetes Docs](https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/)

### 3.4 Pod Priority and Preemption

Define priority classes for critical vs background workloads:

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: montage-ai-critical
value: 1000000
globalDefault: false
description: "Critical render jobs"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: montage-ai-background
value: 100
globalDefault: false
description: "Background analysis tasks"
```

**Source**: [Pod Priority - Kubernetes Docs](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)

---

## 4. Configuration Portability

### 4.1 ConfigMaps for Non-Sensitive Config

**Current**: Mixed ConfigMap generators + inline env vars

**Recommended Structure**:
```
deploy/k3s/base/
├── cluster-config.env      # Cluster-specific (registry, storage class)
├── montage-ai-config       # App config (features, LLM endpoints)
└── kustomization.yaml      # ConfigMapGenerator references
```

### 4.2 Secrets for Sensitive Values

Move API keys from ConfigMaps to Secrets:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: montage-ai-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: "${OPENAI_API_KEY}"
  GOOGLE_API_KEY: "${GOOGLE_API_KEY}"
```

**Source**: [ConfigMaps and Secrets - Kubernetes Docs](https://kubernetes.io/docs/concepts/configuration/)

---

## 5. AI/ML Workload Considerations (SOTA Research)

### 5.1 ServerlessLLM Insights (OSDI 2024)

Key findings for AI workload scheduling:
- **Locality-aware placement**: 10-200x latency reduction
- **Warm pools**: Keep model checkpoints in memory
- **Fast model discovery**: Minimize cold-start overhead

**Recommendation**: For LLM inference endpoints, maintain warm inference pods rather than scale-to-zero.

### 5.2 Gwydion Framework (JNCA 2025)

Reinforcement learning-based autoscaling for microservices:
- Proactive scaling based on queue depth + request latency
- Better than reactive CPU-based scaling for bursty workloads

**Recommendation**: Use queue-depth metrics (KEDA) rather than CPU-only HPA.

### 5.3 GPU Scheduling Best Practices

| Strategy | Use Case |
|----------|----------|
| GPU sharing (MPS/MIG) | Multiple small models |
| Time-slicing | Burst inference |
| Exclusive allocation | Training workloads |

**Recommendation**: Use NVIDIA MIG for multi-tenant GPU sharing when available.

---

## 6. Current Issues in Montage AI Cluster

### 6.1 Stuck Render Job Analysis

```
Pod: montage-ai-render-4ccgg
Status: Pending (2d21h)
Events:
  - Insufficient cpu (2 nodes)
  - Insufficient memory (4 nodes)
  - Insufficient amd.com/gpu (7 nodes)
  - Untolerated taint(s) (2 nodes)
```

**Root Cause**: Job requests GPU resources that aren't available cluster-wide.

**Recommendations**:
1. Add fallback to CPU-only rendering when GPU unavailable
2. Use `preferredDuringSchedulingIgnoredDuringExecution` for GPU affinity
3. Implement job timeout with automatic retry on different resource tier

### 6.2 Fragmented Mode Detection

Current state:
- `CLUSTER_MODE` (boolean) in config.py
- `MONTAGE_CLUSTER_MODE` (enum) in node_capabilities.py

**Recommendation**: Unify into single `DEPLOYMENT_MODE` env var:
```python
class DeploymentMode(Enum):
    LOCAL = "local"
    CLUSTER = "cluster"
```

---

## 7. Action Items for Infrastructure Team

### High Priority

1. **Define canonical cluster overlay path**
   - Use `overlays/cluster` as single production entry point
   - Archive legacy overlays to `overlays/legacy/`

2. **Implement queue-based autoscaling**
   - Deploy KEDA operator
   - Create ScaledObject for worker deployments
   - Configure Redis list length triggers

3. **Remove hardcoded scheduling constraints**
   - Audit all manifests for hostname-based nodeSelectors
   - Replace with label-based affinity rules

### Medium Priority

4. **Evaluate JobSet for distributed rendering**
   - Prototype coordinated multi-pod render jobs
   - Test topology-aware scheduling for network-intensive tasks

5. **Implement PriorityClasses**
   - Define critical/standard/background tiers
   - Configure PodDisruptionBudgets accordingly

6. **Migrate sensitive config to Secrets**
   - OPENAI_API_KEY
   - GOOGLE_API_KEY
   - Any cluster credentials

### Lower Priority

7. **Consider Kueue for multi-tenant scenarios**
   - If multiple teams share GPU resources
   - For quota enforcement and fair sharing

8. **Evaluate Volcano for gang scheduling**
   - If distributed training workloads are added
   - For strict all-or-nothing pod scheduling

---

## 8. References

### Official Documentation
- [Kubernetes Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [JobSet SIG](https://jobset.sigs.k8s.io/)
- [KEDA](https://keda.sh/)
- [Kueue](https://kueue.sigs.k8s.io/)
- [Volcano](https://volcano.sh/)
- [Taints/Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/)
- [TopologySpreadConstraints](https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/)
- [PodPriority](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)

### Academic Research
- **ServerlessLLM** (OSDI 2024): Locality-aware LLM serving
- **Gwydion** (JNCA 2025): RL-based microservice autoscaling
- **λScale** (arXiv 2025): Fast scaling for serverless LLM inference
- **JSSPP 2025**: Job Scheduling Strategies for Parallel Processing

### Industry Best Practices
- [Kubernetes AI/ML Best Practices - Komodor](https://komodor.com/blog/why-kubernetes-is-becoming-the-platform-of-choice-for-running-ai-mlops-workloads/)
- [MLOps on AKS - Microsoft](https://learn.microsoft.com/en-us/azure/aks/best-practices-ml-ops)
- [Batch Scheduling Comparison - InfraCloud](https://www.infracloud.io/blogs/batch-scheduling-on-kubernetes/)

---

## 9. Summary

The recommended architecture consolidates deployment into two clear modes:

| Aspect | Local Mode | Cluster Mode |
|--------|-----------|--------------|
| Entry point | `./montage-ai.sh` | `kubectl apply -k overlays/cluster` |
| Job execution | Direct | Queue (Redis) + Workers |
| Scaling | Manual | KEDA (queue-based) |
| Storage | Local paths | RWX PVCs (NFS) |
| GPU access | Auto-detect | Node labels + tolerations |

This approach maximizes:
- **Portability**: Works on any Kubernetes cluster
- **DRYness**: Single canonical overlay, no env-specific patches
- **Observability**: Unified metrics and logging
- **Scalability**: Event-driven autoscaling with scale-to-zero

---

*Generated by Montage AI development team, 2026-01-29*
