# Polish & Stabilization Plan (Post-Cluster Integration)

This document outlines the roadmap for polishing the Montage AI distributed pipeline and stabilizing the developer experience.

## 🎯 Phase 1: Observability & Health (Short Term)

### 1.1 Hardware Acceleration Diagnostics

- [ ] Add `./montage-ai.sh check-hw` to verify VAAPI/NVENC/QSV availability.
- [ ] Implement `montage_ai.utils.hwaccel_probe` to log driver versions and skip failing devices during startup.

### 1.2 Distributed Job Visibility

- [ ] Update Web UI to show "Worker Status" in the progress bar (e.g., "Node 2: Processing 00:05:00").
- [ ] Implement `JobSubmitter.get_pod_logs(job_id)` to surface worker errors directly in the Web UI.

### 1.3 Pre-flight Checks

- [ ] Validate NFS mount path and write permissions before starting a Cluster Job.
- [ ] Check registry connectivity and image presence early in the workflow.

## 🎨 Phase 2: Creative & AI Refinement (Mid Term)

### 2.1 Wes Anderson Style (Symmetry Detection)

- [ ] Implement visual center-of-gravity analysis in `SceneAnalyzer`.
- [ ] Filter clips for "Wes Anderson" style based on compositional symmetry.

### 2.2 Story Engine Node Scaling

- [ ] Parallelize the `B-Roll Planner` using the new `JobSubmitter` patterns.
- [ ] Implement semantic search sharding for massive asset libraries.

## 🛠️ Phase 3: Developer & Ops Experience (SOTA Alignment)

### 3.1 Environment Auto-Detection

- [ ] Automatically enable `CLUSTER_MODE` if `KUBERNETES_SERVICE_HOST` is detected.
- [ ] Default `REGISTRY` and `PVC_NAME` based on canonical Fluxibri namespace patterns if unspecified.

### 3.2 Chaos Testing & Recovery

- [ ] Implement "Retry Shard" logic in `AnalysisEngine` if a single worker pod fails (OOM/Node Drain).
- [ ] Validate multi-GPU node affinity with different vendor drivers (AMD vs. NVIDIA).

---

## 📅 Roadmap Summary

| Milestone | Target | Description |
| :--- | :--- | :--- |
| **V2.5-Alpha** | 2026-01-20 | Observability & Pre-flight checks |
| **V2.5-Beta** | 2026-01-30 | Wes Anderson Style & Story Sharding |
| **V2.5-Stable** | 2026-02-15 | Full Distributed Lifecycle & Chaos Recovery |
