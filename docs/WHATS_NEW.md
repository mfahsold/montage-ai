# What's New (2026-01-15)

This update focuses on **scalability and extreme performance**, transitioning Montage AI from a local tool to a production-grade distributed video pipeline.

## Key achievements

- **High-Performance Infrastructure (3x Speedup)**
  - Optimized system startup from **6s to 1.8s** via aggressive lazy loading and module pruning.
  - Reduced representative render times from **9m to 2.2m** by implementing VAAPI hardware acceleration defaults and streamlining the `SegmentWriter` disk I/O.
  - SOTA VAAPI initialization with dynamic driver discovery for stable hardware encoding.

- **Canonical Kubernetes Orchestration**
  - Fully refactored `JobSubmitter` to use the official **Kubernetes Python API**, eliminating flaky subprocess calls.
  - Integrated **Fluxibri SOTA patterns**: Resource Tiers (`minimal` to `gpu`), logical labels, and affinity rules are now standard.
  - Robust Job monitoring with automatic retries and propagation delay handling.

- **Hybrid Distributed Sharding**
  - New worker logic in `distributed_scene_detection.py` supports **Parallel Sharding**:
    - **Mode A (File-based)**: Distributes multiple videos across cluster nodes.
    - **Mode B (Time-based)**: Splits single large videos into segments for parallel processing on multiple workers.
  - Seamless result aggregation back to the shared NFS store (`/data`).

- **GPU Affinity & Reliability**
- Implemented Pod Affinity to ensure encode jobs land on high-performance GPU nodes (e.g., `example.com/gpu-enabled: true`).
  - Resolved architecture blockers (`exec format error`) by standardizing on `latest-amd64` images and node selectors.

## How this helps contributors

- **Scale Out**: You can now process massive libraries across a multi-node cluster with one command.
- **Lower Costs**: Reduced CPU/Memory overhead means you can run many more workers in the same resource envelope.
- **Improved DevX**: No more `kubectl` parsing errors; use native Python objects to extend the cluster logic.

---

## What's New (2026-01-13)
