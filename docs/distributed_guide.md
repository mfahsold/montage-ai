# Distributed Rendering Guide

This guide explains how to set up and run Montage AI in a distributed Kubernetes environment (e.g., K3s cluster with mixed AMD64/ARM64 nodes).

## Architecture

The distributed setup leverages:
- **Kubernetes Jobs**: For orchestrating the rendering process.
- **Shared NFS Storage**: Mounted at `/data/output` for sharing assets and caching analysis results.
- **Metadata Cache**: A file-based cache stored on the shared volume to prevent redundant scene analysis across nodes.
- **Multi-Arch Docker Images**: Built for both `linux/amd64` and `linux/arm64`.

## Prerequisites

- A Kubernetes cluster (K3s recommended).
- `kubectl` configured to talk to your cluster.
- Shared storage (PVC) available in the cluster.

## Setup

We provide an idempotent setup script to prepare your cluster:

```bash
./scripts/setup_cluster.sh
```

This script will:
1. Create the `montage-ai` namespace.
2. Set up image pull secrets (from your local Docker config).
3. Set up CGPU secrets (if available).
4. Apply the base Kubernetes configurations.

## Building Images

To build the multi-architecture images with optimized caching:

```bash
make build-multiarch
```

This command uses Docker Buildx with registry caching (`type=registry`) to speed up subsequent builds.

## Running a Distributed Job

To run a distributed trailer generation job:

```bash
kubectl apply -f deploy/k3s/job-distributed-trailer.yaml
```

### Caching Configuration

The job is configured to use a shared cache for scene analysis metadata. This significantly speeds up processing on slower nodes (like Jetson ARM64 devices) by reusing analysis performed by faster nodes.

- **Env Var**: `METADATA_CACHE_DIR`
- **Default Path**: `/data/output/cache` (on the shared volume)

## Troubleshooting

- **Pods stuck in ContainerCreating**: Check `kubectl describe pod <pod-name>` for image pull errors or PVC mounting issues.
- **Slow Performance**: Ensure the `METADATA_CACHE_DIR` is correctly mounted and writable. Check logs to see if "Loading scene data from cache" messages appear.
