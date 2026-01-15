# Network Robustness Analysis: WiFi-Connected GPU Nodes

## Overview

This document analyzes the networking challenges encountered when running high-performance GPU workloads (Montage AI) on cluster nodes connected via WiFi, specifically focusing on the `codeai-fluxibriserver` node.

## Identified Issues

### 1. VXLAN Overlay Instability

K3s uses Flannel with VXLAN by default. When the underlying transport is WiFi:

- **High Latency**: Packet encapsulation and WiFi medium contention lead to 100-900ms latency.
- **TCP Timeouts**: The high latency and jitter cause TCP connection establishments (3-way handshake) to time out, particularly for Registry pulls and ClusterIP traffic.
- **Dial Errors**: Pods fail to pull images with `DeadlineExceeded` or `i/o timeout`.

### 2. DNS and ClusterIP Connectivity

Internal service discovery (CoreDNS) and traffic routing (kube-proxy/nftables) are sensitive to the MTU and latency of the overlay network. WiFi nodes often lose connectivity to services like the Docker Registry or the Exo API.

## Implementation: WiFi-Node Survivability Mode

To ensure Montage AI remains operational on these nodes, we have implemented the following architectural changes:

### A. Image Pull Strategy (`IfNotPresent`)

All manifests in [deploy/k3s/](../../../deploy/k3s/) now use `imagePullPolicy: IfNotPresent`. This prevents the Kubelet from attempting to pull the image over the unstable overlay if it exists locally.

### B. Manual Image Sideloading

We have introduced a sideloading pipeline for environments where the registry is unreachable:

1. **Pull** the image on a wired worker (e.g., `codeai-worker-amd64`).
2. **Stream** the image via SSH directly to the target node's K3s storage.
3. **Import** using the K3s-specific containerd socket:

   ```bash
   ssh node "sudo ctr --address /run/k3s/containerd/containerd.sock -n k8s.io images import -"
   ```

### C. Manual Pre-load Script

A utility script is available at [scripts/ops/preload-image-wifi-node.sh](../../../scripts/ops/preload-image-wifi-node.sh).

## Future Recommendations

1. **Switch to host-gw**: If nodes are on the same Layer 2 network, switching Flannel to `host-gw` instead of `vxlan` will remove encapsulation overhead.
2. **Local Registry Mirror**: Deploy a pull-through cache (Registry Mirror) directly on the WiFi node using `hostNetwork: true`.
3. **Ethernet Backhaul**: Hardware connection is always preferred for GPU nodes to ensure consistent data throughput.
