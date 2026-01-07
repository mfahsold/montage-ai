# Montage-AI + K3s Cluster Integration

## Status

✅ **K3s Cluster**: 7 nodes ready (AMD GPU, Jetson, Raspberry Pi, ThinkPad, etc.)  
✅ **Registry**: 192.168.1.12:5000 (contains montage-ai:latest + many other images)  
✅ **Kustomize Overlays**: dev/staging/prod configured  
⚠️ **Image Pull**: HTTP registry not yet configured for K3s  

## Quick Start

### 1. Configure K3s for Insecure HTTP Registry

The registry at `192.168.1.12:5000` is HTTP-only (not HTTPS). K3s needs configuration to allow this.

Add to `/etc/rancher/k3s/registries.yaml`:

```yaml
mirrors:
  "192.168.1.12:5000":
    endpoint:
      - "http://192.168.1.12:5000"
configs:
  "192.168.1.12:5000":
    tls:
      insecure_skip_verify: true
```

Then **restart K3s on control plane**:

```bash
# On control plane (raspillm8850)
sudo systemctl restart k3s

# On worker nodes, restart k3s-agent
sudo systemctl restart k3s-agent

# Wait for cluster to stabilize
kubectl get nodes  # All should be Ready
```

### 2. Deploy to Dev Environment

```bash
cd /home/codeai/montage-ai

# Deploy with dev overlay (local cluster)
./deploy/k3s/deploy.sh dev

# Watch pods
kubectl get pods -n montage-ai -w

# Check logs
kubectl logs -n montage-ai -l app.kubernetes.io/name=montage-ai -f
```

### 3. Access Web UI

```bash
# Port forward to local
kubectl port-forward -n montage-ai svc/montage-ai-web 5000:80

# Open browser to http://localhost:5000
```

## Configuration Matrix

| Environment | Registry | Resources | Preferences |
|---|---|---|---|
| dev | 192.168.1.12:5000 | 2Gi/500m requests | Any node (local-path OK) |
| staging | TBD | 8Gi/2 CPU requests | Prefer GPU nodes |
| prod | TBD | 16Gi/4 CPU requests | Require GPU nodes |

## Known Issues

### Issue 1: ImagePullBackOff
**Error**: `http: server gave HTTP response to HTTPS client`
**Fix**: Add K3s registry configuration (see above)

### Issue 2: PVC Pending with RWMany
**Error**: `local-path` storage class doesn't support ReadWriteMany
**Fix**: Use ReadWriteOnce (RWO) single-mount model (current implementation)

### Issue 3: Insufficient Memory
**Error**: Pod scheduling fails due to resource requests too high
**Fix**: Dev overlay uses smaller requests (2Gi/500m)

## Distributed Build Setup

For building multi-arch images locally:

```bash
# Use fluxibri_core's distributed build
cd ../fluxibri_core

# Build montage-ai image
make build-montage-ai-multiarch

# Push to local registry
make push-montage-ai-multiarch
```

## Next Steps

1. ✅ Configure K3s registry
2. ✅ Verify montage-ai pod runs
3. ✅ Test web UI access
4. Test video upload/processing
5. Setup Flux CD integration
6. Configure prod overlays with HA
