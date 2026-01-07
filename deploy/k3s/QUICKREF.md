# Montage AI on K3s - Quick Reference

## Build & Deploy

```bash
# Full deployment
cd /home/codeai/montage-ai
./deploy/k3s/build-and-push.sh  # ~5-10 min
./deploy/k3s/deploy.sh          # ~2 min
```

## Access

```bash
# Option 1: Port forward (quick)
kubectl port-forward -n montage-ai svc/montage-ai 5000:5000
# Open http://localhost:5000

# Option 2: Ingress (permanent)
# Add to /etc/hosts: 192.168.1.12  montage.local
# Open https://montage.local
```

## Common Operations

```bash
# View logs
kubectl logs -n montage-ai -l app=montage-ai -f

# Restart pod
kubectl rollout restart deployment/montage-ai -n montage-ai

# Scale down (stop)
kubectl scale deployment montage-ai -n montage-ai --replicas=0

# Scale up (start)
kubectl scale deployment montage-ai -n montage-ai --replicas=1

# Get shell access
kubectl exec -it -n montage-ai deployment/montage-ai -- /bin/bash

# Check Exo connection
kubectl exec -it -n montage-ai deployment/montage-ai -- curl http://exo-api.default.svc.cluster.local:8000/v1/models

# Update code (without rebuild)
kubectl rollout restart deployment/montage-ai -n montage-ai
# (code mounted from hostPath)

# Full rebuild & update
./deploy/k3s/build-and-push.sh
kubectl rollout restart deployment/montage-ai -n montage-ai
```

## Status Check

```bash
kubectl get all -n montage-ai
kubectl describe pod -n montage-ai -l app=montage-ai
```

## Troubleshooting

```bash
# Check pod status
kubectl get pods -n montage-ai

# View events
kubectl get events -n montage-ai --sort-by='.lastTimestamp'

# Check PVC
kubectl get pvc -n montage-ai

# Test Exo API
kubectl port-forward -n default svc/exo-api 8000:8000
curl http://localhost:8000/v1/models
```

## Integration Points

- **LLM**: `http://exo-api.default.svc.cluster.local:8000`
- **Registry**: `192.168.1.12:5000`
- **Storage**: `local-path` PVC (100Gi)
- **Source**: `/home/codeai/montage-ai/src` (hostPath)
