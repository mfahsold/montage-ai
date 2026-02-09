# Rollback & Recovery Guide

How to recover from failed deployments, revert to a previous version, and safely re-deploy.

---

## Idempotency Guarantees

`deploy.sh` and `make deploy-cluster` are designed to be **safe to run multiple times**:

| Resource | First Deploy | Re-deploy | Notes |
|----------|-------------|-----------|-------|
| Namespace | Created | Skipped (exists) | Labels updated |
| PVCs | Created | **Skipped** (immutable) | Data preserved |
| ConfigMaps | Created | Updated | New config values applied |
| Deployments | Created | Rolling update | Zero-downtime |
| Services | Created | No change | Stable endpoints |
| Ingress | Created | Updated | Hostname from config |

**Key:** PVCs are never deleted or modified during re-deploy. This protects your data.

### Safe Re-deploy

```bash
# Always safe to run:
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster
# → Existing PVCs preserved, Deployments updated
```

### After a Failed Image Push

```bash
# Fix the image, then re-push and re-deploy:
cd deploy/k3s
./build-and-push.sh
./deploy.sh cluster
# → Pods will pull the new image
```

### PVC Lifecycle

PVCs are **immutable** after creation (storageClassName, accessModes, size cannot change).

```bash
# Check existing PVCs
kubectl get pvc -n "${CLUSTER_NAMESPACE:-montage-ai}"

# If PVC names changed in config, you must either:
# 1. Update config-global.yaml storage.pvc.* to match existing names
# 2. Or delete old PVCs first (WARNING: data loss!)
kubectl delete pvc <old-name> -n "${CLUSTER_NAMESPACE:-montage-ai}"
```

### Recovering from Interrupted PVC Creation

If deploy was interrupted mid-PVC-creation:

```bash
# Check for stuck PVCs
kubectl get pvc -n "${CLUSTER_NAMESPACE:-montage-ai}" | grep -v Bound

# Delete stuck (Pending) PVCs and re-deploy
kubectl delete pvc <stuck-pvc> -n "${CLUSTER_NAMESPACE:-montage-ai}"
make -C deploy/k3s deploy-cluster
```

### Scaling Workers Safely

```bash
NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"

# Scale up
kubectl scale deployment montage-ai-worker -n "$NAMESPACE" --replicas=4

# Scale down (graceful — waits for in-progress jobs)
kubectl scale deployment montage-ai-worker -n "$NAMESPACE" --replicas=1

# Scale to zero (stop all workers)
kubectl scale deployment montage-ai-worker -n "$NAMESPACE" --replicas=0
```

---

## Kubernetes Rollback

### Quick Rollback (Previous Version)

```bash
NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"

# Check rollout history
kubectl rollout history deployment/montage-ai-web -n "$NAMESPACE"

# Rollback to previous version
kubectl rollout undo deployment/montage-ai-web -n "$NAMESPACE"

# Rollback workers too
kubectl rollout undo deployment/montage-ai-worker -n "$NAMESPACE"

# Verify pods are running
kubectl get pods -n "$NAMESPACE"
```

### Rollback to Specific Revision

```bash
# List revisions with details
kubectl rollout history deployment/montage-ai-web -n "$NAMESPACE"

# Rollback to specific revision
kubectl rollout undo deployment/montage-ai-web --to-revision=3 -n "$NAMESPACE"
```

### Full Rollback (Including Config)

If ConfigMaps also need to be reverted:

```bash
# 1. Checkout the previous known-good commit
git checkout <previous-tag-or-commit>

# 2. Re-render config from that version
make -C deploy/k3s config

# 3. Re-deploy
make -C deploy/k3s deploy-cluster

# 4. Verify
kubectl get pods -n "$NAMESPACE"
kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/name=montage-ai --tail=20
```

---

## Docker Rollback

### Rollback to Previous Image

```bash
# List available images
docker images montage-ai

# Re-tag previous image as latest
docker tag montage-ai:<previous-tag> montage-ai:latest

# Restart
docker compose down
docker compose up
```

### Rebuild from Previous Commit

```bash
# Checkout previous version
git checkout <previous-commit>

# Rebuild
docker compose build
docker compose up
```

### Clean Rebuild (Nuclear Option)

```bash
docker compose down
docker rmi montage-ai:latest
docker compose build --no-cache
docker compose up
```

---

## Emergency Recovery

**All pods crashing:**
```bash
kubectl rollout undo deployment/montage-ai-web -n "$NAMESPACE"
kubectl rollout undo deployment/montage-ai-worker -n "$NAMESPACE"
```

**ConfigMap not found errors:**
Pods referencing a deleted ConfigMap will fail. Re-render and re-deploy:
```bash
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster
```

**PVC issues (data corruption):**
```bash
# Check PVC status
kubectl get pvc -n "$NAMESPACE"

# If a PVC is stuck in Pending, check events:
kubectl describe pvc <name> -n "$NAMESPACE"
```

> **Warning:** Deleting a PVC deletes the underlying data. Only do this as a last resort.

---

## See Also

- [Cluster Deployment](../cluster-deploy.md) — Deploy guide
- [Troubleshooting](../troubleshooting.md) — Common issues
- [deploy/k3s/README.md](../../deploy/k3s/README.md) — Full K8s reference
