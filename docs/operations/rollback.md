# Rollback Guide

How to recover from failed deployments or revert to a previous version.

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
