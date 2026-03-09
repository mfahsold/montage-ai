# RWX PVC Migration Runbook (Montage AI)

**Purpose:** Migrate existing Montage AI PVCs from `ReadWriteOnce` to `ReadWriteMany` without losing media data.

**Scope:**

- `montage-ai-input-nfs`
- `montage-ai-output-nfs`
- `montage-ai-music-nfs`
- `montage-ai-assets-nfs`

## Why This Is Required

Kubernetes PVC access mode is immutable after creation. If existing PVCs are `RWO`, updating manifests to `RWX` does not change runtime state. The PVCs must be recreated or migrated to new claims.

## Preconditions

- Cluster has an RWX-capable StorageClass (`nfs-client`, `nfs-exo`, Longhorn RWX, CephFS, EFS, ...).
- Full application backup window is approved.
- Deployment is currently healthy (`web`, `worker`, `redis` running).
- Rollback owner is assigned.

## Phase 1: Prepare

1. Capture current state:

```bash
NAMESPACE=montage-ai
kubectl get pvc -n "$NAMESPACE" -o custom-columns='NAME:.metadata.name,ACCESS:.spec.accessModes[*],SC:.spec.storageClassName'
kubectl get deploy -n "$NAMESPACE"
```

1. Scale write paths down to avoid in-flight file mutations:

```bash
kubectl scale deploy/montage-ai-worker -n "$NAMESPACE" --replicas=0
kubectl scale deploy/montage-ai-web -n "$NAMESPACE" --replicas=0
```

1. Keep Redis up (queue metadata) and confirm no jobs are running.

## Phase 2: Create Target RWX PVCs

Create new PVCs with `ReadWriteMany` and `nfs-client` (example names with `-rwx` suffix).

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: montage-ai-input-rwx
  namespace: montage-ai
spec:
  accessModes: ["ReadWriteMany"]
  storageClassName: nfs-client
  resources:
    requests:
      storage: 50Gi
```

Repeat for output/music/assets with matching sizes.

Apply and verify:

```bash
kubectl apply -f /tmp/montage-rwx-pvcs.yaml
kubectl get pvc -n "$NAMESPACE" -o custom-columns='NAME:.metadata.name,ACCESS:.spec.accessModes[*],STATUS:.status.phase'
```

## Phase 3: Copy Data (Old -> New)

Use a one-shot helper pod mounting both source and target PVCs.

```bash
kubectl apply -f /tmp/montage-pvc-copy-job.yaml
kubectl logs -n "$NAMESPACE" job/montage-pvc-copy -f
kubectl wait --for=condition=complete -n "$NAMESPACE" job/montage-pvc-copy --timeout=1800s
```

Validation commands in copy pod:

```bash
# sample checks
find /src/input | wc -l
find /dst/input | wc -l
du -sh /src/output /dst/output
```

## Phase 4: Cutover Configuration

Update `deploy/k3s/config-global.yaml`:

- `storage.pvc.input: montage-ai-input-rwx`
- `storage.pvc.output: montage-ai-output-rwx`
- `storage.pvc.music: montage-ai-music-rwx`
- `storage.pvc.assets: montage-ai-assets-rwx`

Then deploy:

```bash
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster
```

## Phase 5: Post-Cutover Validation

1. Access modes now RWX:

```bash
kubectl get pvc -n "$NAMESPACE" -o custom-columns='NAME:.metadata.name,ACCESS:.spec.accessModes[*]'
```

1. App health:

```bash
kubectl get pods -n "$NAMESPACE"
kubectl exec -n "$NAMESPACE" deploy/montage-ai-web -- python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/api/status', timeout=5).status)"
```

1. Data validation:

- Open existing projects
- Render one known test asset
- Verify output files visible in `/data/output`

## Rollback

If cutover fails:

1. Scale web/worker to 0.
2. Revert `storage.pvc.*` values in `deploy/k3s/config-global.yaml` to old PVC names.
3. `make -C deploy/k3s config && make -C deploy/k3s deploy-cluster`
4. Scale app back up.

## Decommission Old PVCs

Only after successful soak period and sign-off.

```bash
kubectl delete pvc -n "$NAMESPACE" montage-ai-input-nfs montage-ai-output-nfs montage-ai-music-nfs montage-ai-assets-nfs
```

## Sign-Off Evidence

Record migration completion in `docs/operations/MOE_GO_LIVE_SIGNOFF.md`.
