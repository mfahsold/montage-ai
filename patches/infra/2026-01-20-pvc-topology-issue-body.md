Title: [staging] local-path (RWO) PVC topology prevents worker autoscaling — convert staging PVCs to RWM or provide NFS overlay

Summary
-------
Staging `montage-ai` cannot scale worker pods across nodes because PVCs are provisioned using the `local-path` StorageClass (ReadWriteOnce). The cluster and scheduler are functioning correctly — the PVCs/PVs are node‑affined, which prevents pods that require those PVCs from scheduling to other nodes. This blocks KEDA/HPA‑driven autoscaling validation.

Why this matters
-----------------
- Autoscaling requires worker pods to land on any node; RWO local‑path PVCs tie volumes to a single node.
- Current staging tests and the opt‑in autoscale smoke (RUN_SCALE_TESTS=1) fail intermittently or cannot validate scaling beyond a single node.

Immediate evidence (collected 2026-01-20)
-----------------------------------------
- PV is node‑affined (local-path behaviour):

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pvc-df021c0e-eba1-493d-9392-e93c13e7a5fd
spec:
  capacity:
    storage: 50Gi
  local:
    path: /var/lib/rancher/k3s/storage/.../montage-input
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - codeai
  storageClassName: local-path
```

- Pod scheduling event (example):

```
Warning  FailedScheduling  31m  default-scheduler  0/8 nodes are available: 2 Insufficient memory, 2 node(s) didn't match PersistentVolume's node affinity, 4 node(s) didn't match Pod's node affinity/selector.
```

- Repo config (k3s example): `deploy/k3s/base/pvc.yaml` defaults to an RWX access mode and pulls `storageClassName` from `deploy/k3s/config-global.yaml`. Legacy overlays are archived under `deploy/k3s/overlays/legacy/` and should not be used for canonical deployments.

Root cause
----------
Configuration+design: the k3s example and default staging use `local-path` (RWO) by design for single-node performance. This is expected behavior for that StorageClass — the cluster is operating correctly. The effect is that pods needing those PVCs cannot schedule on nodes that do not host the matching PV.

Impact
------
- Blocks E2E validation of queue-driven autoscaling (KEDA) across nodes.
- Prevents running multiple worker replicas on different nodes for high‑concurrency workloads.
- Causes flaky staging runs and CI smoke failures unless staging is constrained to a single node.

Recommended IaC changes (priority order)
----------------------------------------
1) HIGH — Provide an RWX StorageClass for staging and migrate `montage-*` PVCs in staging to use it (NFS, CephFS, or CSI that supports RWX). Update `deploy/k3s/config-global.yaml` with the class and provision the backend in IaC/Flux.

2) HIGH — Add a CI job that mirrors images to an in‑cluster registry (or ensures `imagePullSecrets` for GHCR); include `push-to-cluster-registry` helper and document secrets handling.

3) MEDIUM — Add a `staging` kustomize guard that toggles between `local-path` (single-node dev) and `nfs-pvc` (multi-node autoscale validation). Document the expected behavior in `deploy/k3s/README.md`.

4) MEDIUM — Add a validation step in CI that detects incompatible combinations (e.g., `storageClassName: local-path` + `replicas > 1` for workloads that mount RWO volumes) and fails early.

5) LOW — For short lived integration tests, provide a safe staging overlay that uses `emptyDir` for `montage-cache` and hostPath/ephemeral for `montage-input/output` (non-persistent test-only overlay).

Reproduction (how I validated)
-----------------------------
- Observed PV nodeAffinity and PVC bindings in staging:
  - `kubectl get pvc -n montage-ai` (shows RWO/local-path)
  - `kubectl get pv <pv>` (shows `local.path.provisioner/selected-node: codeai`)
  - `kubectl describe pod <montage-ai-worker-pod>` → failed scheduling due to PV nodeAffinity
- Repo references:
  - `deploy/k3s/base/pvc.yaml` (defaults to `local-path`, RWO)
  - `deploy/k3s/overlays/distributed/nfs-pvc.yaml` (already present as an override)

Suggested PRs (what I can draft)
--------------------------------
- Add `flux/helmrelease` for an NFS provisioner or RWM storage module + enable `deploy/k3s/overlays/distributed/nfs-pvc.yaml` for staging. (small, ~50 LOC)
- CI: `build → mirror-to-cluster-registry → deploy:staging → RUN_SCALE_TESTS=1` (new workflow job)
- Add kustomize overlay `overlays/staging/test-rwm` and CI checks to flip it on for autoscale validation.

Severity & rollout
-------------------
- Severity: P1 for autoscaling validation in staging (blocks KEDA/HPA verification).
- Rollout: migrate staging first; keep `local-path` for single-node dev. Add migration notes and a rollback (revert to `local-path` overlay).

Attachments / logs
------------------
- PV nodeAffinity excerpt and pod scheduling events are attached to this issue (captured 2026-01-20).

Suggested labels: infra, storage, staging, blocker
Suggested assignees: infra team / storage team

Next steps I can take
---------------------
- Open this issue in `mfahsold/fluxibri_core` (ready).  
- Draft PR that: (A) provisions RWM storage for staging and (B) adds CI mirror + smoke job.  
