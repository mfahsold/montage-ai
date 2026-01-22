Title: DEV distributed deploy — partial deploy, blocked by storage quota
Date: 2026-01-22
Author: automation (repo agent)

Summary
-------
Attempted to deploy `montage-ai` using the `deploy/k3s/overlays/distributed` overlay after promoting local changes (branch `infra/dev-deploy-preview-7ebbcbf` → `main`). The overlay applied partially: web + worker deployments were updated but creation of several NFS PVCs failed due to namespace ResourceQuota limits. Some Job manifests also failed validation due to missing volume definitions.

What I did
---------
- Merged `infra/dev-deploy-preview-7ebbcbf` into `main` and pushed to `origin/main`.
- Applied `kubectl -k deploy/k3s/overlays/distributed` against the DEV namespace.
- Verified pod/deployment state and attempted distributed smoke runs (benchmark script).
- Uploaded small test fixtures to existing RWX PVCs and re-ran lightweight smoke tests.

Current state
-------------
- `montage-ai-worker`: RUNNING (rolled out) — able to pick up jobs from Redis and execute limited preview flows using existing RWX PVCs.
- `montage-ai-web`: existing replica running; new replica Pending (some PVCs Pending).
- PVCs: overlay attempted to create NFS-style PVCs (`montage-*-nfs`) but creation failed due to namespace quota. Existing `montage-*-rwx` PVCs are Bound and were reused for smoke.
- Smoke benchmark: attempted but **many runs failed** (jobs either queued or failed) — root causes observed: missing music file (fixed by uploading a 4s WAV) and an audio-analysis IndexError (fixed in repo but not present in the image currently running on cluster).

Primary blockers
----------------
1. Namespace ResourceQuota prevents creating the additional PVCs required by `deploy/k3s/overlays/distributed`. (quota: persistentvolumeclaims=10, requests.storage=1Ti; attempted new PVCs exceeded limits)
2. Some Job manifests reference volume mounts (`shm`, `script`, `patch-*`) that are not defined in the overlay — manifest validation errors.
3. The runtime image for commit `7ebbcbf` is not yet available in the cluster registry (CI needs to build/publish), so the cluster is running an older image that still contained the audio-analysis IndexError.

Recommended next steps
----------------------
- P0: Infra to increase the `montage-ai` namespace ResourceQuota (PVC count + storage) or pre-provision the required NFS volumes and update the overlay to reference them.
- P0: Either trigger CI/Tekton to build and publish the `main@7ebbcbf` image, or allow me to push a tested image to the internal registry (`127.0.0.1:30500`) for immediate verification.
- P1: Small kustomize fix: add fallback volume definitions or make the Job mounts conditional so validation does not fail in environments without optional volumes.
- P1: After quota + image are available: re-apply `deploy/k3s/overlays/distributed`, run `./scripts/ci/preview-benchmark.sh BASE=https://<dev-host> RUNS=10` and attach `/metrics`, job JSONs and preview artifact.

Artifacts collected
-------------------
- `/tmp/preview-benchmark-run-2.txt` (benchmark attempt output)
- Worker logs (tail) showing the IndexError and subsequent partial successes (available on request)
- Cluster `kubectl get pvc` / `kubectl get pods` snapshots (included in PR comments)

Decision / ask
--------------
Please confirm one of the following:
- `quota` — infra increases namespace PVC/storage quota (I will reapply and verify). (recommended)
- `reuse-pvc` — allow me to open a short PR that updates `deploy/k3s/overlays/distributed` to reuse existing `montage-*-rwx` PVCs for immediate verification (non-destructive). I will then re-deploy and run the benchmark.
- `ci-build` — you prefer Tekton to build/publish the image; I will wait and re-run verification once CI publishes the image.

I will NOT close the issue until the distributed overlay can create all required PVCs and the benchmark passes the SLOs (p50/p95). If you want me to open the fluxibri_core PR/issue to request quota changes, say `open-infra-issue` and I will prepare the description and PR/issue draft.