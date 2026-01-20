# [action required] Staging: KEDA, image-distribution & PV topology — changes made + IaC recommendations

Repository: mfahsold/fluxibri_core — infra team
Date: 2026-01-20
Author: montage-ai (automation)

SUMMARY
-------
Ich habe heute Änderungen in der Staging‑Cluster‑Umgebung vorgenommen, um das Queue‑getriebene Autoscaling für `montage-ai` zu ermöglichen und einen kurzfristig reproduzierbaren Smoke‑Test zu ermöglichen. Dieser Issue‑Draft fasst die Änderungen, beobachteten Probleme und konkrete IaC‑Änderungen zusammen, damit wir die Fixes dauerhaft in `fluxibri_core` persistieren und stabil betreiben können.

CHANGES APPLIED (staging, live)
-------------------------------
- KEDA (kedacore/keda) per Helm installiert (namespace: `keda`).
- `ScaledObject` für `montage-ai-worker` angelegt (queue length → HPA).
- Vorher vorhandene HPA für denselben Deployment entfernt (Konflikt behoben).
- Staging image distribution: lokales Image in in‑cluster Registry (NodePort) gepusht und Deployments auf `127.0.0.1:30500/montage-ai:staging-<sha>` umgestellt (Workaround für GHCR pull auth).
- Staging Deployment‑patches angewendet:
  - `emptyDir` fallback für `cache` (staging only)
  - permissive `tolerations` und reduzierte resource requests (staging only)
  - entfernte/relaxte nodeSelector (hostname) für bessere Schedulability
  - `REDIS_HOST` auf `redis.default.svc.cluster.local` gesetzt
- Opt‑in Smoke‑Test hinzugefügt (tests/integration/test_queue_scaling.py, RUN_SCALE_TESTS=1).

CURRENT BEHAVIOR / OBSERVATIONS
-------------------------------
- KEDA ist installiert und `ScaledObject` akzeptiert (no HPA conflict now).
- At least one worker pod reached Running and connected to Redis.
- Smoke test currently fails to complete E2E in staging because multiple worker replicas stay Pending due to PV node‑affinity and cluster capacity constraints (local-path PVs bound to a single node).
- GHCR image pulls failed in-cluster (401) — mitigated by pushing image to local cluster registry.

IMPACT / RISK
-------------
- Short term: staging autoscaling can be validated only at low scale (1→2 pods) because PV topology restricts pods to specific nodes.
- Without persisting image distribution or GHCR pull secrets, CI/staging rollouts will continue to fail intermittently.
- If both an HPA and a ScaledObject are present for one workload, KEDA will reject the ScaledObject — IaC must ensure only one controller manages the target.

RECOMMENDED IaC ACTIONS (high priority — please persist)
-------------------------------------------------------
1) Install Helm charts (idempotent) in `fluxibri_core` (values + versioned):
   - KEDA (mandatory for queue→autoscale)
   - metrics-server (HPA resource metrics)
   - prometheus + prometheus-adapter (optional but recommended for SLOs and custom metrics)

   Example (HelmRelease/helmfile / Terraform-helm):
   - chart: kedacore/keda
     name: keda
     namespace: keda
     version: <pin-version>
     values:
       installCRDs: true

2) Persist an in‑cluster registry or CI image mirror workflow:
   - Add a CI job that pushes tested images into the cluster registry used by staging (NodePort or internal registry).
   - Alternatively: add `imagePullSecrets` support and document GHCR token provisioning for staging nodes.

3) PV topology / storage class changes (blocking for autoscaling):
   - Convert staging `montage-*` PVCs to ReadWriteMany (NFS/managed RWM) OR
   - Provide an NFS/CEPH RWM StorageClass for `montage-input|output|assets|music` so multiple worker pods can schedule on different nodes.
   - Short-term: allow a staging-only hostPath/emptyDir overlay for non‑persistent integration tests.

4) Node pool / capacity automation:
   - Add node-pool autoscaler (cluster autoscaler) or Terraform config to increase worker-capable nodes on demand.
   - Tag nodes with GPU/CGPU labels for future CGPU scheduling (MIG-aware labels).

5) CI / GitHub Actions / Flux changes:
   - Add a staged job that: build → push to cluster registry → deploy staging overlay → run RUN_SCALE_TESTS=1 smoke test.
   - Ensure the job injects `IMAGE_PULL_SECRET` if GHCR is used.

6) Kustomize / chart hygiene:
   - Keep separate overlays: `staging` uses KEDA + staging fallbacks; `production` uses stricter PV/nodeSelector and GHCR.
   - Ensure manifests do not apply both HPA and ScaledObject to same workload (add a guard in kustomize or CI validation).

REPRO / EXACT COMMANDS (for the issue body / runbook)
----------------------------------------------------
# install KEDA
helm repo add kedacore https://kedacore.github.io/charts && helm repo update
helm upgrade --install keda kedacore/keda --namespace keda --create-namespace --wait --timeout 5m

# push tested image into cluster registry (example)
docker tag montage-ai:${TAG} 127.0.0.1:30500/montage-ai:${TAG}
docker push 127.0.0.1:30500/montage-ai:${TAG}
kubectl -n montage-ai set image deploy/montage-ai-worker worker=127.0.0.1:30500/montage-ai:${TAG}

# quick PV check (why pods remain Pending)
kubectl -n montage-ai get pvc -o wide
kubectl get pv <pv-name> -o yaml  # check nodeAffinity

SUGGESTED IaC PRS / CHANGES (small, prioritized)
-----------------------------------------------
- Add `charts/keda/` entry in Helm values and a `helm_release` block in Terraform (or Flux HelmRelease) — pin version + enable CRDs.
- Add `staging/overlays/registry-mirror.yaml` (kustomize) + CI step to push images to in-cluster registry for staging runs.
- Add `storage/rwm-nfs` module (example) and change storage class for `montage-*` PVCs in staging.
- Add CI smoke job: `staging-e2e-autoscale` that runs `RUN_SCALE_TESTS=1` with a short timeout.

TESTING & VALIDATION (what to run after IaC changes)
---------------------------------------------------
- kubectl -n montage-ai apply -k deploy/k3s/overlays/staging
- Verify KEDA ScaledObject exists and HPA is not present for the same target.
- Trigger CI job that builds & pushes image to cluster registry, then runs: RUN_SCALE_TESTS=1 pytest tests/integration/test_queue_scaling.py
- Validate: jobs complete and worker replicas scale (observe HPA created by KEDA or HPA metrics).

ROLLBACK / MITIGATION
---------------------
- To revert KEDA: helm uninstall keda -n keda
- To revert Deployments to GHCR: set image back to ghcr.io/... and remove in-cluster registry tag from manifests

REQUEST / ACTIONS FOR INFRA TEAM
--------------------------------
- Please create IaC entries for KEDA + metrics-server + an RWM storage class (or approve a staging NFS) and add an image mirror/push CI step. 
- Optional: add cluster secret management for GHCR credentials (secrets in Vault → injected into cluster via ExternalSecrets).

PROPOSED ISSUE TITLE
--------------------
[staging] Persist KEDA + registry mirror + RWM storage for montage‑ai autoscaling validation

LABELS / ASSIGNEES (suggestion)
-------------------------------
labels: infra, staging, keda, blocker
assignees: @infra-lead, @storage-team

---

If you want, I can open this issue in `mfahsold/fluxibri_core` (I can run `gh issue create` if you provide repo permissions / token). Otherwise: copy the text above into a new issue — it's ready to post.
