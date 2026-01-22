Canonical multi-arch + distributed deploy (runbook)

Scope
- Build multi-arch images (amd64 + arm64)
- Mirror third-party images to internal registry
- Preflight quota & PVs
- Deploy `deploy/k3s/overlays/distributed` and run smoke

Assumptions
- Internal registry: `192.168.1.12:30500` is reachable from cluster nodes
- SRE will provision (or bump) PVC quota when requested

Steps (summary)

1) Preflight (must-pass)
- ./scripts/k8s/check-quota.sh montage-ai
- Ensure registry reachable from nodes (curl from a node)

2) Build & mirror images (canonical/in-cluster preferred)
- Preferred (canonical): run kaniko in-cluster (deploy/k3s/base/kaniko-build-job.yaml)
  - For amd64: kaniko job -> `192.168.1.12:30500/montage-ai:main-<sha>-amd64`
  - For arm64: kaniko (nodeSelector: arm64) -> `...:main-<sha>-arm64`
- Create manifest list (multi-arch):
  - docker buildx imagetools create --tag 192.168.1.12:30500/montage-ai:main-<sha> \
      192.168.1.12:30500/montage-ai:main-<sha>-amd64 \
      192.168.1.12:30500/montage-ai:main-<sha>-arm64

3) Mirror 3rd-party images used in overlay (avoid Docker Hub rate limits)
- Example: docker pull redis:6.2-alpine && docker tag/push to internal registry

4) Deploy (safe sequence)
- kubectl -n montage-ai apply -k deploy/k3s/base   # uses existing RWX PVCs
- kubectl -n montage-ai apply -k deploy/k3s/overlays/dev  # DEV safe defaults
- After quota/PVs available: kubectl -n montage-ai apply -k deploy/k3s/overlays/distributed

5) Verify (smoke)
- RUN_DEV_E2E=true dev_base_url=https://<dev-host> pytest -q tests/integration/test_preview_smoke_ci.py::test_preview_smoke_and_metrics
- ./scripts/ci/preview-benchmark.sh BASE=https://<dev-host> RUNS=10
- Collect: /metrics snapshot, job JSONs, preview MP4s

6) Post-deploy: docs & alerts
- Add SLO alerts (p95 > threshold) and dashboard (we already add preview_slo.json)
- Attach artifacts to infra ticket and close

Troubleshooting (common)
- Failed to create PVCs → ResourceQuota (open infra ticket, or pre-create PVs)
- ImagePullBackOff on arm nodes → ensure multi-arch manifest exists or mirror arch-specific image
- Pip/socket timeouts during arm build → retry with in-cluster builder or increase pip timeout / use a wheelhouse

Useful commands
- kubectl -n montage-ai get pvc
- kubectl -n montage-ai describe resourcequota storage-quota
- curl -sS https://<dev-host>/metrics | egrep 'montage_time_to_preview_seconds|montage_proxy_cache'

Contact SRE checklist
- Increase PVC quota or pre-provision NFS PVs
- Mirror internal registry images
- Approve kaniko in-cluster builder job if needed
