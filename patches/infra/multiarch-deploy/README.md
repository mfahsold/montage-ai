What this patchset contains

- `0001-increase-montage-ai-quota.yaml` — suggested ResourceQuota bump for `montage-ai` namespace.
- `0002-precreate-nfs-pvs.yaml` — example NFS PV manifests (fill in `server`/`path` with infra values).

How to use

1. Review and approve the ResourceQuota change with SRE. Do NOT apply without approval.
2. If quota cannot be bumped immediately, pre-create the PVs from `0002-*` (SRE will supply NFS server path).
3. Once PVs/quota are in place, re-run:
   - `kubectl -n montage-ai apply -k deploy/k3s/overlays/distributed`
   - `./scripts/ci/preview-benchmark.sh BASE=https://<dev-host> RUNS=10`

Runbook notes

- PVC quota is the current blocker. The minimal unblock is either increasing the quota or pre-creating PVs.
- Mirroring 3rd-party images (redis, etc.) into the internal registry is recommended to avoid Docker Hub rate limits.
- After infra changes, re-run the distributed deploy and validate with the opt-in smoke test `tests/integration/test_preview_smoke_ci.py`.
