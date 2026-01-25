# Dev autoscale smoke (developer guide)

Purpose: provide a deterministic, low‑risk smoke to validate KEDA/HPA + worker autoscaling in `dev` or ephemeral environments.

When to use
- Validate queue-driven autoscaling after infra or worker changes
- Run as an opt-in CI job on a self-hosted runner
- Manual validation before promoting RWX migration to staging

Prerequisites
- Cluster access (kubeconfig) with your target namespace (`CLUSTER_NAMESPACE`)
- A self-hosted runner (for CI) with label `scale-smoke` and tools: kubectl, kustomize, docker
- `ENABLE_DEV_ENDPOINTS=true` enabled in the deployment overlay

How it works (high level)
1. Deploy `deploy/k3s/overlays/legacy/clean-deploy` (ephemeral volumes, dev guards)
2. Post 10–20 lightweight dev test jobs to `/api/internal/testjob`
3. Assert that KEDA creates an HPA and workers scale to expected replicas
4. Verify jobs complete and collect logs

Quick commands (local)

```bash
# apply clean/dev overlay
CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
LOCAL_PORT="${LOCAL_PORT:-8080}"

kubectl -n "$CLUSTER_NAMESPACE" apply -k deploy/k3s/overlays/legacy/clean-deploy/

# port-forward to reach web UI from your machine
kubectl -n "$CLUSTER_NAMESPACE" port-forward svc/montage-ai-web "${LOCAL_PORT}:80" &

# POST 12 dev jobs (6s each)
MONTAGE_API_BASE="http://localhost:${LOCAL_PORT}"
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"count":12,"duration":6}' "${MONTAGE_API_BASE}/api/internal/testjob"

# watch worker logs
kubectl -n "$CLUSTER_NAMESPACE" logs deployment/montage-ai-worker -f
```

CI (recommended)
- Use the workflow `/.github/workflows/dev-autoscale-smoke.yml` on a self-hosted runner labeled `scale-smoke`.
- The workflow runs `scripts/ci/run-dev-smoke.sh` (pass `--overlay` to apply a specific overlay), posts dev jobs and asserts scaling.

Troubleshooting
- If jobs are not picked up: check Redis connectivity and worker logs.
- If HPA does not react: check KEDA ScaledObject status and ensure numeric trigger values are used.

Safety notes
- This overlay uses `emptyDir` volumes; it is safe for environments without production data.
- The workflow targets `deploy/k3s/overlays/legacy/clean-deploy` only (dev). Do NOT run against production overlays.
