# Dev autoscale smoke (developer guide)

Purpose: provide a deterministic, low‑risk smoke to validate KEDA/HPA + worker autoscaling in `dev` or ephemeral environments.

When to use
- Validate queue-driven autoscaling after infra or worker changes
- Run as an opt-in CI job on a self-hosted runner
- Manual validation before promoting RWX migration to staging

Prerequisites
- Cluster access (kubeconfig) with namespace `montage-ai`
- A self-hosted runner (for CI) with label `scale-smoke` and tools: kubectl, kustomize, docker
- `ENABLE_DEV_ENDPOINTS=true` enabled in the deployment overlay

How it works (high level)
1. Deploy `deploy/k3s/overlays/clean-deploy` (ephemeral volumes, dev guards)
2. Post 10–20 lightweight dev test jobs to `/api/internal/testjob`
3. Assert that KEDA creates an HPA and workers scale to expected replicas
4. Verify jobs complete and collect logs

Quick commands (local)

# apply clean/dev overlay
kubectl -n montage-ai apply -k deploy/k3s/overlays/clean-deploy/

# port-forward to reach web UI from your machine
kubectl -n montage-ai port-forward svc/montage-ai-web 8080:80 &

# POST 12 dev jobs (6s each)
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"count":12,"duration":6}' http://127.0.0.1:8080/api/internal/testjob

# watch worker logs
kubectl -n montage-ai logs deployment/montage-ai-worker -f

CI (recommended)
- Use the workflow `/.github/workflows/dev-autoscale-smoke.yml` on a self-hosted runner labeled `scale-smoke`.
- The workflow runs `scripts/ci/run-dev-smoke.sh` which applies the overlay, posts dev jobs and asserts scaling.

Troubleshooting
- If jobs are not picked up: check Redis connectivity and worker logs.
- If HPA does not react: check KEDA ScaledObject status and ensure numeric trigger values are used.

Safety notes
- This overlay uses `emptyDir` volumes; it is safe for environments without production data.
- The workflow targets `deploy/k3s/overlays/clean-deploy` only (dev). Do NOT run against production overlays.
