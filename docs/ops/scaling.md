# Scaling & Autoscaling (overview)

This short runbook describes the recommended, DRY approach used by Montage AI
for scaling background workers and media encoders.

Principles
- Queue‑driven autoscaling (KEDA) is primary; HPA (CPU/memory) is a safety net.
- Keep Web (API) and Worker responsibilities separated (one process per role).
- Avoid hardcoded values in base manifests — use `deploy/k3s/config-global.yaml` for environment overrides.
- **Cluster mode is canonical for distributed jobs**; use local mode only for single‑node/dev runs.

Optional (SOTA)
- Use JobSet for distributed render phases when you need shard‑level coordination.
- Prefer cluster-native queues (Redis/KEDA) over ad‑hoc job manifests.

Canonical cluster workflow (Fluxibri core)
- `fluxibri hardware quotas` to set adaptive quotas and prevent OOM
- `fluxibri hardware autoscale` to observe KEDA/HPA scaling
- `fluxibri cluster services` for live routing/availability

Quick checklist
- [ ] KEDA installed and configured (staging first)
- [ ] Metrics server present (HPA CPU metrics)
- [ ] Redis exporter or KEDA redis scaler configured
- [ ] Worker Deployment: resource requests/limits set
- [ ] HPA + ScaledObject deployed to staging
- [ ] CI smoke test enabled (RUN_SCALE_TESTS=1)
- [ ] Quotas/priority managed via `fluxibri hardware quotas` (avoid in-repo hardcoding)

Tuning pointers
- Start conservative: listLength threshold = 10, maxReplicas = 8
- Use stabilizationWindowSeconds to avoid flapping
- Route heavy ffmpeg jobs to GPU/encoder node pool via nodeAffinity

Runbook (cluster validation)
1. Apply canonical overlay: `make -C deploy/k3s deploy-cluster`
2. Validate that HPA/ScaledObjects exist and that workers scale with real queue
   load (use your preferred job generator or integration tests).
3. Observe: `kubectl -n "${CLUSTER_NAMESPACE:-montage-ai}" top pods --containers`

Detailed smoke workflows and cluster-specific steps are maintained in the
private docs set.
