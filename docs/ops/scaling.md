# Scaling & Autoscaling (overview)

This short runbook describes the recommended, DRY approach used by Montage AI
for scaling background workers and media encoders.

Principles
- Queue‑driven autoscaling (KEDA) is primary; HPA (CPU/memory) is a safety net.
- Keep Web (API) and Worker responsibilities separated (one process per role).
- Avoid hardcoded values in base manifests — use `deploy/config.env` and overlays for environment overrides.

Quick checklist
- [ ] KEDA installed and configured (staging first)
- [ ] Metrics server present (HPA CPU metrics)
- [ ] Redis exporter or KEDA redis scaler configured
- [ ] Worker Deployment: resource requests/limits set
- [ ] HPA + ScaledObject deployed to staging
- [ ] CI smoke test enabled (RUN_SCALE_TESTS=1)

Tuning pointers
- Start conservative: listLength threshold = 10, maxReplicas = 8
- Use stabilizationWindowSeconds to avoid flapping
- Route heavy ffmpeg jobs to GPU/encoder node pool via nodeAffinity

Runbook (staging validation)
1. Apply staging overlay: `kubectl apply -k deploy/k3s/overlays/staging`
2. Quick dev validation (ephemeral, no production data):
   - Prefer: run the smoke against the canonical overlay in a staging-like namespace (example uses `montage-ai-clean`):
     - `./scripts/ci/run-dev-smoke.sh --image <REGISTRY>/montage-ai:<TAG> --overlay deploy/k3s/overlays/production`
     - The helper will apply the overlay into an isolated namespace for validation (avoid touching prod PVCs).
   - Optional: `deploy/k3s/overlays/clean-deploy/` remains available for quick local checks but is not canonical.
   - Runner requirement (CI): self-hosted runner with label `scale-smoke`.
3. Run smoke (integration): `RUN_SCALE_TESTS=1 pytest -q tests/integration/test_queue_scaling.py -q`
4. Observe: `kubectl -n montage-ai top pods --containers` and dashboard

Notes:
- Use `clean-deploy` to validate KEDA/HPA behaviour without touching production PVCs.
- CI smoke is gated and non-blocking for `main`; failures are diagnostic (do not auto-promote).

For full tuning and production checklist see docs/ops/scale_tuning.md (TBD).