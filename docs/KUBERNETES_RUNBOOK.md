# Kubernetes Runbook (Public)

This public runbook contains basic, non-environment-specific checks. The full operational runbook (cluster topology, internal hosts, incident playbooks) lives in the private docs set. Contact the maintainers if you need access.

---

## Quick Health Checks

```bash
# Namespaces and pods
kubectl get ns
kubectl get pods -n "${CLUSTER_NAMESPACE:-montage-ai}"

# Recent events
kubectl get events -n "${CLUSTER_NAMESPACE:-montage-ai}" --sort-by=.lastTimestamp
```

## Common Diagnostics

```bash
# Pod details
kubectl describe pod -n "${CLUSTER_NAMESPACE:-montage-ai}" -l app.kubernetes.io/name=montage-ai

# Logs
kubectl logs -n "${CLUSTER_NAMESPACE:-montage-ai}" -l app.kubernetes.io/name=montage-ai --tail=200
```

## Service Access

```bash
# Port-forward for local access
LOCAL_PORT="${LOCAL_PORT:-5000}"
kubectl port-forward -n "${CLUSTER_NAMESPACE:-montage-ai}" svc/montage-ai-web "${LOCAL_PORT}:8080"
```

Open `http://localhost:${LOCAL_PORT}` in your browser.

If you have ingress configured, use your own domain, for example:

```bash
curl -H "Host: YOUR_MONTAGE_HOST" http://YOUR_INGRESS_IP/health
```

## DEV: preview fast-path — quick validation checklist

When validating the **preview** fast-path (distributed overlay) in DEV, follow these fast checks to avoid common failures:

- PVC / Quota
  - Ensure the `montage-ai` namespace has sufficient `ResourceQuota` for `persistentvolumeclaims` **and** `requests.storage`. The `distributed` overlay creates several NFS PVCs and will fail if the quota is exceeded (pods remain `Pending`).
  - Short-term workaround: reuse existing `montage-*-rwx` PVCs in DEV to validate functionality without provisioning new volumes.

- Minimal smoke fixtures
  - Copy two small test clips to `/data/input/` and a short WAV (>= 3–4s) to `/data/music/` on the input PVC before running the smoke.

- Run the preview SLO smoke (canonical steps are in docs/operations/preview-slo.md)

```bash
# run from a machine that can reach the DEV ingress
./scripts/ci/preview-benchmark.sh BASE=https://<dev-host> RUNS=10 --collect-metrics
```

See `patches/infra/2026-01-22-dev-distributed-deploy-report.md` for a recent deployment attempt and remediation steps.

## Escalation

If these steps do not resolve the issue, gather the pod logs/events and open an issue with the details.
