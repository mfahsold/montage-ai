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

## Preview Validation (Public)

Use the canonical preview SLO doc for public, non-environment-specific guidance:

`docs/operations/preview-slo.md`

## Escalation

If these steps do not resolve the issue, gather the pod logs/events and open an issue with the details.
