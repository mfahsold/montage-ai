# Deployment Notes — 2026‑02‑10

## Summary

This deployment focused on restoring cluster rendering reliability and enabling high‑quality creative runs on the K3s environment.

## Changes Applied

1. **In‑cluster Kubernetes discovery (no kubectl dependency)**
   - Node discovery now uses the in‑cluster Kubernetes API when available.
   - This removes the `kubectl` hard dependency inside renderer pods.

2. **Story arc spec compatibility**
   - `StoryArc.from_spec` now accepts Pydantic models by converting them into dicts.
   - Fixes: `StoryArcConfig` object has no attribute `get` during Story Engine planning.

3. **Render job hardening**
   - Renderer jobs now use the `montage-ai-cluster` service account to allow K8s API access.
   - JobSet usage disabled in the job environment (`CLUSTER_USE_JOBSET=false`) due to CRD incompatibility on current cluster.

4. **Cluster RBAC for node discovery**
   - Added a ClusterRole/ClusterRoleBinding to allow listing nodes for in‑cluster discovery.

5. **Registry connectivity (cluster‑local workaround)**
   - Containerd `hosts.toml` entries were added inside the K3d node for HTTP registry endpoints to allow image pulls.

## Observed Warnings (Expected)

- **CGPU tension analysis unavailable** → Story Engine falls back to dummy tension values.
- **CGPU credentials secret missing** → `cgpu status` reports missing OAuth credentials in worker pods.
- **JobSet CRD install failure** → Cluster version does not accept current JobSet validation rules.

## Current Status

- Render jobs start successfully in the cluster and proceed through analysis phases.
- Scene detection runs via standard distributed job (JobSet disabled).

## Follow‑ups

- Align cluster JobSet CRD version with API server validation rules.
- Restore CGPU availability for tension analysis in Story Engine (run `scripts/ops/cgpu-refresh-session.sh`).
- Document preferred registry endpoint once infra standardizes on TLS/HTTP.
