# Montage AI Go-Live Sign-Off Tracker

**Status:** OPEN  
**Release Candidate:** `main`  
**Started:** 2026-03-09  
**Last Updated:** 2026-03-09

This file is the canonical sign-off record for cluster rollout readiness.
Use it as a living approval matrix with dated evidence links.

## Must-Gate Closure Matrix

| Gate | Owner | Status | Evidence | Target Date | Verified Date |
| ---- | ----- | ------ | -------- | ----------- | ------------- |
| Shared storage contract (RWX PVCs for `/data/input`, `/data/output`, `/data/music`, `/data/assets`) | Infra + App | pending | `deploy/k3s/base/pvc.yaml` | 2026-03-09 | TBD |
| Restrictive NetworkPolicies (default deny + explicit allows, no allow-all) | Infra | pending | `deploy/k3s/base/network-policy.yaml` | 2026-03-09 | TBD |
| Montage-specific observability (ServiceMonitor + PrometheusRule + dashboard refs) | SRE/Ops | pending | `deploy/k3s/base/monitoring.yaml`, `deploy/grafana/dashboards/preview_slo.json` | 2026-03-09 | TBD |
| Go-live acceptance test run attached (load/resilience/recovery + alert validation) | QA + SRE/Ops | pending | `docs/operations/MOE_CLUSTER_ROLLOUT_CHECKLIST.md` | 2026-03-10 | TBD |

## Approval Sign-Off

| Role | Name | Decision | Date | Notes |
| ---- | ---- | -------- | ---- | ----- |
| Deploy Engineer | TBD | pending | TBD | TBD |
| QA Lead | TBD | pending | TBD | TBD |
| SRE/Ops | TBD | pending | TBD | TBD |
| Product Owner | TBD | pending | TBD | TBD |

## Exit Criteria

A release candidate is `GO` only when:

- Every must-gate above is `ready`.
- All evidence links are populated and verifiable.
- All four approver roles marked `approved` with date.

## Evidence Commands

```bash
# Verify canonical manifests contain restrictive policy and observability artifacts
bash deploy/k3s/pre-flight-check.sh

# Verify rendered manifests directly
kustomize build --load-restrictor LoadRestrictionsNone deploy/k3s/base | grep -E "kind: (NetworkPolicy|ServiceMonitor|PrometheusRule)"

# Verify PVC access modes in cluster
kubectl get pvc -n montage-ai -o custom-columns='NAME:.metadata.name,ACCESS:.spec.accessModes'
```
