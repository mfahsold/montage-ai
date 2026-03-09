# Operations Hub

Public operational guidance for Montage AI. Cluster-specific runbooks (internal
endpoints, node topology, incident playbooks) live in the private docs set.

## Core Runbooks

- **[Cluster Infrastructure Requirements](CLUSTER_REQUIREMENTS.md)** — Technical requirements for production clusters (Muss/Soll/Kann)
- **[Cluster Rollout Checklist](MOE_CLUSTER_ROLLOUT_CHECKLIST.md)** — Pre-deployment validation checklist
- **[Go-Live Sign-Off Tracker](MOE_GO_LIVE_SIGNOFF.md)** — Dated owner/status/evidence matrix for release approval
- **[RWX PVC Migration Runbook](RWX_PVC_MIGRATION_RUNBOOK.md)** — Runtime migration path from legacy RWO to RWX PVCs
- [Rollback & Recovery Guide](rollback.md) — Idempotency, PVC lifecycle, rollback, scaling
- [Preview SLO (canonical)](preview-slo.md)
- [Scaling & Autoscaling](scaling.md)
- [KEDA / Metrics / Autoscaler](infra_keda.md)
- [Dev Autoscale Smoke](dev-autoscale-smoke.md)

## Quick Links

| Document | Purpose |
| ---------- | ------- |
| [CLUSTER_REQUIREMENTS.md](CLUSTER_REQUIREMENTS.md) | Complete infrastructure requirements (GPU, storage, networking, observability) |
| [MOE_CLUSTER_ROLLOUT_CHECKLIST.md](MOE_CLUSTER_ROLLOUT_CHECKLIST.md) | Go-live validation checklist |
| [MOE_GO_LIVE_SIGNOFF.md](MOE_GO_LIVE_SIGNOFF.md) | Canonical dated sign-off and gap closure evidence |
| [RWX_PVC_MIGRATION_RUNBOOK.md](RWX_PVC_MIGRATION_RUNBOOK.md) | Step-by-step RWO -> RWX migration and rollback |

## Need More Detail?

If you need the internal runbooks, contact the maintainers.
