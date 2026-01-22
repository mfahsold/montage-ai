---
title: "Montage‑AI — Preview SLO (canonical)"
summary: "Canonical SLOs, benchmark, developer and operator verification instructions for the Montage‑AI preview fast-path. This file is the canonical source intended for the montage‑ai repository (app-owned)."
updated: 2026-01-22
---

## Purpose

Canonical, developer‑facing specification for Montage‑AI preview SLOs, benchmark procedure, metrics schema and guidance for reproducing and debugging failures. This document is *owned by the montage‑ai team* — infra (fluxibri_core) should reference this file for cluster‑specific runbooks.

## SLO (preview, canonical)

- p50 time_to_preview < 8s
- p95 time_to_preview < 30s
- proxy_cache_hit_rate > 70%

All SLOs assume: stable DEV deployment, N=10 warm runs, identical input dataset. See Benchmarking section for exact invocation.

## Metrics (recommended)

- montage_time_to_preview_seconds (Histogram)
  - buckets: [0.1, 0.5, 1, 2, 4, 8, 16, +Inf]
  - labels: {model, route, worker}

- montage_proxy_cache_request_total{status="hit|miss"}

- montage_preview_requests_total{status="ok|error"}

Prometheus-friendly examples (canonical):

- p95: histogram_quantile(0.95, sum(rate(montage_time_to_preview_seconds_bucket[5m])) by (le))

- proxy hit rate: sum(rate(montage_proxy_cache_request_total{status="hit"}[5m])) / sum(rate(montage_proxy_cache_request_total[5m]))

## Benchmark (developer)

1. Build image and push to dev registry (see CI/README in this repo).
2. Deploy to DEV (kustomize overlay):
   - `kubectl apply -k deploy/k8s/overlays/preview`
3. Run local benchmark script (10 runs, reproducible seed):
   - `./scripts/ci/preview-benchmark.sh BASE=https://preview.example.org RUNS=10`
4. Collect artifacts:
   - `/metrics` snapshot, pod logs (`kubectl logs`), and worker traces.

## Developer troubleshooting (app-owned)

- If time_to_preview high: check worker CPU/memory, model load times, and cache hit labels.
- If cache miss storm: review cache keys and proxy configuration.
- Attach `/metrics` snapshot + `kubectl describe pod` + `kubectl logs` to the PR/issue.

## Operator notes (short)

- The app should export the histogram and the two counters above; label with `route` and `model`.
- The Montage‑AI repo is the canonical location for API, examples, and developer troubleshooting.

## CHANGELOG (what to update when changing this doc)

- update `updated:` front-matter
- add short note in PR describing impact to infra (if any)

---

*To the montage‑ai maintainers*: this file is the canonical app-owned SLO and benchmark guide. Use it for `docs/operations/preview-slo.md` (mark canonical) and notify infra for any cluster-specific runbook links.
