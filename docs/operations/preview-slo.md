---
title: "Montage-AI -- Preview SLO (canonical)"
summary: "Canonical SLOs, benchmark, and verification instructions for the Montage-AI preview fast-path. This file is the canonical source in the montage-ai repo."
updated: 2026-01-22
---

## Purpose

Canonical, developer-facing specification for Montage-AI preview SLOs, benchmark procedure, metrics schema, and guidance for reproducing and debugging failures. Infra runbooks should link here and keep cluster-specific steps elsewhere.

## SLO (preview, canonical)

- p50 time_to_preview < 8s
- p95 time_to_preview < 30s
- proxy_cache_hit_rate > 70%

All SLOs assume: stable DEV deployment, N=10 warm runs, identical input dataset.

## Metrics (recommended)

- montage_time_to_preview_seconds (Histogram)
  - buckets: [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, +Inf]
- montage_proxy_cache_request_total
- montage_proxy_cache_hit_total
- montage_preview_requests_total

Prometheus examples (canonical):

- p95: histogram_quantile(0.95, sum(rate(montage_time_to_preview_seconds_bucket[5m])) by (le))
- proxy hit rate: sum(rate(montage_proxy_cache_hit_total[5m])) / sum(rate(montage_proxy_cache_request_total[5m]))

## Benchmark (developer)

1. Build image and push to the dev registry (see CI/README in this repo).
2. Deploy to cluster (canonical overlay):
   - kubectl apply -k deploy/k3s/overlays/cluster/
3. Run local benchmark (10 runs, reproducible seed):
   - ./scripts/ci/preview-benchmark.sh BASE=https://preview.example.org RUNS=10 --collect-metrics
4. Collect artifacts:
   - /metrics snapshots, job JSONs, pod logs (kubectl logs).

Benchmark script options:
- Override thresholds: SLO_P50=<seconds> SLO_P95=<seconds>
- Save artifacts to a specific folder: --out <dir> (implies --collect-metrics)

## Developer troubleshooting (app-owned)

- If time_to_preview is high: check worker CPU/memory, model load times, and cache hit ratios.
- If cache miss storm: review cache keys and proxy configuration.
- For reproducible CI runs: set FFMPEG_HWACCEL=none and MONTAGE_PREVIEW_CPU_LIMIT=<n>.
- Attach /metrics snapshots, kubectl describe pod, and worker logs to the issue/PR.

## Operator notes (short)

- The app should export the histogram and counters above.
- Keep this file canonical in the montage-ai repo; infra runbooks should link here.

## CHANGELOG (what to update when changing this doc)

- update updated: front-matter
- add a short note in the PR describing infra impact (if any)
