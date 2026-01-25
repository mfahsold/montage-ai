Preview SLO benchmark results — feature branch `perf/preview-proxy-cache-ttl` (commit 73165d7)

Summary:
- p50 (observed successful run): 1.67s
- p95 (observed successful run): 1.67s
- successful runs: 1/5 (API visibility flakiness caused several runs to be recorded as timeouts)

Key artifacts:
- `benchmark_results/preview-perf-73165d7.json` — structured summary
- `benchmark_results/preview-perf-73165d7.log` — raw output and worker log excerpts

Reproduction:
- Follow the canonical preview SLO doc: docs/operations/preview-slo.md (use this commit/branch).

Follow-up (recommended):
- Make JobStore writes from workers retriable/synchronous (PR: follow-up).
- Re-run with current `preview-benchmark.sh` and archive artifacts via --collect-metrics.
