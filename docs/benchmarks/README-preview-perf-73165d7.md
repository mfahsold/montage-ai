Preview SLO benchmark results — feature branch `perf/preview-proxy-cache-ttl` (commit 73165d7)

Summary:
- p50 (observed successful run): 1.67s
- p95 (observed successful run): 1.67s
- successful runs: 1/5 (API visibility flakiness caused several runs to be recorded as timeouts)

Key artifacts:
- `benchmark_results/preview-perf-73165d7.json` — structured summary
- `benchmark_results/preview-perf-73165d7.log` — raw output and worker log excerpts

Reproduction:
1. Copy benchmark clips onto the cluster PVC (`/data/input`) for the web pod.
2. Port-forward the web service: `kubectl -n montage-ai port-forward svc/montage-ai-web 8080:80`
3. Run: `SLO_P50=8 SLO_P95=30 ./scripts/ci/preview-benchmark.sh http://127.0.0.1:8080`

Follow-up (recommended):
- Make JobStore writes from workers retriable/synchronous (PR: follow-up).
- Harden `preview-benchmark.sh` to tolerate transient API staleness and add warm-up runs.
