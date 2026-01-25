#!/usr/bin/env bash
set -euo pipefail

# On-cluster preview benchmark (non-destructive). Intended for self-hosted runner.
# Exits non-zero when SLOs are violated.

usage() {
  cat <<'EOF'
Usage: preview-benchmark.sh [options] [BASE_URL]

Options:
  --collect-metrics   Save /metrics snapshots and job JSON to disk
  --out DIR           Output directory for artifacts (implies --collect-metrics)
  -h, --help          Show this help

Environment:
  BASE, RUNS, SLO_P50, SLO_P95, COLLECT_METRICS, OUT_DIR
EOF
}

BASE_ARG=""
COLLECT_METRICS=${COLLECT_METRICS:-0}
OUT_DIR=${OUT_DIR:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collect-metrics)
      COLLECT_METRICS=1
      ;;
    --out)
      OUT_DIR="${2:-}"
      if [[ -z "$OUT_DIR" ]]; then
        echo "Missing value for --out" >&2
        exit 2
      fi
      COLLECT_METRICS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "$BASE_ARG" ]]; then
        BASE_ARG="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
  shift
done

BASE=${BASE:-${BASE_ARG:-http://127.0.0.1:18080}}
SLO_P50=${SLO_P50:-8}   # seconds
SLO_P95=${SLO_P95:-30}  # seconds
RUNS=${RUNS:-5}

if [[ "$COLLECT_METRICS" -eq 1 ]]; then
  if [[ -z "$OUT_DIR" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    OUT_DIR="${REPO_ROOT}/benchmark_results/preview-benchmark-$(date +%Y%m%d-%H%M%S)"
  fi
  mkdir -p "$OUT_DIR"
  RESULTS_FILE="${OUT_DIR}/results.jsonl"
  echo "Collecting artifacts in $OUT_DIR"
else
  OUT_DIR="$(mktemp -d)"
  RESULTS_FILE="${OUT_DIR}/results.jsonl"
  trap 'rm -rf "$OUT_DIR"' EXIT
fi

echo "Running on-cluster preview benchmark against $BASE (runs=$RUNS)"
: > "$RESULTS_FILE"

for i in $(seq 1 $RUNS); do
  echo "Run $i"
  FINISH_TS=""
  RESP=$(curl -sS -X POST "$BASE/api/jobs" -H 'Content-Type: application/json' -d '{"style":"dynamic","options":{"video_files":["benchmark_clip_a.mp4","benchmark_clip_b.mp4"]},"quality_profile":"preview"}')
  JOBID=$(echo "$RESP" | jq -r '.id // .job_id // .jobId')
  if [ -z "$JOBID" ] || [ "$JOBID" = "null" ]; then
    echo "Failed to create job: $RESP" >&2
    exit 2
  fi

  POST_TS=$(date +%s%3N)
  DEADLINE=$(( $(date +%s) + 90 ))
  got=0
  while [ $(date +%s) -lt $DEADLINE ]; do
    STATUS=$(curl -sS "$BASE/api/jobs/$JOBID" | jq -r '.status // "unknown"') || STATUS=unknown
    if [ "$STATUS" = "finished" ] || [ "$STATUS" = "completed" ]; then
      FINISH_TS=$(date +%s%3N)
      echo "  job $JOBID finished"
      break
    fi
    sleep 1
  done
  if [ -z "${FINISH_TS:-}" ]; then
    echo "Job $JOBID did not finish within timeout" >&2
    echo "{\"job\":\"$JOBID\",\"post\":$POST_TS,\"finish\":null}" >> "$RESULTS_FILE"
    if [[ "$COLLECT_METRICS" -eq 1 ]]; then
      # attempt to capture partial metrics/state
      curl -sS "$BASE/api/jobs/$JOBID" > "$OUT_DIR/job.$JOBID.json" || true
      curl -sS "$BASE/metrics" > "$OUT_DIR/metrics.run${i}.txt" || true
    fi
    continue
  fi
  echo "{\"job\":\"$JOBID\",\"post\":$POST_TS,\"finish\":$FINISH_TS}" >> "$RESULTS_FILE"

  if [[ "$COLLECT_METRICS" -eq 1 ]]; then
    # capture job details and metrics for post-run analysis
    curl -sS "$BASE/api/jobs/$JOBID" > "$OUT_DIR/job.$JOBID.json" || true
    curl -sS "$BASE/metrics" > "$OUT_DIR/metrics.run${i}.txt" || true
  fi
  sleep 1
done

# Compute deltas (seconds)
DELTAS=$(jq -s '[.[] | select(.finish != null) | ((.finish - .post)/1000.0)] | {count: length, p50: (sort | .[ (length/2) | floor ]), p95: (sort | .[ ( (length*95/100) | floor ) ]) }' "$RESULTS_FILE")
COUNT=$(echo "$DELTAS" | jq -r '.count')
P50=$(echo "$DELTAS" | jq -r '.p50')
P95=$(echo "$DELTAS" | jq -r '.p95')

echo "results: count=$COUNT p50=${P50}s p95=${P95}s"

if [ "$COUNT" -eq 0 ]; then
  echo "No successful runs recorded" >&2
  exit 2
fi

if [[ "$COLLECT_METRICS" -eq 1 ]]; then
  echo "$DELTAS" > "$OUT_DIR/summary.json"
fi

# Compare against SLOs
awk -v p50="$P50" -v p95="$P95" -v slo50="$SLO_P50" -v slo95="$SLO_P95" 'BEGIN{exit_code=0; if(p50>slo50){print "p50 exceeded: " p50 "s > " slo50 "s"; exit_code=1} if(p95>slo95){print "p95 exceeded: " p95 "s > " slo95 "s"; exit_code=1} exit exit_code }'

echo "Preview benchmark completed (SLOs: p50<${SLO_P50}s p95<${SLO_P95}s)"
if [[ "$COLLECT_METRICS" -eq 1 ]]; then
  echo "Artifacts saved to $OUT_DIR"
fi
exit 0
