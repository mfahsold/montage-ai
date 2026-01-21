#!/usr/bin/env bash
set -euo pipefail

# On-cluster preview benchmark (non-destructive). Intended for self-hosted runner.
# Exits non-zero when SLOs are violated.

BASE=${1:-http://127.0.0.1:18080}
SLO_P50=${SLO_P50:-8}   # seconds
SLO_P95=${SLO_P95:-30}  # seconds
RUNS=${RUNS:-5}

echo "Running on-cluster preview benchmark against $BASE (runs=$RUNS)"

TMP_RESULTS=$(mktemp)
trap 'rm -f "$TMP_RESULTS"' EXIT

for i in $(seq 1 $RUNS); do
  echo "Run $i"
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
    echo "{\"job\":\"$JOBID\",\"post\":$POST_TS,\"finish\":null}" >> "$TMP_RESULTS"
    # attempt to capture partial metrics/state
    curl -sS "$BASE/api/jobs/$JOBID" > "$TMP_RESULTS.job.$JOBID.json" || true
    curl -sS "$BASE/metrics" > "$TMP_RESULTS.metrics.run${i}.txt" || true
    continue
  fi
  echo "{\"job\":\"$JOBID\",\"post\":$POST_TS,\"finish\":$FINISH_TS}" >> "$TMP_RESULTS"

  # capture job details and metrics for post-run analysis
  curl -sS "$BASE/api/jobs/$JOBID" > "$TMP_RESULTS.job.$JOBID.json" || true
  curl -sS "$BASE/metrics" > "$TMP_RESULTS.metrics.run${i}.txt" || true
  sleep 1
done

# Compute deltas (seconds)
DELTAS=$(jq -r '[.[] | select(.finish != null) | ((.finish - .post)/1000.0)] | {count: length, p50: (sort | .[ (length/2) | floor ]), p95: (sort | .[ ( (length*95/100) | floor ) ]) }' "$TMP_RESULTS")
COUNT=$(echo "$DELTAS" | jq -r '.count')
P50=$(echo "$DELTAS" | jq -r '.p50')
P95=$(echo "$DELTAS" | jq -r '.p95')

echo "results: count=$COUNT p50=${P50}s p95=${P95}s"

if [ "$COUNT" -eq 0 ]; then
  echo "No successful runs recorded" >&2
  exit 2
fi

# Compare against SLOs
awk -v p50="$P50" -v p95="$P95" -v slo50="$SLO_P50" -v slo95="$SLO_P95" 'BEGIN{exit_code=0; if(p50>slo50){print "p50 exceeded: " p50 "s > " slo50 "s"; exit_code=1} if(p95>slo95){print "p95 exceeded: " p95 "s > " slo95 "s"; exit_code=1} exit exit_code }'

echo "Preview benchmark completed (SLOs: p50<${SLO_P50}s p95<${SLO_P95}s)"
exit 0
