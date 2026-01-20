#!/usr/bin/env bash
set -euo pipefail
TAG=${1:-"dev-local"}

echo "Running dev autoscale smoke (image=${TAG})"
# ensure image is set
kubectl -n montage-ai set image deploy/montage-ai-web montage-ai=127.0.0.1:30500/montage-ai:${TAG} --record
kubectl -n montage-ai set image deploy/montage-ai-worker worker=127.0.0.1:30500/montage-ai:${TAG} --record

# rollout
kubectl -n montage-ai rollout restart deployment/montage-ai-web montage-ai-worker
kubectl -n montage-ai rollout status deploy/montage-ai-web --timeout=3m
kubectl -n montage-ai rollout status deploy/montage-ai-worker --timeout=3m

# quick health
kubectl -n montage-ai get pods -l app=montage-ai -o wide
kubectl -n montage-ai exec -it deploy/montage-ai-web -- curl -fsS -m 5 http://127.0.0.1:80/api/status | jq -C '.'

# run the repo's opt-in smoke (runner must have cluster networking or port-forward)
export RUN_SCALE_TESTS=1
pytest -q tests/integration/test_queue_scaling.py -q

echo "Dev autoscale smoke completed"
