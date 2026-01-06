#!/bin/bash
set -euo pipefail

SCRIPT_TEXT="Montage AI cuts raw footage into a story. It syncs beats, finds moments, and exports to your NLE. Privacy-first. Built for creators."

echo "Preparing public-domain assets..."
python3 scripts/prepare_trailer_assets.py --videos 4 --duration 30

echo "Planning B-roll..."
python3 -m montage_ai.broll_planner "$SCRIPT_TEXT" || true

if command -v cgpu >/dev/null 2>&1; then
  export CGPU_ENABLED=true
  export CGPU_GPU_ENABLED=true
  export CGPU_MAX_CONCURRENCY=${CGPU_MAX_CONCURRENCY:-2}
else
  echo "cgpu not found; using local GPU/CPU only."
  export CGPU_GPU_ENABLED=false
fi

export UPSCALE=true
export UPSCALE_MODEL=${UPSCALE_MODEL:-realesrgan-x4plus}
export UPSCALE_FRAME_FORMAT=${UPSCALE_FRAME_FORMAT:-png}
export UPSCALE_CRF=${UPSCALE_CRF:-16}
export QUALITY_PROFILE=${QUALITY_PROFILE:-high}
export TARGET_DURATION=${TARGET_DURATION:-30}

echo "Running trailer render..."
./montage-ai.sh run documentary
