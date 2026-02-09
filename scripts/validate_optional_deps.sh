#!/usr/bin/env bash
# Validate optional dependencies for Montage AI features.
# Usage: ./scripts/validate_optional_deps.sh [--strict]
set -euo pipefail

STRICT="${1:-}"
TOTAL=0
FOUND=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_python_module() {
  local module="$1"
  local feature="$2"
  TOTAL=$((TOTAL + 1))
  if python3 -c "import $module" 2>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} $module — $feature"
    FOUND=$((FOUND + 1))
    return 0
  else
    echo -e "  ${RED}[--]${NC} $module — $feature (not installed)"
    return 1
  fi
}

check_command() {
  local cmd="$1"
  local feature="$2"
  TOTAL=$((TOTAL + 1))
  if command -v "$cmd" &>/dev/null; then
    local ver
    ver=$("$cmd" -version 2>&1 | head -1 || echo "unknown")
    echo -e "  ${GREEN}[OK]${NC} $cmd — $feature"
    FOUND=$((FOUND + 1))
    return 0
  else
    echo -e "  ${RED}[--]${NC} $cmd — $feature (not found)"
    return 1
  fi
}

echo "Montage AI — Optional Dependencies Check"
echo "========================================="

echo ""
echo "System Tools:"
check_command "ffmpeg" "Video processing (required)" || true
check_command "ffprobe" "Media analysis (required)" || true

echo ""
echo "Core Python:"
check_python_module "montage_ai" "Montage AI package" || true
check_python_module "moviepy" "Video composition" || true
check_python_module "cv2" "OpenCV (scene detection, stabilization)" || true
check_python_module "numpy" "Numerical processing" || true
check_python_module "PIL" "Image processing (Pillow)" || true

echo ""
echo "AI / ML (pip install montage-ai[ai]):"
check_python_module "mediapipe" "Smart Reframing (face detection)" || true
check_python_module "scipy" "Path optimization (camera motion)" || true
check_python_module "librosa" "Advanced beat detection (FFmpeg is primary)" || true
check_python_module "color_matcher" "Shot-to-shot color consistency" || true

echo ""
echo "Web UI (pip install montage-ai[web]):"
check_python_module "flask" "Web framework" || true
check_python_module "redis" "Job queue backend" || true
check_python_module "rq" "Background job processing" || true

echo ""
echo "Cloud / Integrations:"
check_python_module "openai" "OpenAI-compatible LLM API" || true
check_python_module "soundfile" "Cloud audio handling" || true
check_python_module "torch" "PyTorch (voice isolation, advanced audio)" || true
check_python_module "webrtcvad" "Voice activity detection" || true

echo ""
echo "========================================="
echo -e "Result: ${FOUND}/${TOTAL} dependencies available"
echo ""

if [ "$FOUND" -lt "$TOTAL" ]; then
  MISSING=$((TOTAL - FOUND))
  echo -e "${YELLOW}${MISSING} optional dependencies not installed.${NC}"
  echo "Install all optional deps: pip install montage-ai[all]"
  echo "Or selectively: pip install montage-ai[ai], pip install montage-ai[web]"
  if [ "$STRICT" = "--strict" ]; then
    exit 1
  fi
else
  echo -e "${GREEN}All dependencies available.${NC}"
fi
