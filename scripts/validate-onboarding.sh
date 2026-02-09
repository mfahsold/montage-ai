#!/bin/bash
# Technical onboarding validation script for montage-ai
# Tests all prerequisites and runs first montage on any architecture (including ARM)

set -e

echo "🔍 Montage AI - Technical Onboarding Validation"
echo "================================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for checks
PASSED=0
FAILED=0
WARNINGS=0

check_passed() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED++))
}

check_failed() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED++))
    if [ -n "$2" ]; then
        echo -e "${RED}   Error: $2${NC}"
    fi
}

check_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

# ===== SYSTEM REQUIREMENTS =====
echo "📋 SYSTEM REQUIREMENTS"
echo "================================================================"

# 1. Docker version
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [[ "$DOCKER_VERSION" > "20.10" ]] || [[ "$DOCKER_VERSION" == "20.10" ]]; then
        check_passed "Docker installed: $DOCKER_VERSION"
    else
        check_failed "Docker version too old: $DOCKER_VERSION (need >= 20.10)"
    fi
else
    check_failed "Docker not found. Install from https://docs.docker.com/get-docker/"
fi

# 2. Docker Compose version
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "unknown")
    if [[ "$COMPOSE_VERSION" > "v2.0" ]] || [[ "$COMPOSE_VERSION" == "v2.0" ]]; then
        check_passed "Docker Compose installed: $COMPOSE_VERSION"
    else
        check_failed "Docker Compose version too old: $COMPOSE_VERSION (need >= v2.0)"
    fi
else
    check_failed "Docker Compose not found or not working"
fi

echo ""

# ===== HARDWARE CHECKS =====
echo "🖥️  HARDWARE REQUIREMENTS"
echo "================================================================"

# 1. RAM
if [ -f /proc/meminfo ]; then
    RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    RAM_GB=$((RAM_KB / 1024 / 1024))
else
    RAM_GB=$(sysctl hw.memsize 2>/dev/null | awk '{print $3 / 1024 / 1024 / 1024}' || echo "unknown")
fi

if [ -n "$RAM_GB" ] && [ "$RAM_GB" -ge 16 ]; then
    check_passed "RAM: $RAM_GB GB (sufficient for high quality)"
elif [ -n "$RAM_GB" ] && [ "$RAM_GB" -ge 8 ]; then
    check_warning "RAM: $RAM_GB GB (minimum met, use QUALITY_PROFILE=preview for better performance)"
else
    check_failed "RAM: $RAM_GB GB (minimum 8 GB required, 16 GB recommended)"
fi

# 2. CPU cores
if [ -f /proc/cpuinfo ]; then
    CPU_CORES=$(grep -c "^processor" /proc/cpuinfo)
else
    CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
fi

if [ -n "$CPU_CORES" ] && [ "$CPU_CORES" -ge 4 ]; then
    check_passed "CPU cores: $CPU_CORES (sufficient)"
elif [ -n "$CPU_CORES" ] && [ "$CPU_CORES" -ge 2 ]; then
    check_warning "CPU cores: $CPU_CORES (minimum met, will be slower)"
else
    check_failed "CPU cores: $CPU_CORES (minimum 2 cores required, 4+ recommended)"
fi

# 3. Disk space
DISK_AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ -n "$DISK_AVAILABLE" ] && [ "$DISK_AVAILABLE" -ge 10 ]; then
    check_passed "Disk space: ${DISK_AVAILABLE}GB available (sufficient)"
elif [ -n "$DISK_AVAILABLE" ] && [ "$DISK_AVAILABLE" -ge 5 ]; then
    check_warning "Disk space: ${DISK_AVAILABLE}GB available (low, may fill up with outputs)"
else
    check_failed "Disk space: ${DISK_AVAILABLE}GB available (minimum 10 GB recommended)"
fi

# 4. Architecture
if [ -f /proc/cpuinfo ]; then
    ARCH=$(uname -m)
else
    ARCH=$(uname -m)
fi
check_passed "Architecture: $ARCH"

echo ""

# ===== ENVIRONMENT SETUP =====
echo "📂 ENVIRONMENT SETUP"
echo "================================================================"

if [ ! -d "src/montage_ai" ]; then
    check_failed "Not in montage-ai repository root (src/montage_ai not found)"
    echo ""
    echo "Fix: cd to montage-ai repository directory first:"
    echo "  cd /path/to/montage-ai"
    exit 1
else
    check_passed "In montage-ai repository root"
fi

# Create data directories
if mkdir -p data/input data/music data/output data/assets 2>/dev/null; then
    check_passed "Data directories created (data/{input,music,output,assets})"
else
    check_failed "Could not create data directories"
fi

echo ""

# ===== DOCKER BUILD TEST =====
echo "🐳 DOCKER BUILD TEST"
echo "================================================================"
echo ""
echo "Building Docker image (this takes 2-3 minutes on first run)..."
echo "Architecture: $ARCH"

if docker compose build --no-cache 2>&1 | tail -20; then
    check_passed "Docker image built successfully"
else
    check_failed "Docker image build failed"
    exit 1
fi

echo ""

# ===== PYTHON IMPORT TEST =====
echo "🐍 PYTHON IMPORT TEST"
echo "================================================================"

if docker compose run --rm montage-ai python3 -c "import montage_ai; print('  ✅ montage_ai module imports')" 2>&1; then
    check_passed "Python module imports successfully"
else
    check_failed "Python module import failed"
fi

# Check key dependencies
if docker compose run --rm montage-ai python3 -c "import cv2, moviepy, librosa; print('  ✅ Key dependencies available')" 2>&1; then
    check_passed "Key dependencies (OpenCV, MoviePy, Librosa) available"
else
    check_failed "Key dependencies missing"
fi

echo ""

# ===== FFMPEG TEST =====
echo "🎬 FFMPEG TEST"
echo "================================================================"

if docker compose run --rm montage-ai ffmpeg -version 2>&1 | head -3; then
    check_passed "FFmpeg available in container"
else
    check_failed "FFmpeg not found in container"
fi

echo ""

# ===== FIRST MONTAGE TEST (PREVIEW) =====
echo "🎥 FIRST MONTAGE TEST (PREVIEW MODE)"
echo "================================================================"
echo ""

# Check if test data exists
if [ ! -f test_data/sample_video.mp4 ]; then
    echo "ℹ️  Creating synthetic test video (15 seconds)..."
    mkdir -p test_data
    if docker compose run --rm montage-ai \
        ffmpeg -f lavfi -i color=c=blue:s=320x240:d=5 \
                -f lavfi -i color=c=red:s=320x240:d=5 \
                -f lavfi -i color=c=green:s=320x240:d=5 \
                -filter_complex concat=n=3:v=1:a=0 \
                -y test_data/sample_video.mp4 2>&1 | grep -E "(error|Error|ERROR)" && \
        check_failed "Could not create test video"; then
        check_passed "Test video created"
    fi
fi

if [ -f test_data/sample_video.mp4 ]; then
    echo "✅ Test video ready"
    echo ""
    echo "Running preview render (this may take 30-60 seconds)..."
    echo ""
    
    if QUALITY_PROFILE=preview docker compose run --rm montage-ai \
        ./montage-ai.sh run 2>&1 | tail -30; then
        check_passed "Preview render completed"
    else
        check_warning "Preview render encountered issues (check logs above)"
    fi
else
    check_warning "Could not create test video, skipping render test"
fi

echo ""

# ===== RESULTS SUMMARY =====
echo "================================================================"
echo "📊 VALIDATION SUMMARY"
echo "================================================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo -e "${GREEN}Failed: $FAILED${NC}"
fi

echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Add your media files to data/input/ and data/music/"
    echo "2. Start the Web UI:   docker compose up"
    echo "3. Open http://localhost:8080 in your browser"
    echo "4. Or use CLI:         docker compose run --rm montage-ai ./montage-ai.sh run"
    echo ""
    echo "Documentation:"
    echo "  - Getting started: docs/getting-started.md"
    echo "  - Configuration: docs/configuration.md"
    echo "  - Troubleshooting: docs/troubleshooting.md"
    exit 0
else
    echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
    echo ""
    echo "Please fix the errors above and try again."
    echo "See docs/troubleshooting.md for common issues."
    exit 1
fi
