#!/bin/bash
# Quick setup for ARM64 (Snapdragon, Apple Silicon, etc.)
# Run this immediately after cloning the repository

set -e

echo "🚀 Montage AI - ARM64 Quick Setup"
echo "================================================================"
echo ""

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
    aarch64)
        echo "✅ Detected: ARM64 (aarch64) - Snapdragon, Apple Silicon, or similar"
        ;;
    arm64)
        echo "✅ Detected: Apple Silicon (arm64)"
        ;;
    *)
        echo "⚠️  Architecture: $ARCH"
        echo "   This script is optimized for ARM64 (aarch64)."
        echo "   It may not work correctly on other architectures."
        ;;
esac

echo ""
echo "Step 1: Checking prerequisites..."
echo "================================================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found! Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "✅ Docker: $DOCKER_VERSION"

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found!"
    echo "   Please ensure Docker Compose v2 is installed."
    exit 1
fi

echo "✅ Docker Compose: $(docker compose version --short)"

echo ""
echo "Step 2: Creating data directories..."
echo "================================================================"

mkdir -p data/input data/music data/output data/assets
echo "✅ Created: data/{input,music,output,assets}"

echo ""
echo "Step 3: Checking docker-compose.yml for ARM compatibility..."
echo "================================================================"

# Check current memory setting
CURRENT_MEMORY=$(grep -A 10 "resources:" docker-compose.yml | grep "memory:" | head -1 | awk '{print $2}' | tr -d '\n')
if [ -z "$CURRENT_MEMORY" ]; then
    CURRENT_MEMORY="12g (default)"
fi

echo "Current memory limit: $CURRENT_MEMORY"
echo ""
echo "Recommended memory settings for ARM64:"
echo "  - Snapdragon (12GB total): 8g"
echo "  - Apple Silicon M2 (16GB): 10g"
echo "  - Apple Silicon M1 (8GB): 6g"
echo ""
read -p "Press Enter to continue with current settings, or Ctrl+C to edit docker-compose.yml first"

echo ""
echo "Step 4: Building Docker image for ARM64..."
echo "================================================================"
echo ""
echo "⏳ This will take 2-3 minutes on first run..."
echo "   (Progress will be shown below)"
echo ""

if docker compose build 2>&1 | tail -30; then
    echo ""
    echo "✅ Docker image built successfully for ARM64"
else
    echo "❌ Build failed. Check output above."
    exit 1
fi

echo ""
echo "Step 5: Verifying Python environment..."
echo "================================================================"

if docker compose run --rm montage-ai python3 -c "import montage_ai; print('✅ Python ready')" 2>&1 | grep "✅"; then
    echo "✅ Python environment working"
else
    echo "❌ Python environment check failed"
    exit 1
fi

echo ""
echo "Step 6: Testing FFmpeg..."
echo "================================================================"

if docker compose run --rm montage-ai ffmpeg -version 2>&1 | head -1; then
    echo "✅ FFmpeg available"
else
    echo "⚠️  FFmpeg check inconclusive"
fi

echo ""
echo "================================================================"
echo "🎉 SETUP COMPLETE!"
echo "================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  Web UI (Recommended):"
echo "   docker compose up"
echo "   → Open http://localhost:8080 in your browser"
echo ""
echo "2️⃣  Command Line:"
echo "   # Add your media first:"
echo "   cp video.mp4 data/input/"
echo "   cp music.mp3 data/music/"
echo ""
echo "   # Run (preview mode, faster):"
echo "   QUALITY_PROFILE=preview docker compose run --rm montage-ai ./montage-ai.sh run"
echo ""
echo "   # Or full quality:"
echo "   docker compose run --rm montage-ai ./montage-ai.sh run"
echo ""
echo "3️⃣  Run Full Validation:"
echo "   ./scripts/validate-onboarding.sh"
echo ""
echo "Documentation: docs/getting-started-arm.md"
echo ""
