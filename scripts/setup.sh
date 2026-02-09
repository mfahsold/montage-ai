#!/usr/bin/env bash
# Initialize montage-ai development environment with proper permissions
# Ensures data/ directories are correctly set up for Docker use

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Setting up montage-ai development environment..."

# Create data directories with correct permissions
echo "📂 Creating data directories..."
mkdir -p "$REPO_ROOT/data/input"
mkdir -p "$REPO_ROOT/data/music"
mkdir -p "$REPO_ROOT/data/output"
mkdir -p "$REPO_ROOT/data/assets"
mkdir -p "$REPO_ROOT/data/luts"

# On Linux, ensure directories are writable by the current user
# Docker will run as user 1000 (montage) inside the container
# If directories are owned by root or wrong user, we need to fix them

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CURRENT_USER=$(whoami)
    CURRENT_UID=$(id -u)
    
    echo "👤 Detected user: $CURRENT_USER (UID: $CURRENT_UID)"
    
    # Check if directories are owned by root and current user is not root
    if [ "$CURRENT_UID" != "0" ]; then
        for dir in input music output assets luts; do
            DIR_PATH="$REPO_ROOT/data/$dir"
            DIR_OWNER=$(stat -c "%U" "$DIR_PATH" 2>/dev/null || echo "unknown")
            
            if [ "$DIR_OWNER" == "root" ]; then
                echo "⚠️  $dir/ is owned by root. Attempting to fix permissions..."
                if sudo -n true 2>/dev/null; then
                    # Can use sudo without password
                    sudo chown -R "$CURRENT_USER:$CURRENT_USER" "$DIR_PATH"
                    echo "✅ Fixed ownership: $dir/ now owned by $CURRENT_USER"
                else
                    echo "❌ Cannot fix $dir/ ownership (requires root/sudo)"
                    cat > /tmp/docker-perm-fix.sh << 'EOF'
#!/bin/bash
# Run this with sudo to fix permissions:
# sudo bash /tmp/docker-perm-fix.sh
EOF
                    echo "sudo chown -R \$USER:\$USER $(dirname \"$DIR_PATH\")/data" >> /tmp/docker-perm-fix.sh
                    echo ""
                    echo "💡 To fix permissions, run:"
                    echo "   sudo chown -R \$USER:\$USER $REPO_ROOT/data"
                fi
            fi
        done
    fi
fi

# Verify Docker can access data directories
echo "🔍 Verifying Docker setup..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop or Docker Engine."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker || ! command -v docker-compose &> /dev/null; then
    if ! docker compose version &> /dev/null; then
        echo "⚠️  docker compose not found. Install Docker Compose v2."
    fi
fi

docker_version=$(docker --version | grep -oE "[0-9]+\.[0-9]+")
echo "  ✅ Docker $docker_version installed"

# Check disk space (portable across Linux and macOS)
DISK_AVAIL=$(df -k "$REPO_ROOT" | tail -1 | awk '{print int($4 / 1024 / 1024)}')
if [ "$DISK_AVAIL" -lt 30 ]; then
    echo "⚠️  Warning: Only ${DISK_AVAIL}GB free disk space. Videos need 30GB+."
fi

# Verify Docker resource allocation
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: Check Docker Desktop settings
    DOCKER_MEMORY=$(docker run --rm ubuntu:22.04 grep MemTotal /proc/meminfo 2>/dev/null | awk '{print int($2 / 1024 / 1024)}' || echo "unknown")
    echo "  Docker memory: ${DOCKER_MEMORY}GB (check Docker Desktop settings if too low)"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux: Check system memory and Docker memory (portable across Linux versions)
    SYSTEM_MEMORY=$(awk '/MemTotal/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
    echo "  System memory: ${SYSTEM_MEMORY}GB available"
fi

# ARM64 note about MediaPipe
MACHINE_ARCH="$(uname -m)"
if [[ "$MACHINE_ARCH" == "aarch64" || "$MACHINE_ARCH" == "arm64" ]]; then
    echo ""
    echo "ℹ️  ARM64 detected: MediaPipe (face detection for auto-reframe) may not be available."
    echo "   Auto-reframe will use center-crop fallback — this works fine for most content."
    echo "   See docs/OPTIONAL_DEPENDENCIES.md for details."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎬 Next steps:"
echo "   1. Generate test media:    ./scripts/ops/create-test-video.sh"
echo "   2. Build Docker image:     docker compose build"
echo "   3. Test setup:             docker compose run --rm montage-ai python -c 'import montage_ai; print(\"OK\")'  "
echo "   4. Run first montage:      QUALITY_PROFILE=preview docker compose run --rm montage-ai /app/montage-ai.sh run"
echo "   5. Or start Web UI:        docker compose up  (then http://localhost:8080)"
echo ""
