#!/usr/bin/env bash
# Test that the Docker container starts successfully.
# Catches WORKDIR/CMD path issues, broken imports, and startup failures.
# Usage: ./scripts/ci-docker-startup.sh
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

TIMEOUT="${1:-30}"

cleanup() {
  echo "Cleaning up..."
  docker compose down 2>/dev/null || true
}
trap cleanup EXIT

echo "Docker container startup test"
echo "=============================="

# Clean state
docker compose down 2>/dev/null || true

# Build (use cache)
echo "Building image..."
docker compose build --quiet

# Start in background
echo "Starting container..."
docker compose up -d

# Wait for container to be running
echo "Waiting for container (max ${TIMEOUT}s)..."
for i in $(seq 1 "$TIMEOUT"); do
  if docker compose ps --format json 2>/dev/null | grep -q '"running"'; then
    echo -e "${GREEN}[OK]${NC} Container is running (${i}s)"
    break
  fi
  # Fallback for older docker compose without --format json
  if docker compose ps 2>/dev/null | grep -q "Up"; then
    echo -e "${GREEN}[OK]${NC} Container is running (${i}s)"
    break
  fi
  if [ "$i" -eq "$TIMEOUT" ]; then
    echo -e "${RED}[FAIL]${NC} Container did not start within ${TIMEOUT}s"
    docker compose logs --tail=30
    exit 1
  fi
  sleep 1
done

# Check that container didn't exit immediately
sleep 3
if docker compose ps 2>/dev/null | grep -qE "(Exit|exited)"; then
  echo -e "${RED}[FAIL]${NC} Container exited after startup"
  docker compose logs --tail=30
  exit 1
fi

# Test Web UI endpoint
echo "Testing Web UI endpoint..."
sleep 5
if curl -f -s --max-time 10 http://localhost:${WEB_PORT:-8080} >/dev/null 2>&1; then
  echo -e "${GREEN}[OK]${NC} Web UI responding at http://localhost:${WEB_PORT:-8080}"
else
  echo -e "${RED}[WARN]${NC} Web UI not responding (may still be initializing)"
fi

echo ""
echo -e "${GREEN}Docker startup test passed.${NC}"
