# Portable Development Guide

**Purpose:** Fast, offline-capable development workflow for Montage AI  
**Use Case:** Working "unterwegs" (on the go), mobile development, no cluster access  
**Philosophy:** "Edit → Test → Ship" cycle in under 60 seconds

---

## Quick Start

### Option 1: Local Filesystem Cache (Fastest)

```bash
# Build with local cache (single-platform, fast iteration)
./scripts/build_local_cache.sh

# Or with custom settings
CACHE_DIR=/tmp/buildx-cache TAG=dev ./scripts/build_local_cache.sh
```

**Performance:**
- **First Build:** 8-12 min (single-arch, amd64 only)
- **Code Changes:** ~1 min (cached base layers)
- **No Changes:** ~15 sec (full cache hit)

### Option 2: Docker Compose (No Build Required)

```bash
# Run with volume mounts for live code reload
docker-compose up

# Or web UI only
docker-compose -f docker-compose.web.yml up
```

**Advantages:**
- No build step (uses pre-built image or local cache)
- Live code reload via volume mounts
- Instant feedback for Python changes

### Option 3: Development Container (VS Code)

```bash
# Open in VS Code Dev Container
code .

# VS Code will prompt to "Reopen in Container"
# Or: Cmd/Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

**Advantages:**
- Consistent environment across machines
- No local Python/ffmpeg installation
- Integrated debugging

---

## Local Build Script (`build_local_cache.sh`)

### Usage

```bash
# Basic build (defaults to linux/amd64)
./scripts/build_local_cache.sh

# Custom cache directory
CACHE_DIR=$HOME/.montage-cache ./scripts/build_local_cache.sh

# Custom tag
TAG=my-feature ./scripts/build_local_cache.sh

# Load to local Docker daemon (single-platform only)
LOAD=true ./scripts/build_local_cache.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_DIR` | `/tmp/buildx-cache` | Local cache directory path |
| `TAG` | `latest` | Image tag (e.g., `dev`, `v1.2.3`) |
| `PLATFORM` | `linux/amd64` | Target platform (single-arch) |
| `BUILDER` | `multiarch-builder` | Buildx builder name |
| `LOAD` | `false` | Load image to local Docker daemon |

### Cache Management

```bash
# Check cache size
du -sh /tmp/buildx-cache

# Clear cache (forces fresh build)
rm -rf /tmp/buildx-cache

# Prune old cache layers (Docker will auto-clean old data)
docker buildx prune
```

---

## Development Workflows

### Workflow A: Fast Iteration (Volume Mounts)

**Best for:** Frequent Python code changes, testing small edits

1. **Initial Setup:**
   ```bash
   # Build once with local cache
   ./scripts/build_local_cache.sh
   ```

2. **Edit Code:**
   ```bash
   # Edit files in src/montage_ai/
   code src/montage_ai/core/montage_builder.py
   ```

3. **Test Immediately:**
   ```bash
   # Run with volume mount (no rebuild required)
   docker run --rm -it \
     -v $(pwd)/src:/app/src \
     -v $(pwd)/data:/data \
     montage-ai:latest \
     montage-ai run --style dynamic
   ```

**Time:** ~5 seconds from edit to test

### Workflow B: Full Rebuild (Dockerfile Changes)

**Best for:** Dependency updates, system package changes, Dockerfile edits

1. **Edit Dependencies:**
   ```bash
   # Update requirements.txt or Dockerfile
   echo "new-package==1.0.0" >> requirements.txt
   ```

2. **Rebuild with Cache:**
   ```bash
   # Cache ensures fast rebuild (only changed layers)
   ./scripts/build_local_cache.sh
   ```

3. **Test:**
   ```bash
   docker run --rm -it montage-ai:latest montage-ai --version
   ```

**Time:** ~1-2 min for Python dependency changes

### Workflow C: Docker Compose Development

**Best for:** Web UI development, multi-service testing

1. **Start Services:**
   ```bash
   # Starts montage-ai + web UI + cgpu-server
   docker-compose up -d
   ```

2. **Watch Logs:**
   ```bash
   docker-compose logs -f montage-ai
   ```

3. **Make Changes:**
   ```bash
   # Edit files, then restart service
   docker-compose restart montage-ai
   ```

4. **Access Web UI:**
   ```
   http://localhost:5001
   ```

**Time:** ~10 seconds to restart after code change

---

## Offline Development

### Prepare for Offline Work

```bash
# 1. Build and cache all layers
./scripts/build_local_cache.sh

# 2. Pull base image
docker pull continuumio/miniconda3:latest

# 3. Verify cache exists
ls -lh /tmp/buildx-cache

# 4. Export cache (optional, for backup)
tar czf montage-cache-backup.tar.gz /tmp/buildx-cache
```

### Work Offline

```bash
# All layers cached, no network required
CACHE_DIR=/tmp/buildx-cache ./scripts/build_local_cache.sh

# Or use pre-built image with volume mounts
docker run --rm -it -v $(pwd)/src:/app/src montage-ai:latest bash
```

---

## IDE Integration

### VS Code Dev Container

**Setup:**

1. **Create `.devcontainer/devcontainer.json`:**
   ```json
   {
     "name": "Montage AI Dev",
     "dockerComposeFile": "../docker-compose.yml",
     "service": "montage-ai",
     "workspaceFolder": "/app",
     "customizations": {
       "vscode": {
         "extensions": [
           "ms-python.python",
           "ms-python.vscode-pylance",
           "GitHub.copilot"
         ],
         "settings": {
           "python.defaultInterpreterPath": "/opt/conda/bin/python"
         }
       }
     },
     "forwardPorts": [5001],
     "postCreateCommand": "pip install -e ."
   }
   ```

2. **Reopen in Container:**
   - `Cmd/Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
   - VS Code will build/start container automatically

3. **Develop Natively:**
   - Edit files in VS Code (inside container)
   - Run tests: `pytest tests/`
   - Debug: Set breakpoints, press F5

### PyCharm Remote Interpreter

**Setup:**

1. **Build Docker Image:**
   ```bash
   ./scripts/build_local_cache.sh
   ```

2. **Configure Remote Interpreter:**
   - Settings → Project → Python Interpreter
   - Add Interpreter → Docker
   - Select `montage-ai:latest` image
   - Set path: `/opt/conda/bin/python`

3. **Run Configurations:**
   - Edit Configurations → Add Docker
   - Command: `montage-ai run --style dynamic`
   - Volume bindings: `src:/app/src`, `data:/data`

---

## Performance Comparison

| Method | Build Time | Edit-Test Cycle | Network Required | Cache Persistent |
|--------|------------|-----------------|------------------|------------------|
| **Local Cache** | 8-12 min | ~1 min | First build only | Yes (filesystem) |
| **Volume Mounts** | One-time | ~5 sec | No | N/A |
| **Docker Compose** | One-time | ~10 sec | No | N/A |
| **Dev Container** | One-time | ~5 sec | No | Yes (container) |
| **Cluster Registry** | 15-20 min | ~2 min | Always | Yes (registry) |

---

## Troubleshooting

### Issue: Cache Not Working

**Symptom:** Every build starts from scratch

**Solution:**
```bash
# Verify cache exists
ls -lh /tmp/buildx-cache

# Check buildx builder
docker buildx ls

# Rebuild cache
rm -rf /tmp/buildx-cache
./scripts/build_local_cache.sh
```

### Issue: Volume Mounts Not Reflecting Changes

**Symptom:** Code edits don't appear in running container

**Solution:**
```bash
# Restart container (not rebuild)
docker-compose restart montage-ai

# Or force recreate
docker-compose up -d --force-recreate

# Verify mount:
docker exec -it montage-ai-container ls -l /app/src
```

### Issue: Slow Builds on Laptop

**Symptom:** Builds take 20+ minutes

**Solution:**
```bash
# Use single-platform (amd64 only)
PLATFORM=linux/amd64 ./scripts/build_local_cache.sh

# Reduce Docker resource limits
# Docker Desktop → Settings → Resources → Decrease CPU/Memory

# Or use pre-built image with volume mounts (no build)
docker run -v $(pwd)/src:/app/src montage-ai:latest
```

### Issue: "No Space Left on Device"

**Symptom:** Build fails with disk space error

**Solution:**
```bash
# Clean Docker build cache
docker buildx prune -af

# Remove unused images
docker system prune -a

# Change cache location
CACHE_DIR=$HOME/.montage-cache ./scripts/build_local_cache.sh
```

---

## Makefile Shortcuts (Coming Soon)

```makefile
# Quick local development build
dev:
	./scripts/build_local_cache.sh

# Run with volume mounts (no rebuild)
dev-run:
	docker run --rm -it -v $(PWD)/src:/app/src montage-ai:latest bash

# Start Docker Compose stack
dev-up:
	docker-compose up -d

# Tail logs
dev-logs:
	docker-compose logs -f montage-ai

# Clean cache
dev-clean:
	rm -rf /tmp/buildx-cache
	docker buildx prune -f
```

**Usage:**
```bash
make dev          # Build with cache
make dev-run      # Run with volume mounts
make dev-up       # Start Compose stack
```

---

## Best Practices

1. **Use volume mounts** for Python code changes (instant feedback)
2. **Use local cache** for Dockerfile/dependency changes (fast rebuild)
3. **Use Docker Compose** for web UI development (multi-service)
4. **Use Dev Containers** for consistent environment (team development)
5. **Prepare offline** before traveling (cache layers + backup)

---

## Related Documentation

- [Cluster Workflow](../deploy/CLUSTER_WORKFLOW.md) - Multi-arch cluster builds
- [Build Caching](build-caching.md) - Layer optimization details
- [Getting Started](getting-started.md) - Installation guide
- [Configuration](configuration.md) - Environment variables

---

**Last Updated:** January 5, 2026  
**Maintained By:** Montage AI Team
