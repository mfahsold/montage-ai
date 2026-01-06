# Developer Experience (DX) - Golden Path

**Philosophy:** Two workflows. No choices. Maximum productivity.

---

## 🚀 The Golden Path

### Local Development → `make dev`
**When:** Coding, testing, debugging  
**Speed:** 5 seconds from edit to test  
**Network:** Offline-capable

```bash
# 1. One-time setup (builds base image)
make dev

# 2. Code → Test cycle (instant)
# Edit files in src/, then:
make dev-test

# That's it.
```

**How it works:**
- Base image built once with local cache (`/tmp/buildx-cache`)
- Volume mounts: `src/` → container
- Changes reflected instantly (no rebuild)
- Full offline support

---

### Cluster Deployment → `make cluster`
**When:** Deploying to production/staging  
**Speed:** 2 min incremental, 15 min cold  
**Network:** Requires cluster access

```bash
# Build + push + deploy in one command
make cluster

# Or step by step:
make cluster-build   # Multi-arch with registry cache
make cluster-deploy  # Update K8s deployment
```

**How it works:**
- Multi-arch build (amd64 + arm64)
- Registry cache shared across cluster nodes
- Automatic deployment to K8s
- Zero-downtime rollout

---

## 📋 Complete Command Reference

```bash
# === LOCAL DEVELOPMENT ===
make dev          # Build base image with cache (once)
make dev-test     # Run with volume mounts (instant)
make dev-shell    # Interactive shell for debugging

# === CLUSTER DEPLOYMENT ===
make cluster      # Build + deploy (all-in-one)

# === MAINTENANCE ===
make clean        # Clean local caches
make help         # Show all commands
```

That's the entire API. **4 commands for 90% of work.**

---

## 🎯 Decision Tree

```
Are you writing code?
├─ YES → make dev-test (5 sec feedback)
└─ NO  → Are you deploying?
         ├─ YES → make cluster (2-15 min)
         └─ NO  → What are you doing? 🤔
```

---

## 🔧 Under the Hood

### Local Development Stack
```
Code Editor
    ↓
src/ (volume mount)
    ↓
Docker Container (montage-ai:dev)
    ↓
Instant Execution (no build)
```

**Files:**
- `Makefile` - Entry points
- `scripts/build_local_cache.sh` - Caching logic
- `docker-compose.yml` - Volume mount config

### Cluster Deployment Stack
```
Local Machine
    ↓
Docker Buildx (multi-arch)
    ↓
Cluster Registry (10.43.17.166:5000)
    ↓
Kubernetes (6 nodes)
```

**Files:**
- `scripts/build_with_cache.sh` - Registry cache
- `deploy/k3s/` - K8s manifests

---

## 🐛 Troubleshooting

### "make dev is slow"
**Expected:** 8-12 min first time, ~1 min subsequent
**If slower:** Check Docker resource limits (Settings → Resources)

### "make dev-test doesn't see my changes"
**Check:** Files are in `src/` directory (not root)
**Fix:** Restart container: `make dev-test` (doesn't rebuild)

### "make cluster fails with registry timeout"
**Expected:** Cache import may fail, build continues
**Impact:** Minimal (BuildKit caches locally)
**Fix:** See [CLUSTER_WORKFLOW.md](../deploy/CLUSTER_WORKFLOW.md#issue-registry-cache-import-timeout)

### Offline Development

**Prepare before going offline:**
```bash
# Build and cache everything
make dev

# Verify cache exists
ls -lh /tmp/buildx-cache
```

**Work offline:**
```bash
# All layers cached, no network needed
make dev-test
```

---

## 🔧 Build Caching Internals

<details>
<summary><b>How caching works (click to expand)</b></summary>

### Dockerfile Layer Structure
```dockerfile
# Layer 1: Base (changes rarely) → 95% cache hit
FROM continuumio/miniconda3:latest

# Layer 2: System packages → 90% cache hit
RUN apt-get update && apt-get install ffmpeg vulkan-tools...

# Layer 3: Node.js → 85% cache hit
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y nodejs

# Layer 4: npm globals → 80% cache hit
RUN npm install -g cgpu @google/generative-ai

# Layer 5: Python deps → 70% cache hit
COPY requirements.txt .
RUN pip install -r requirements.txt

# Layer 6: Project metadata → 50% cache hit
COPY pyproject.toml .
RUN pip install -e .

# Layer 7: Source code → 0% cache hit (changes every commit)
COPY src/ /app/src/
```

**Result:** Code changes only rebuild Layer 7 (~2 min vs. 15 min full build)

### Local vs. Registry Cache

| Cache Type | Speed | Shared | Use Case |
|------------|-------|--------|----------|
| **Local Filesystem** | Fast | No | `make dev` |
| **Registry** | Medium | Yes | `make cluster` |
| **BuildKit Internal** | Fastest | No | Automatic |

### Cache Locations
- **Local:** `/tmp/buildx-cache` (cleared on reboot)
- **Registry:** `10.43.17.166:5000/montage-ai:buildcache`
- **BuildKit:** Managed by Docker daemon

</details>

---

## 📚 Additional Resources

Only if you need them:
- [Cluster Details](../deploy/CLUSTER_WORKFLOW.md) - Registry troubleshooting
- [Architecture](architecture.md) - System design
- [Features](features.md) - All capabilities

---

**Last Updated:** January 5, 2026
