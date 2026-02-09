# Installation Test & Verification Guide

This document describes how to verify a clean Montage AI installation on a new system (local or remote).

**Last Tested:** 2026-02-09  
**System:** Linux 29GB RAM, 12 CPUs, Docker 28.2.2, Kubernetes available

---

## Quick Verification Checklist

### System Requirements

```bash
# ✅ Check Docker
docker --version        # Should be >= 20.10
docker compose version  # Should be >= 2.0

# ✅ Check Resources
free -h                 # >= 16 GB RAM recommended
nproc                   # >= 4 cores recommended
df -h /                 # >= 10 GB disk free
```

### Repository Setup

```bash
# ✅ Clone and initialize
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
./scripts/setup.sh      # Should complete without errors
```

### Docker Build & Test

```bash
# ✅ Build Docker image
docker compose build

# ✅ Test Python import
docker compose run --rm montage-ai python -c "import montage_ai; print('OK')"

# ✅ Test CLI help (IMPORTANT: Use /app/montage-ai.sh, not ./montage-ai.sh)
docker compose run --rm montage-ai /app/montage-ai.sh --help
```

### Kubernetes Deployment (Optional)

```bash
# ✅ If K8s cluster available
kubectl cluster-info

# ✅ Configure and validate
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
$EDITOR deploy/k3s/config-global.yaml  # Replace all <...> placeholders

make -C deploy/k3s config              # Generate cluster-config.env
make -C deploy/k3s validate            # Validate kustomize manifests
```

---

## Common Issues & Fixes

### Issue 1: scripts/setup.sh fails with "free: Ungültige Option"

**Cause:** Non-portable GNU free extension
**Solution:** Already fixed in version >= 19da914
**Test:** `./scripts/setup.sh` should complete without errors

### Issue 2: Docker CLI fails - "stat ./montage-ai.sh: no such file or directory"

**Cause:** Incorrect path in docker compose run
**Wrong:** `docker compose run --rm montage-ai ./montage-ai.sh run`
**Correct:** `docker compose run --rm montage-ai /app/montage-ai.sh run`
**Test:** Run with correct path

### Issue 3: Kubernetes deployment fails with "unresolved placeholder"

**Cause:** config-global.yaml contains `<CLUSTER_NAMESPACE>` etc.
**Fix:** Replace all angle-bracket placeholders in `deploy/k3s/config-global.yaml`
**Test:** `make -C deploy/k3s config` should generate cluster-config.env

---

## Full Installation Test Workflow

Run this complete test on a new system:

```bash
#!/bin/bash
set -e

REPO_URL="${1:-https://github.com/mfahsold/montage-ai.git}"
TEST_DIR="/tmp/montage-ai-test-$$"

echo "🚀 Full Montage AI Installation Test"
echo "📁 Test directory: $TEST_DIR"
echo ""

# 1. Clone
echo "1️⃣  Cloning repository..."
git clone "$REPO_URL" "$TEST_DIR"
cd "$TEST_DIR"

# 2. Setup
echo "2️⃣  Running setup script..."
./scripts/setup.sh

# 3. Docker build
echo "3️⃣  Building Docker image..."
docker compose build

# 4. Python test
echo "4️⃣  Testing Python import..."
docker compose run --rm montage-ai python -c "import montage_ai; print('✅ OK')"

# 5. CLI test
echo "5️⃣  Testing CLI..."
docker compose run --rm montage-ai /app/montage-ai.sh --version | head -5

# 6. Kubernetes (if available)
if kubectl cluster-info &>/dev/null; then
    echo "6️⃣  Testing Kubernetes config generation..."
    cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
    # Apply minimal valid config
    sed -i 's/<CLUSTER_NAMESPACE>/montage-ai-test/g' deploy/k3s/config-global.yaml
    sed -i 's/<CLUSTER_DOMAIN>/cluster.local/g' deploy/k3s/config-global.yaml
    sed -i 's/<CONTROL_PLANE_IP>/127.0.0.1/g' deploy/k3s/config-global.yaml
    sed -i 's/<NFS_SERVER_IP>/localhost/g' deploy/k3s/config-global.yaml
    
    make -C deploy/k3s config
    make -C deploy/k3s validate
    echo "✅ Kubernetes validation passed"
else
    echo "6️⃣  Kubernetes not available (skipping)"
fi

echo ""
echo "✅ Installation test complete!"
echo "📂 Test files in: $TEST_DIR"
```

---

## Deployment Scenarios

### Scenario 1: Local Docker Development

```bash
# Build
docker compose build

# Test
docker compose run --rm montage-ai /app/montage-ai.sh --help

# Run with preview quality
QUALITY_PROFILE=preview docker compose run --rm montage-ai /app/montage-ai.sh run

# Web UI
docker compose up
# Open http://localhost:8080
```

### Scenario 2: Kubernetes Single Node

```bash
# Setup config
cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml
# Edit with your registry, namespace, storage details

# Generate manifests
make -C deploy/k3s config

# Validate
make -C deploy/k3s validate

# Deploy
./deploy/k3s/deploy.sh cluster

# Check status
kubectl get pods -n montage-ai
```

### Scenario 3: Kubernetes Multi-Node Cluster

Same as Scenario 2, but ensure:

- All nodes have K8s labels/taints configured
- Storage class supports RWX (ReadWriteMany)
- Registry is accessible from all nodes
- NFS mount is shared across nodes

---

## Verification Commands

```bash
# System check
./scripts/setup.sh

# Docker check
docker compose config | head -30

# Image availability
docker images | grep montage-ai

# Container test
docker compose run --rm montage-ai env | grep MONTAGE

# Kubernetes resources
kubectl get all -n montage-ai
kubectl logs -n montage-ai -l app=montage-ai-web
```

---

## Reporting Issues

If you encounter problems during installation:

1. Run the full test workflow
2. Capture the output (especially the error message)
3. Include system info:

```bash
uname -a
docker --version
docker compose version
free -h
nproc
```

4. Create an issue at: <https://github.com/mfahsold/montage-ai/issues>

---

## Related Documentation

- [README.md](../README.md) — Quick start
- [deploy/README.md](../deploy/README.md) — Deployment overview
- [deploy/k3s/README.md](../deploy/k3s/README.md) — Kubernetes details
- [docs/configuration.md](configuration.md) — Configuration reference
- [docs/troubleshooting.md](troubleshooting.md) — Common problems & solutions
