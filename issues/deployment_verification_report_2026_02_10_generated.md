# Deployment Verification Report - 2026-02-10

## Summary
Successfully verified installation and deployment on a fresh environment (Simulated Agent Workspace).
- **Environment**: Linux, Docker, K3d (simulated local cluster).
- **Architecture**: ARM64 (detected during deployment).
- **Outcome**: Successful end-to-end montage job execution in cluster after fixes.

## Issues Encountered & Fixes

### 1. Registry Port Defaults (Critical Bug)
**Issue**: Distributed jobs failed with `ImagePullBackOff` attempting to pull from `docker.io:5000/montage-ai:latest`.
**Cause**: `src/montage_ai/config.py` default logic forces `:5000` port if `REGISTRY_HOST` is present but `REGISTRY_PORT` is empty. This breaks standard registries like `docker.io`.
**Fix Applied**: Updated `ClusterConfig.image_full` property in `src/montage_ai/config.py` to respect `IMAGE_FULL` environment variable (Single Source of Truth), preventing incorrect reconstruction of the image URL.

### 2. Architecture Mismatch
**Issue**: `make pre-flight` failed due to default configuration expecting `amd64` while cluster was `arm64`.
**Workaround**: Manually updated `deploy/k3s/config-global.yaml` to set `arch: "arm64"`.
**Recommendation**: Make `config-global.yaml.example` more explicit or auto-detect in `make config`.

### 3. Render Engine Fallback Failure
**Issue**: When the distributed render job failed (due to Issue #1), the system attempted a local fallback but crashed.
**Error**: `AttributeError: 'RenderEngine' object has no attribute 'render'`
**Location**: `src/montage_ai/core/render_engine.py` line 399.
**Recommendation**: Fix the fallback method call (likely renamed/moved).

### 4. Configuration Placeholders
**Issue**: `make config` fails if config file contains `<...>` placeholders.
**Observation**: For local K3d/Kind setups, these placeholders create friction.
**Recommendation**: Provide a `make config-local` target or sensible defaults for localhost.

### 5. LLM/cGPU Unavailable
**Observation**: Tests ran in fallback mode (Template/Heuristic) as expected when tokens/credentials are missing.
**Status**: Verified that the system degrades gracefully without crashing.

## Verification Steps Performed
1.  `scripts/setup.sh` - Successful.
2.  `docker compose build` - Successful.
3.  `docker compose run ... montage-ai.sh run` - Successful (Preview mode).
4.  `make deploy-cluster` (after config fixes) - Successful.
5.  Submitted Job via API - Successful (after code fix).
6.  Downloaded Result - Successful (`downloads/result_20260210_144558.mp4`).
