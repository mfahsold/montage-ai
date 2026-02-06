# Configuration Management - Centralized Values

This document outlines how hardcoded values have been centralized to support multiple deployment environments.

## Overview

Deployment-related hardcoded values are centralized in:
- `deploy/k3s/config-global.yaml` (cluster/k8s defaults)
- `deploy/config.env` (runtime/service endpoints and deploy-time defaults)

These two sources are rendered into `deploy/k3s/base/cluster-config.env` for K8s.
- Registry URLs and credentials
- Kubernetes configuration (namespace, domain, storage)
- Service endpoints and ports
- Resource limits and requests
- Testing URLs and API endpoints

## File Structure

```
deploy/
└── config.env             # ← Runtime/deploy defaults (sourced by scripts)
└── k3s/
    ├── config-global.yaml     # ← Canonical config (environment-specific)
    ├── base/cluster-config.env# ← Generated from config-global.yaml
    ├── Makefile               # Cluster operations (deploy, diff, validate)
    ├── deploy.sh              # Uses cluster-config.env
    ├── build-and-push.sh      # Uses config-global.yaml via scripts/common.sh
    └── legacy/manifests/      # Archived legacy manifests (reference only)
```

## Configuration Variables

### Registry Configuration
```bash
REGISTRY_URL="<REGISTRY_URL>"
IMAGE_NAME="montage-ai"                    # Image name (default)
IMAGE_TAG="<IMAGE_TAG>"
IMAGE_FULL="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
```

### Kubernetes Configuration
```bash
CLUSTER_NAMESPACE="<CLUSTER_NAMESPACE>"   # K8s namespace
K3S_CLUSTER_DOMAIN="cluster.local"
MONTAGE_HOSTNAME="<MONTAGE_HOSTNAME>"
```

### Cluster Scheduling (Architecture & Node Selection)
```bash
# Allow scheduling across mixed architectures (requires multi-arch image)
CLUSTER_ALLOW_MIXED_ARCH="true"

# Label key used when pinning to the current architecture
CLUSTER_ARCH_SELECTOR_KEY="kubernetes.io/arch"

# Optional selector for distributed jobs (comma-separated key=value)
CLUSTER_NODE_SELECTOR=""
```

### Runtime Services (Queues & LLM)
```bash
REDIS_HOST="<REDIS_HOST>"
REDIS_PORT="6379"
OPENAI_API_BASE="<OPENAI_API_BASE>"
OPENAI_API_KEY="<OPENAI_API_KEY>"
OPENAI_MODEL="auto"
OPENAI_VISION_MODEL=""
CGPU_ENABLED="false"
CGPU_GPU_ENABLED="false"
CGPU_HOST="<CGPU_HOST>"
CGPU_PORT="8080"
CGPU_MODEL="gemini-2.0-flash"
CGPU_TIMEOUT="1200"
CGPU_MAX_CONCURRENCY="1"
CGPU_STATUS_TIMEOUT="30"
CGPU_GPU_CHECK_TIMEOUT="120"
FINAL_ENCODE_BACKEND="ffmpeg"  # ffmpeg|router
FORCE_CGPU_ENCODING="false"
```

### Storage Configuration
```bash
STORAGE_CLASS_DEFAULT="<STORAGE_CLASS>"    # StorageClass name
STORAGE_CLASS_NFS="nfs-client"
NFS_SERVER="<NFS_SERVER>"                  # Optional NFS server IP
NFS_PATH="<NFS_PATH>"
```

### Testing Configuration
```bash
TEST_BASE_URL="<MONTAGE_API_BASE>"
TEST_UPLOAD_URL="<MONTAGE_API_BASE>/api/upload"
```

## Usage

**Hardcoded values policy:** Do not commit new hardcoded values (IPs, paths, registry URLs, resource limits, or credentials) into the codebase. Add new deployment or runtime settings to `deploy/k3s/config-global.yaml` (for k8s/cluster settings) or the appropriate `config`/`settings` module for runtime defaults. Run `./scripts/check-hardcoded-registries.sh` and the pre-push hook to detect accidental literals.

### 1. Deployment Scripts

All deployment scripts read `deploy/k3s/config-global.yaml` (preferred) and/or `deploy/k3s/base/cluster-config.env` (generated). The `Dockerfile` also accepts a build-arg `SERVICE_PORT` and sets `ENV SERVICE_PORT` so the web UI listen port can be configured at build time.

```bash
# deploy.sh
source "${DEPLOY_ROOT}/k3s/base/cluster-config.env"
kubectl apply -k deploy/k3s/overlays/cluster

# build-and-push.sh
source "${DEPLOY_ROOT}/k3s/base/cluster-config.env"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" --build-arg SERVICE_PORT="${SERVICE_PORT:-5000}"
docker push "${IMAGE_FULL}"
```

### 2. Environment Overrides

Override default values via environment variables:

```bash
# Point to an alternate config-global.yaml
export CONFIG_GLOBAL="/path/to/config-global.prod.yaml"

# Render env + run deployment
make -C deploy/k3s config
./deploy/k3s/deploy.sh
```

### 3. Testing URLs

Tests automatically use environment variables:

```bash
# Local testing (default)
pytest tests/ -q

# Cluster testing
export TEST_BASE_URL="http://<MONTAGE_SERVICE_HOST>:<PORT>"
pytest tests/ -q

# Production testing
export TEST_BASE_URL="https://<MONTAGE_HOSTNAME>"
pytest tests/ -q
```

## Environment-Specific Configurations

### Local Development

```yaml
# deploy/k3s/config-global.yaml (local)
registry:
  url: "<REGISTRY_URL>"
cluster:
  namespace: "<CLUSTER_NAMESPACE>"
  hostnames:
    montage: "<MONTAGE_HOSTNAME>"
images:
  montage_ai:
    tag: "<IMAGE_TAG>"
```

### Production

```yaml
# deploy/k3s/config-global.yaml (production)
registry:
  url: "<REGISTRY_URL>"
cluster:
  namespace: "<CLUSTER_NAMESPACE>"
  hostnames:
    montage: "<MONTAGE_HOSTNAME>"
images:
  montage_ai:
    tag: "main-<sha>"
```

### Private/Enterprise Cluster (example)

If your organization provides an internal registry or cluster CI (Tekton, Jenkins, etc.), use those endpoints instead of public registries.

```yaml
registry:
  url: "<REGISTRY_URL>"
cluster:
  namespace: "<CLUSTER_NAMESPACE>"
  hostnames:
    montage: "<MONTAGE_HOSTNAME>"
images:
  montage_ai:
    tag: "latest"
```

### Recommended: Tekton task snippet (push to your registry)

Below is a minimal Tekton snippet showing how a Kaniko or Kaniko-compatible task can push the built image to your registry (adapt to your Tekton setup):

```yaml
# tekton snippet (example)
- name: build-and-push
  taskRef:
    name: kaniko-build-cached
  params:
  - name: IMAGE
    value: "${REGISTRY_URL}/montage-ai:${IMAGE_TAG}"
  - name: CONTEXT
    value: "$(resources.inputs.workspace.path)"
  - name: DOCKERFILE
    value: "./Dockerfile"
  workspaces:
  - name: source
    workspace: shared-workspace
```

**Notes:**
- Use `kubectl create secret docker-registry` to create the registry credential in the target namespace and reference it in your Tekton task or pipeline via `imagePullSecrets`/`taskrun` secret refs.
- Prefer cluster-native CI (Tekton/Jenkins) or a node image import pattern if your environment requires it.
- The repository includes a Tekton Task that runs the hardcoded-registry scanner: `deploy/k3s/tekton/tasks/run-hardcoded-scan.yaml`. You can add a `Pipeline` or `PipelineRun` that mounts the source workspace and runs this task in your Tekton namespace.


## Migration from Hardcoded Values

### Before (Hardcoded)
```yaml
image: <HARDCODED_REGISTRY_URL>/montage-ai:latest
namespace: montage-ai
```

```bash
docker push "<HARDCODED_REGISTRY_URL>/montage-ai:latest"
```

### After (Centralized)
```bash
# In deploy/k3s/config-global.yaml (rendered to cluster-config.env)
REGISTRY_URL="<REGISTRY_URL>"
IMAGE_FULL="${REGISTRY_URL}/montage-ai:latest"

# In scripts
docker push "${IMAGE_FULL}"

# In tests
export TEST_BASE_URL="${BACKEND_API_URL}"
```

## Adding New Configuration Variables

1. **Add to `deploy/k3s/config-global.yaml`** (and update the `.example` file).

2. **Export via `scripts/ops/render_cluster_config_env.sh`** (and optionally `scripts/ops/lib/config_global.sh`).

3. **Use in scripts/manifests**:
   ```bash
   source "${DEPLOY_ROOT}/k3s/base/cluster-config.env"
   echo "Using: ${NEW_SETTING}"
   ```

## Pre-Push Hook Integration

The pre-push hook validates that no new hardcoded values are committed:
```bash
# .git/hooks/pre-push checks for:
# - Hardcoded IPs/domains not using env vars
# - Registry URLs not in config-global.yaml
# - Namespace names not in config-global.yaml
# - Accidental additions to .github/workflows/ (this repo enforces a no-Actions policy)
```

## KEDA Thresholds & Autoscaling Configuration

KEDA list-length thresholds for queue-driven autoscaling are configurable from `deploy/config.env`.
Set the following values and re-run `make config` before rendering or deploying:

- `WORKER_QUEUE_SCALE_THRESHOLD` (default: 10)

These values are rendered into `deploy/k3s/base/cluster-config.env` and used to generate `deploy/k3s/overlays/cluster/keda-scaledobjects.yaml` via `deploy/k3s/prepare-scaledobjects.sh`.

Run:

```bash
make -C deploy/k3s config
make -C deploy/k3s validate
```

to ensure the generated manifests reflect your configured thresholds.

## Benefits

✅ **Single Source of Truth**: All deployment config in one place  
✅ **Environment Flexibility**: Easy to support dev/staging/prod  
✅ **Reduced Errors**: Variables shared across scripts eliminate typos  
✅ **Maintainability**: Update once, applies everywhere  
✅ **Security**: Sensitive values can be managed separately  
✅ **Testing**: Tests use consistent URLs across environments  
