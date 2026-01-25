# Configuration Management - Centralized Values

This document outlines how hardcoded values have been centralized to support multiple deployment environments.

## Overview

All deployment-related hardcoded values have been moved to `deploy/k3s/config-global.yaml`, which serves as the single source of truth for:
- Registry URLs and credentials
- Kubernetes configuration (namespace, domain, storage)
- Service endpoints and ports
- Resource limits and requests
- Testing URLs and API endpoints

## File Structure

```
deploy/
├── k3s/
│   ├── config-global.yaml     # ← Canonical config (environment-specific)
│   ├── base/cluster-config.env# ← Generated from config-global.yaml
│   ├── Makefile               # Cluster operations (deploy, diff, validate)
│   ├── deploy.sh              # Uses cluster-config.env
│   ├── build-and-push.sh      # Uses config-global.yaml via scripts/common.sh
│   └── montage-ai.yaml        # Kubernetes manifests (legacy)
├── local/                     # (Optional) Local development overlays
└── prod/                      # (Optional) Production overlays
```

## Configuration Variables

### Registry Configuration
```bash
REGISTRY_URL="registry.registry.svc.cluster.local:5000"
IMAGE_NAME="montage-ai"                    # Image name
IMAGE_TAG="latest-amd64"                   # Image tag
IMAGE_FULL="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
```

### Kubernetes Configuration
```bash
CLUSTER_NAMESPACE="montage-ai"             # K8s namespace
K3S_CLUSTER_DOMAIN="cluster.local"
```

### Storage Configuration
```bash
STORAGE_CLASS_DEFAULT="local-path"         # StorageClass name
STORAGE_CLASS_NFS="nfs-client"
NFS_SERVER="10.0.0.10"                     # Optional NFS server IP
NFS_PATH="/mnt/nfs-montage"
```

### Testing Configuration
```bash
TEST_BASE_URL="http://localhost:5000"      # Backend API for tests
TEST_UPLOAD_URL="http://localhost:5001/api/upload"
```

## Usage

**Hardcoded values policy:** Do not commit new hardcoded values (IPs, paths, registry URLs, resource limits, or credentials) into the codebase. Add new deployment or runtime settings to `deploy/k3s/config-global.yaml` (for k8s/cluster settings) or the appropriate `config`/`settings` module for runtime defaults. Run `./scripts/check-hardcoded-registries.sh` and the pre-push hook to detect accidental literals.

### 1. Deployment Scripts

All deployment scripts read `deploy/k3s/config-global.yaml` (preferred) and/or `deploy/k3s/base/cluster-config.env` (generated). The `Dockerfile` also accepts a build-arg `SERVICE_PORT` and sets `ENV SERVICE_PORT` so the web UI listen port can be configured at build time.

```bash
# deploy.sh
source "${DEPLOY_ROOT}/k3s/base/cluster-config.env"
kubectl apply -k deploy/k3s/overlays/production

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
make test

# Cluster testing
export TEST_BASE_URL="http://montage-ai.montage-ai.svc.cluster.local:5000"
make test

# Production testing
export TEST_BASE_URL="https://montage.example.com"
make test
```

## Environment-Specific Configurations

### Local Development

```yaml
# deploy/k3s/config-global.yaml (local)
registry:
  url: "registry.registry.svc.cluster.local:5000"
cluster:
  namespace: "montage-dev"
images:
  montage_ai:
    tag: "latest-amd64"
```

### Production

```yaml
# deploy/k3s/config-global.yaml (production)
registry:
  url: "registry.example.com:443"
cluster:
  namespace: "montage-prod"
images:
  montage_ai:
    tag: "main-<sha>"
```

### Private/Enterprise Cluster (example)

If your organization provides an internal registry or cluster CI (Tekton, Jenkins, etc.), use those endpoints instead of public registries.

```yaml
registry:
  url: "registry.internal.example:5000"
cluster:
  namespace: "montage-ai"
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
image: YOUR_REGISTRY:30500/montage-ai:latest
namespace: montage-ai
```

```bash
docker push "YOUR_REGISTRY:5000/montage-ai:latest"
```

### After (Centralized)
```bash
# In deploy/k3s/config-global.yaml (rendered to cluster-config.env)
REGISTRY_URL="YOUR_REGISTRY:5000"
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
```

## Benefits

✅ **Single Source of Truth**: All deployment config in one place  
✅ **Environment Flexibility**: Easy to support dev/staging/prod  
✅ **Reduced Errors**: Variables shared across scripts eliminate typos  
✅ **Maintainability**: Update once, applies everywhere  
✅ **Security**: Sensitive values can be managed separately  
✅ **Testing**: Tests use consistent URLs across environments  
