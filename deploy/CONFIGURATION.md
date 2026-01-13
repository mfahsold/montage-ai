# Configuration Management - Centralized Values

This document outlines how hardcoded values have been centralized to support multiple deployment environments.

## Overview

All deployment-related hardcoded values have been moved to `deploy/config.env`, which serves as the single source of truth for:
- Registry URLs and credentials
- Kubernetes configuration (namespace, domain, storage)
- Service endpoints and ports
- Resource limits and requests
- Testing URLs and API endpoints

## File Structure

```
deploy/
├── config.env                 # ← Centralized configuration (environment-specific)
├── k3s/
│   ├── deploy.sh             # Uses deploy/config.env
│   ├── build-and-push.sh     # Uses deploy/config.env
│   ├── undeploy.sh           # Uses deploy/config.env
│   └── montage-ai.yaml       # Kubernetes manifests
├── local/                     # (Optional) Local development overlays
└── prod/                      # (Optional) Production overlays
```

## Configuration Variables

### Registry Configuration
```bash
REGISTRY_HOST="YOUR_REGISTRY_IP"              # Registry hostname/IP
REGISTRY_PORT="5000"                       # Registry port
REGISTRY_URL="${REGISTRY_HOST}:${REGISTRY_PORT}"
IMAGE_NAME="montage-ai"                    # Image name
IMAGE_TAG="latest"                         # Image tag
IMAGE_FULL="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
```

### Kubernetes Configuration
```bash
CLUSTER_NAMESPACE="montage-ai"             # K8s namespace
CLUSTER_API_SERVER="https://YOUR_K8S_API:6443"
APP_DOMAIN="montage-ai.fluxibri.local"     # Application domain
TLS_SECRET="montage-ai-tls"                # TLS secret name
```

### Storage Configuration
```bash
STORAGE_CLASS="local-path"                 # StorageClass name
PVC_NAME="montage-data"                    # PersistentVolumeClaim name
PVC_SIZE="50Gi"                            # PVC size
```

### Testing Configuration
```bash
TEST_BASE_URL="http://localhost:5000"      # Backend API for tests
TEST_UPLOAD_URL="http://localhost:5001/api/upload"
```

## Usage

**Hardcoded values policy:** Do not commit new hardcoded values (IPs, paths, registry URLs, resource limits, or credentials) into the codebase. Add new deployment or runtime settings to `deploy/config.env` (for k8s/cluster settings) or the appropriate `config`/`settings` module for runtime defaults. Run `./scripts/check-hardcoded-registries.sh` and the pre-push hook to detect accidental literals.

### 1. Deployment Scripts

All deployment scripts automatically source `deploy/config.env` (preferred) and respect environment overrides. The `Dockerfile` also accepts a build-arg `SERVICE_PORT` and sets `ENV SERVICE_PORT` so the web UI listen port can be configured at build time.

```bash
# deploy.sh
source "${DEPLOY_ROOT}/config.env"
kubectl apply -f montage-ai.yaml

# build-and-push.sh
source "${DEPLOY_ROOT}/config.env"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" --build-arg SERVICE_PORT="${SERVICE_PORT:-5000}"
docker push "${IMAGE_FULL}"

# undeploy.sh
source "${DEPLOY_ROOT}/config.env"
kubectl delete -n ${CLUSTER_NAMESPACE} -f montage-ai.yaml
```

### 2. Environment Overrides

Override default values via environment variables:

```bash
# Override registry
export REGISTRY_HOST="registry.prod.example.com"
export REGISTRY_PORT="443"

# Override Kubernetes settings
export CLUSTER_NAMESPACE="montage-prod"
export APP_DOMAIN="montage.example.com"

# Run deployment
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

```bash
# deploy/config.env (for local)
REGISTRY_HOST="localhost"
REGISTRY_PORT="5000"
CLUSTER_NAMESPACE="montage-dev"
```

### Production

```bash
# deploy/config.env (for production)
REGISTRY_HOST="registry.example.com"
REGISTRY_PORT="443"
CLUSTER_NAMESPACE="montage-prod"
APP_DOMAIN="montage.example.com"
MEMORY_LIMIT="8Gi"
CPU_LIMIT="4000m"
```

### Fluxibri Cluster (example)

If deploying to the Fluxibri cluster, set the canonical registry to the cluster's registry and use in-cluster CI (Tekton) or node import fallbacks instead of GitHub Actions.

```bash
# Example: Fluxibri local registry
export REGISTRY_HOST="192.168.1.16"
export REGISTRY_PORT="30500"
export CLUSTER_NAMESPACE="montage-ai"
export IMAGE_TAG="${IMAGE_TAG:-latest}"
# Full image reference used by scripts and manifests
export IMAGE_FULL="${REGISTRY_HOST}:${REGISTRY_PORT}/montage-ai:${IMAGE_TAG}"
```

# Recommended: Tekton task snippet (push to Fluxibri registry)

Below is a minimal Tekton snippet showing how a Kaniko or Kaniko-compatible task can push the built image to the Fluxibri registry (adapt to your Tekton setup):

```yaml
# tekton snippet (example)
- name: build-and-push
  taskRef:
    name: kaniko-build-cached
  params:
  - name: IMAGE
    value: "${REGISTRY_HOST}:${REGISTRY_PORT}/montage-ai:${IMAGE_TAG}"
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
- We intentionally avoid GitHub Actions for cluster image pushes in this organization; prefer Tekton/cluster-native CI or the node image import pattern as documented in `docs/cluster-deploy/montage-ai.md`.
- The repository includes a Tekton Task that runs the hardcoded-registry scanner: `deploy/k3s/tekton/tasks/run-hardcoded-scan.yaml`. You can add a `Pipeline` or `PipelineRun` that mounts the source workspace and runs this task in your Tekton namespace.
- If you want, I can add a full Tekton `PipelineRun` example as a follow-up commit.


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
# In deploy/config.env
REGISTRY_URL="YOUR_REGISTRY:5000"
IMAGE_FULL="${REGISTRY_URL}/montage-ai:latest"

# In scripts
docker push "${IMAGE_FULL}"

# In tests
export TEST_BASE_URL="${BACKEND_API_URL}"
```

## Adding New Configuration Variables

1. **Add to `deploy/config.env`**:
   ```bash
   NEW_SETTING="${NEW_SETTING:-default_value}"
   ```

2. **Use in scripts**:
   ```bash
   source "${DEPLOY_ROOT}/config.env"
   echo "Using: ${NEW_SETTING}"
   ```

3. **Document** the new variable in this file.

## Pre-Push Hook Integration

The pre-push hook validates that no new hardcoded values are committed:
```bash
# .git/hooks/pre-push checks for:
# - Hardcoded IPs/domains not using env vars
# - Registry URLs not in config.env
# - Namespace names not in config.env
```

## Benefits

✅ **Single Source of Truth**: All deployment config in one place  
✅ **Environment Flexibility**: Easy to support dev/staging/prod  
✅ **Reduced Errors**: Variables shared across scripts eliminate typos  
✅ **Maintainability**: Update once, applies everywhere  
✅ **Security**: Sensitive values can be managed separately  
✅ **Testing**: Tests use consistent URLs across environments  
