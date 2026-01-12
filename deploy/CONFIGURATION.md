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
