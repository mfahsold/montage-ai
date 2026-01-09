# Quick Deployment Reference

## Quick Start

### 1. Configure Registry
Edit `deploy/config.env` and set your registry:
```bash
REGISTRY_HOST="YOUR_REGISTRY_IP"     # Your registry IP
REGISTRY_PORT="5000"              # Registry port
```

### 2. Build & Push Image
```bash
cd deploy/k3s
chmod +x build-and-push.sh deploy.sh undeploy.sh
./build-and-push.sh
```

### 3. Deploy to Environment

#### Development (Fast feedback, local registry)
```bash
./deploy.sh dev
```
- Uses local registry: `YOUR_REGISTRY:5000`
- Fast FFmpeg preset
- Minimal quality
- Good for testing changes

#### Staging (Balanced quality)
```bash
./deploy.sh staging
```
- Moderate quality settings
- Medium resources
- Good for QA testing

#### Production (High quality)
```bash
./deploy.sh production
```
- Highest quality output
- Full upscaling & enhancement
- Premium resources
- Slower rendering

## Configuration

### What Gets Deployed
```
deploy/k3s/
  ├── base/              ← Base resources (all overlays use)
  │   ├── deployment.yaml
  │   ├── namespace.yaml
  │   ├── service-ingress.yaml
  │   └── network-policy.yaml
  │
  └── overlays/          ← Environment-specific configs
      ├── dev/           ← Fast, local registry
      ├── staging/       ← Balanced quality
      └── production/    ← High quality
```

### Registry Configuration

The `build-and-push.sh` script uses `deploy/config.env`:

```bash
# From config.env
REGISTRY_HOST="YOUR_REGISTRY_IP"
REGISTRY_PORT="5000"
IMAGE_FULL="${REGISTRY_HOST}:${REGISTRY_PORT}/montage-ai:latest"
```

**Make sure registry is running:**
```bash
# Check if accessible
curl http://YOUR_REGISTRY:5000/v2/

# Start local registry (if needed)
docker run -d -p 5000:5000 --name registry registry:2
```

## Troubleshooting

### Build Fails
```bash
# 1. Check Docker is running
docker info

# 2. Check Dockerfile exists
ls -la Dockerfile

# 3. View build output (no cleanup on error)
./build-and-push.sh 2>&1 | tail -50
```

### Push Fails
```bash
# 1. Check registry is accessible
curl http://YOUR_REGISTRY:5000/v2/

# 2. Check image was built
docker images | grep montage-ai

# 3. Check Docker can reach registry
docker tag test/app:latest YOUR_REGISTRY:5000/test:latest
docker push YOUR_REGISTRY:5000/test:latest
```

### Deployment Fails
```bash
# 1. Check kubectl is configured
kubectl cluster-info

# 2. Check namespace exists
kubectl get namespace montage-ai

# 3. Check pods
kubectl get pods -n montage-ai

# 4. Check pod logs
kubectl logs -n montage-ai -l app=montage-ai -f

# 5. Describe pod for errors
kubectl describe pod <pod-name> -n montage-ai
```

### Registry Connectivity Issues
```bash
# Check if registry is running
docker ps | grep registry

# Check registry is responding
curl -v http://YOUR_REGISTRY:5000/v2/

# If not accessible, check network
ping YOUR_REGISTRY_IP
netstat -tupln | grep 5000

# Start fresh registry
docker stop registry || true
docker rm registry || true
docker run -d -p 5000:5000 --name registry registry:2
```

## Advanced Deployment

### Deploy Specific Overlay
```bash
# Deploy to specific environment
./deploy.sh dev          # Development
./deploy.sh staging      # Staging
./deploy.sh production   # Production

# The script accepts overlay name as argument
```

### Using Kustomize Directly
```bash
# View what would be deployed
kustomize build overlays/dev

# Build and apply manually
kustomize build overlays/dev | kubectl apply -f -

# Dry-run
kustomize build overlays/staging | kubectl apply -f - --dry-run=client
```

### Monitor Deployment
```bash
# Watch pods starting
kubectl get pods -n montage-ai -w

# Follow logs
kubectl logs -n montage-ai -l app=montage-ai -f

# Port forward to test
kubectl port-forward -n montage-ai svc/montage-ai 8080:80
curl http://localhost:8080
```

### Cleanup
```bash
# Remove deployment
./undeploy.sh

# Verify removal
kubectl get pods -n montage-ai

# Delete namespace completely
kubectl delete namespace montage-ai

# Delete local images (if needed)
docker rmi montage-ai:latest
docker rmi YOUR_REGISTRY:5000/montage-ai:latest
```

## Environment Variables

Override in `deploy/config.env`:

```bash
# Registry
REGISTRY_HOST="YOUR_REGISTRY_IP"
REGISTRY_PORT="5000"

# Kubernetes
CLUSTER_NAMESPACE="montage-ai"
CLUSTER_API_SERVER="https://YOUR_K8S_API:6443"

# Application
APP_NAME="montage-ai"
APP_DOMAIN="montage-ai.fluxibri.local"

# Image
IMAGE_TAG="latest"  # Change to deploy specific version
```

## Helpful Commands

```bash
# Get all resources in namespace
kubectl get all -n montage-ai

# Get detailed pod info
kubectl describe pod -n montage-ai -l app=montage-ai

# Stream logs from all pods
kubectl logs -n montage-ai -l app=montage-ai -f --all-containers=true

# Execute command in running pod
kubectl exec -it <pod-name> -n montage-ai -- /bin/bash

# Port forward to access service
kubectl port-forward -n montage-ai svc/montage-ai 8080:80

# Delete stuck pod
kubectl delete pod <pod-name> -n montage-ai --grace-period=0 --force

# Restart deployment
kubectl rollout restart deployment/montage-ai -n montage-ai
```
