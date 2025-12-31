#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

NAMESPACE="montage-ai"

echo -e "${GREEN}Starting idempotent cluster setup for Montage AI...${NC}"

# 1. Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed.${NC}"
    exit 1
fi

# 2. Create Namespace
echo -e "${YELLOW}Checking namespace '${NAMESPACE}'...${NC}"
if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
    echo "Creating namespace '${NAMESPACE}'..."
    kubectl create namespace "${NAMESPACE}"
else
    echo "Namespace '${NAMESPACE}' already exists."
fi

# 3. Image Pull Secret (regcred)
# This is often needed for private registries or GHCR if not public
echo -e "${YELLOW}Checking image pull secret 'regcred'...${NC}"
if ! kubectl get secret regcred -n "${NAMESPACE}" &> /dev/null; then
    echo -e "${YELLOW}Secret 'regcred' not found.${NC}"
    echo "Attempting to create from local Docker config..."
    
    if [ -f ~/.docker/config.json ]; then
        kubectl create secret generic regcred \
            --from-file=.dockerconfigjson=$HOME/.docker/config.json \
            --type=kubernetes.io/dockerconfigjson \
            -n "${NAMESPACE}" || echo -e "${RED}Failed to create regcred. You may need to create it manually.${NC}"
    else
        echo -e "${RED}No ~/.docker/config.json found. Skipping regcred creation.${NC}"
    fi
else
    echo "Secret 'regcred' exists."
fi

# 4. CGPU Secrets
echo -e "${YELLOW}Checking CGPU secrets...${NC}"
if ! kubectl get secret cgpu-credentials -n "${NAMESPACE}" &> /dev/null; then
    echo -e "${YELLOW}Secret 'cgpu-credentials' not found.${NC}"
    # Check if we have local config to create it from
    if [ -f ~/.config/cgpu/config.json ]; then
        echo "Creating 'cgpu-credentials' from ~/.config/cgpu/config.json..."
        kubectl create secret generic cgpu-credentials \
            --from-file=config.json=$HOME/.config/cgpu/config.json \
            -n "${NAMESPACE}"
    else
        echo -e "${YELLOW}Warning: ~/.config/cgpu/config.json not found. CGPU features might fail.${NC}"
        echo "You can create the secret manually later: kubectl create secret generic cgpu-credentials --from-file=config.json=... -n ${NAMESPACE}"
    fi
else
    echo "Secret 'cgpu-credentials' exists."
fi

# 5. Apply Base Configuration
echo -e "${GREEN}Applying Kubernetes configurations...${NC}"
kubectl apply -k deploy/k3s/base/

# 6. Apply Distributed Overlay (if desired, or make it an argument)
# For now, we'll stick to base + specific jobs as needed, but let's ensure the PVCs are bound.

echo -e "${GREEN}Waiting for PVCs to be bound...${NC}"
# This is a simple check, might not wait indefinitely
kubectl get pvc -n "${NAMESPACE}"

echo -e "${GREEN}Cluster setup complete!${NC}"
echo "You can now run jobs using 'make job' or 'kubectl apply -f deploy/k3s/job-distributed-trailer.yaml'"
