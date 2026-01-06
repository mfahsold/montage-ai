#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🚀 Submitting Trailer Job (Documentary 30s)...${NC}"

# Clean up previous job if it exists
kubectl delete job montage-ai-documentary-trailer -n montage-ai --ignore-not-found

# Apply the job manifest
kubectl apply -f deploy/k3s/job-documentary-trailer.yaml

echo -e "${YELLOW}⏳ Waiting for job to complete (timeout: 10m)...${NC}"
# Wait for the job to complete
if kubectl wait --for=condition=complete job/montage-ai-documentary-trailer -n montage-ai --timeout=600s; then
    echo -e "${GREEN}✅ Job completed successfully!${NC}"
    
    echo -e "${YELLOW}📥 Retrieving results to local laptop...${NC}"
    # Run the python retrieval script
    python3 scripts/retrieve_results.py
    
    echo -e "${GREEN}✅ Done! Check data/output/ for your files.${NC}"
else
    echo -e "${RED}❌ Job failed or timed out.${NC}"
    kubectl logs -n montage-ai job/montage-ai-documentary-trailer --tail=50
    exit 1
fi
