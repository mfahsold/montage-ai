# LiteLLM Unified Proxy Integration Guide

This guide covers deploying **montage-ai** with the **LiteLLM Unified Proxy** pattern from [fluxibri_core](https://github.com/mfahsold/fluxibri_core).

## Overview

LiteLLM is the **canonical**, recommended backend for in-cluster LLM inference in montage-ai when paired with fluxibri_core's llama-box architecture. It provides:

- **Unified API endpoint** for mixed-tier model inference (Eco, Performance, Distributed)
- **Energy-aware scheduling** and automatic tier selection
- **Multi-model support** with intelligent fallback
- **OpenAI-compatible protocol** native to montage-ai's Creative Director

## Architecture

```
montage-ai Pod
  ↓
creative_director.py (OpenAI client)
  ↓
LiteLLM Unified Proxy (litellm.llama-box-system.svc.cluster.local:4000)
  ↓
[Eco 0.5B | Performance 2B-VL | Distributed 32B] ← Energy-aware routing
```

## Prerequisites

1. **Cluster running fluxibri_core** with llama-box deployment
   - Verify LiteLLM service: `kubectl get svc -n llama-box-system litellm`
   - Verify llama-box tiers: `kubectl get deployment -n llama-box-system`

2. **LITELLM_API_KEY secret** created in montage-ai namespace
   ```bash
   kubectl create secret generic litellm-auth \
     -n montage-ai \
     --from-literal=api-key=$(openssl rand -hex 32)
   ```

## Configuration

### Environment Variables

Set these in `deploy/config.env`:

```bash
# Enable LiteLLM unified proxy (canonical pattern)
OPENAI_API_BASE="http://litellm.llama-box-system.svc.cluster.local:4000"
OPENAI_MODEL="auto"  # LiteLLM will auto-select best available tier
OPENAI_VISION_MODEL="auto"  # Performance tier (2B-VL) when available

# API key (from secret, reference below)
LITELLM_API_KEY="${LITELLM_API_KEY:-}"
```

### Kubernetes Secret

Create the API key secret:

```bash
kubectl create secret generic litellm-auth \
  -n montage-ai \
  --from-literal=api-key=your-secret-key-here
```

Reference in deployment manifests:

```yaml
# deploy/k3s/job.yaml or deployment.yaml
spec:
  containers:
  - name: montage-ai
    env:
    - name: LITELLM_API_KEY
      valueFrom:
        secretKeyRef:
          name: litellm-auth
          key: api-key
    envFrom:
    - configMapRef:
        name: montage-ai-config
```

### Deployment Manifest Example

```yaml
---
apiVersion: batch/v1
kind: Job
metadata:
  name: montage-ai-render
  namespace: montage-ai
spec:
  template:
    spec:
      containers:
      - name: montage-ai
        image: ghcr.io/your-org/montage-ai:latest
        env:
        # LiteLLM configuration
        - name: OPENAI_API_BASE
          value: "http://litellm.llama-box-system.svc.cluster.local:4000"
        - name: OPENAI_MODEL
          value: "auto"  # Auto-select from available tiers
        - name: LITELLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: litellm-auth
              key: api-key
        envFrom:
        - configMapRef:
            name: montage-ai-config
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "24Gi"
            cpu: "6"
      volumes:
      # Mount data volumes as needed
      - name: input
        persistentVolumeClaim:
          claimName: montage-input
```

## Tier Selection Strategy

By setting `OPENAI_MODEL="auto"`, montage-ai delegates tier selection to LiteLLM:

| Task | Recommended | Fallback |
|------|------------|----------|
| Standard editing | Performance (2B-VL) | Eco (0.5B) if Performance unavailable |
| Scene analysis | Eco (0.5B) | Performance if Eco busy |
| Complex reasoning | Distributed (32B) | Performance (2B-VL) |
| Vision tasks | Performance (2B-VL) | Text-only fallback (Eco) |

LiteLLM optimizes based on:
- Model availability
- Queue depth
- Energy budget (if configured)
- Request priority

## Testing

### Local Testing (with port-forward)

```bash
# In one terminal, port-forward to LiteLLM
kubectl port-forward -n llama-box-system svc/litellm 4000:4000

# In another, test montage-ai with local LiteLLM
OPENAI_API_BASE="http://localhost:4000" \
OPENAI_MODEL="auto" \
LITELLM_API_KEY="test-key" \
./montage-ai.sh run dynamic
```

### Cluster Testing

Deploy a test Job:

```bash
# Edit deploy/k3s/test-litelllm.yaml with your config
kubectl apply -f deploy/k3s/test-litellm.yaml
kubectl logs -f job/montage-ai-litellm-test -n montage-ai
```

### Diagnostic Commands

```bash
# Check LiteLLM availability
kubectl exec -n montage-ai <pod> -- \
  curl -s "http://litellm.llama-box-system.svc.cluster.local:4000/v1/models" | jq .

# Check API key is mounted
kubectl exec -n montage-ai <pod> -- env | grep LITELLM

# Monitor API calls (if LiteLLM has logging endpoint)
kubectl logs -f -n llama-box-system svc/litellm
```

## Troubleshooting

### Connection Refused
- **Symptom**: `Connection refused to litellm.llama-box-system`
- **Check**: Verify LiteLLM service exists and is running in llama-box-system namespace
  ```bash
  kubectl get svc litellm -n llama-box-system
  kubectl get pods -n llama-box-system | grep litellm
  ```

### 401 Unauthorized
- **Symptom**: "Invalid API key" error
- **Check**: Verify `LITELLM_API_KEY` is set correctly in secret
  ```bash
  kubectl get secret litellm-auth -n montage-ai -o yaml
  ```

### Auto-model resolution taking too long
- **Symptom**: First query is very slow (model discovery)
- **Fix**: Set `OPENAI_MODEL` explicitly instead of "auto"
  ```bash
  OPENAI_MODEL="2b" OPENAI_VISION_MODEL="2b-vl" ...
  ```

### Vision disabled (text-only fallback)
- **Symptom**: Vision tasks return "not supported"
- **Workaround**: Check Performance tier status in llama-box
  ```bash
  kubectl get deployment llama-box-api-performance -n llama-box-system
  ```
- **Temporary fix**: Use text-only extraction instead (see docs/auto_reframe.md)

## Performance Tuning

### Increase LiteLLM request timeout

By default montage-ai uses 60-second timeout. For complex reasoning tasks, increase:

```bash
LLM_TIMEOUT=180  # 3 minutes max per request
```

### Enable LiteLLM caching (if supported by tier)

Add to LiteLLM configuration:

```yaml
cache:
  type: redis
  ttl: 3600
```

### Set model preferences

Fine-tune tier selection priority:

```bash
# Prefer Performance tier for all text tasks
OPENAI_MODEL="2b"

# Use Distributed for complex breakdown tasks only
COMPLEX_REASONING_MODEL="32b"
```

## Integration with Existing Deployments

### Migrating from cgpu to LiteLLM

1. **Stop cgpu service** (if running locally)
   ```bash
   # Disable cgpu in config.env
   CGPU_ENABLED=false
   ```

2. **Update environment variables**
   ```bash
   # In deploy/config.env
   OPENAI_API_BASE="http://litellm.llama-box-system.svc.cluster.local:4000"
   LITELLM_API_KEY="$(kubectl get secret litellm-auth -n montage-ai -o jsonpath='{.data.api-key}' | base64 -d)"
   ```

3. **Redeploy**
   ```bash
   make -C deploy/k3s config
   make -C deploy/k3s deploy-cluster
   ```

### Hybrid Setup (cgpu + LiteLLM)

Keep cgpu as fallback:

```bash
# Deploy montage-ai with LiteLLM primary
OPENAI_API_BASE="http://litellm.llama-box-system.svc.cluster.local:4000"

# cgpu stays as secondary fallback (still configured in creative_director.py)
CGPU_ENABLED=true
CGPU_HOST="..."
```

## References

- **fluxibri_core LLM Documentation**: https://github.com/mfahsold/fluxibri_core/blob/main/DEVELOPER_QUICKSTART.md
- **LiteLLM Proxy**: https://github.com/BerriAI/litellm
- **llama-box Multi-Tier Inference**: See fluxibri_core deployment docs
- **montage-ai Configuration**: [docs/configuration.md](configuration.md)
- **LLM Model Comparison**: [docs/LLM_MODEL_INTEGRATION.md](LLM_MODEL_INTEGRATION.md)

## Support

For issues or questions:
1. Check [docs/troubleshooting.md](troubleshooting.md)
2. Verify fluxibri_core is running: `kubectl get all -n llama-box-system`
3. Check logs: `kubectl logs -f ` in both `llama-box-system` and `montage-ai` namespaces
4. Open issue with: `kubectl describe job montage-ai-render -n montage-ai` output
