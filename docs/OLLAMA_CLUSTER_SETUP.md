# Ollama LLM Integration - Canonical Cluster Approach

## Current State

✅ **Ollama Deployment**: Already running in cluster
- Pod: `ollama-d7d5bc96c-sb5ll`
- Service: `ollama.montage-ai.svc.cluster.local:11434`
- Storage: Persistent volume claim configured

❌ **Configuration Issues**:
- `OLLAMA_HOST` environment variable is **empty** in cluster-config.env
- `DIRECTOR_MODEL` defaults to `llama3.1:70b` (large, needs resources)
- Models not pre-pulled in Ollama

---

## Canonical Approach (From Fluxibri-Core)

Based on the codebase analysis:

### 1. **Priority Chain** (creative_director.py)
```
1. OPENAI_API_BASE (LiteLLM/vLLM or OpenAI-compatible)
2. GOOGLE_API_KEY (direct Gemini API)
3. CGPU_ENABLED (cgpu-server proxy - currently broken, no credentials)
4. OLLAMA_HOST (fallback - working, just needs configuration)
```

### 2. **Environment Variables** (deploy/k3s/base/cluster-config.env)
```bash
# Primary: Disable CGPU (no credentials)
CGPU_ENABLED=false

# Secondary: Enable Ollama (canonical approach for cluster)
OLLAMA_HOST=http://ollama.montage-ai.svc.cluster.local:11434
OLLAMA_MODEL=llava              # For vision/image tasks
DIRECTOR_MODEL=mistral:7b       # Lightweight, fast creative direction
```

### 3. **Model Selection for Montage AI**

| Model | Use Case | Size | Speed | Quality | Status |
|-------|----------|------|-------|---------|--------|
| **mistral:7b** | Creative director | 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **Recommended** |
| **llama2:7b** | Creative direction | 7B | ⭐⭐⭐⭐ | ⭐⭐⭐ | Alternative |
| **llava:7b** | Image/scene analysis | 7B | ⭐⭐⭐ | ⭐⭐⭐ | For vision tasks |
| **llama3.1:70b** | High quality | 70B | ⭐⭐ | ⭐⭐⭐⭐⭐ | Too slow (needs GPU) |

---

## Implementation Plan

### Step 1: Update Environment Variables

```bash
# File: deploy/k3s/base/cluster-config.env
# Change from:
CGPU_ENABLED=true
CGPU_GPU_ENABLED=false
OLLAMA_HOST=

# Change to:
CGPU_ENABLED=false              # Disable cgpu-server (no credentials)
CGPU_GPU_ENABLED=false
OLLAMA_HOST=http://ollama.montage-ai.svc.cluster.local:11434
OLLAMA_MODEL=llava              # For image/scene analysis
DIRECTOR_MODEL=mistral:7b       # For creative direction prompts
```

### Step 2: Pre-pull Models into Ollama

```bash
# Pull models into the running Ollama pod
kubectl -n montage-ai exec ollama-d7d5bc96c-sb5ll -- ollama pull mistral:7b
kubectl -n montage-ai exec ollama-d7d5bc96c-sb5ll -- ollama pull llava:7b

# List available models
kubectl -n montage-ai exec ollama-d7d5bc96c-sb5ll -- ollama list
```

### Step 3: Verify Configuration

```bash
# Test Ollama connectivity from a pod
kubectl -n montage-ai run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://ollama.montage-ai.svc.cluster.local:11434/api/tags

# Expected output: list of available models in JSON
```

### Step 4: Redeploy Jobs with New Config

```bash
# Apply updated config
kubectl -n montage-ai apply -f deploy/k3s/base/cluster-rbac.yaml
set -a && source deploy/k3s/base/cluster-config.env && set +a

# Delete old jobs
kubectl -n montage-ai delete job montage-ai-creative-master --ignore-not-found

# Redeploy
envsubst < /tmp/montage-ai-creative-master.yaml | kubectl apply -f -
```

### Step 5: Test Creative Director with Ollama

```bash
# Run a test
./montage-ai.sh run dynamic

# Monitor logs for Ollama usage:
# Should see: "Creative Director primary backend: ollama at http://ollama.montage-ai.svc.cluster.local:11434"
```

---

## Configuration Hierarchy (How It Works)

When creative_director.py starts:

```
1. Check OPENAI_API_BASE → not set, skip
2. Check GOOGLE_API_KEY → could use, but not set in env
3. Check CGPU_ENABLED → set to false, skip
4. Use OLLAMA_HOST → ✅ http://ollama.montage-ai.svc.cluster.local:11434
   └─ Use DIRECTOR_MODEL=mistral:7b for creative direction
   └─ Use OLLAMA_MODEL=llava for image/scene analysis
```

---

## Expected Performance

| Task | Latency | Quality | Notes |
|------|---------|---------|-------|
| Creative direction (Mistral 7B) | 1-3 sec | Good (7/10) | Fast, acceptable for iteration |
| Scene analysis (LLaVA 7B) | 2-5 sec | Fair (6/10) | Works on images, slower |
| Batch processing | Unlimited | Consistent | No API quotas, runs locally |

---

## Cost Comparison

| Backend | Setup | Cost | Quotas |
|---------|-------|------|--------|
| **Ollama** | Already running | $0 | Unlimited |
| **cgpu (Gemini)** | Needs OAuth | $0.075/$0.30/M tokens | 20 req/day free |
| **Direct Gemini API** | API key only | $0.075/$0.30/M tokens | 20 req/day free |

---

## Files to Modify

1. **deploy/k3s/base/cluster-config.env**
   - Set `CGPU_ENABLED=false`
   - Set `OLLAMA_HOST=http://ollama.montage-ai.svc.cluster.local:11434`
   - Set `DIRECTOR_MODEL=mistral:7b`
   - Set `OLLAMA_MODEL=llava`

2. **No code changes needed**
   - creative_director.py already supports this configuration
   - Config loading via `config.py` handles everything

---

## Verification Checklist

- [ ] Ollama pod is running: `kubectl get pods -l app.kubernetes.io/name=ollama`
- [ ] Models pre-pulled: `kubectl exec ollama-* -- ollama list`
- [ ] Config updated: `cat deploy/k3s/base/cluster-config.env | grep OLLAMA`
- [ ] Jobs deleted: `kubectl delete job montage-ai-creative-master --ignore-not-found`
- [ ] New job created: `envsubst < /tmp/montage-ai-creative-master.yaml | kubectl apply -f -`
- [ ] Job running: `kubectl get pods -l job-name=montage-ai-creative-master`
- [ ] Logs show Ollama: `kubectl logs <pod> | grep -i ollama`

---

## Rollback Plan

If something goes wrong:

```bash
# Revert to cgpu-server (restore credentials first)
kubectl -n montage-ai edit configmap montage-ai-config
# Set CGPU_ENABLED=true and OLLAMA_HOST=''

# Or use previous config
git checkout deploy/k3s/base/cluster-config.env

# Restart jobs
kubectl -n montage-ai delete job montage-ai-creative-master
envsubst < /tmp/montage-ai-creative-master.yaml | kubectl apply -f -
```

---

## Next Steps

1. Update **cluster-config.env** with Ollama settings
2. Pre-pull models into Ollama
3. Verify connectivity
4. Redeploy jobs with new config
5. Test creative director with Ollama backend
6. Monitor performance and adjust model selection if needed
