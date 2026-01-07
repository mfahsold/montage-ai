---
title: "Montage-AI Deployment Automation Guide"
summary: "Complete GitOps automation with Flux, Tekton, and Git SHA tagging"
date: 2025-12-04
---

# Montage-AI Deployment Automation Guide

## 🎯 Overview

This guide implements **production-grade GitOps automation** for montage-ai:

### What This Solves
- ❌ **Old:** Manual `kubectl apply`, `:latest` tag drift, no reproducible builds
- ✅ **New:** Fully automated Git → Build → Deploy pipeline with Git SHA tagging

### Architecture
```
Git Push (GitHub)
  ↓ (webhook)
Tekton Pipeline
  ├─ Clone repo
  ├─ Extract Git SHA (e.g., abc1234)
  ├─ Build image with Kaniko (cached layers)
  └─ Push to registry as :abc1234 + :latest
        ↓
Flux ImageRepository (scans registry every 1min)
  ↓ (detects new SHA tag)
Flux ImagePolicy (selects latest alphabetical)
  ↓
Flux ImageUpdateAutomation
  ├─ Updates deployment.yaml with new tag
  ├─ Commits to Git (main branch)
  └─ Triggers Kustomization reconcile
        ↓
K3s Cluster deploys new version
```

---

## 🚀 Quick Start (Step-by-Step)

### Step 1: Fix Current Deployment (IMMEDIATE)

```bash
# Run this NOW to fix the selector immutable issue
cd /home/matthias/fluxibri_core/deploy/k3s/montage-ai
./FIX_DEPLOYMENT_NOW.sh
```

This will:
1. Delete deployment (keeping pods alive)
2. Force Flux reconcile
3. Wait for new deployment to be ready

---

### Step 2: Deploy Kaniko Cache PVC

```bash
# Create persistent cache for faster builds
kubectl apply -f /home/matthias/fluxibri_core/deploy/k3s/tekton/storage/kaniko-cache-pvc.yaml

# Verify
kubectl -n tekton-pipelines get pvc kaniko-cache
```

**Expected:** `STATUS: Bound` (50Gi PVC on local-path storage)

---

### Step 3: Deploy Enhanced Tekton Tasks/Pipelines

```bash
# Deploy cached Kaniko task
kubectl apply -f /home/matthias/fluxibri_core/deploy/k3s/tekton/tasks/kaniko-build-cached.yaml

# Deploy Git SHA tagging pipeline
kubectl apply -f /home/matthias/fluxibri_core/deploy/k3s/tekton/pipelines/build-with-git-sha.yaml

# Verify
kubectl -n tekton-pipelines get task kaniko-build-cached
kubectl -n tekton-pipelines get pipeline build-with-git-sha
```

---

### Step 4: Test Build with Git SHA Tagging

```bash
# Create a test build (adjust repo-url if needed)
kubectl create -f /home/matthias/fluxibri_core/deploy/k3s/tekton/runs/montage-ai-build-example.yaml

# Watch build progress
tkn pipelinerun logs -f -n tekton-pipelines $(kubectl -n tekton-pipelines get pr -o name | tail -1)

# Check result
kubectl -n tekton-pipelines get pr -l tekton.dev/pipeline=build-with-git-sha
```

**Expected Output:**
```
IMAGE_SHA_TAG: 192.168.1.16:30500/montage-ai:abc1234
IMAGE_DIGEST: sha256:...
GIT_COMMIT: abc1234567890...
```

---

### Step 5: Deploy Flux Image Automation

```bash
# Generate webhook token
WEBHOOK_TOKEN=$(openssl rand -hex 32)
echo "Webhook token: $WEBHOOK_TOKEN"

# Update secret in receiver.yaml
sed -i "s/CHANGEME_GENERATE_WITH_OPENSSL_RAND/$WEBHOOK_TOKEN/" \
  /home/matthias/fluxibri_core/deploy/k3s/montage-ai/flux-automation/receiver.yaml

# Apply Flux automation configs
kubectl apply -k /home/matthias/fluxibri_core/deploy/k3s/montage-ai/flux-automation/

# Verify
kubectl -n montage-ai get imagerepository,imagepolicy,imageupdateautomation,receiver
```

**Expected:**
```
NAME                                           LAST SCAN   TAGS
imagerepository.image.toolkit.fluxcd.io/...   1m          abc1234, def5678, ...

NAME                                        LATEST IMAGE
imagepolicy.image.toolkit.fluxcd.io/...     192.168.1.16:30500/montage-ai:abc1234

NAME                                                  LAST RUN
imageupdateautomation.image.toolkit.fluxcd.io/...    2025-12-04T12:00:00Z

NAME                                          AGE   READY   STATUS
receiver.notification.toolkit.fluxcd.io/...   1m    True    Receiver initialized
```

---

### Step 6: Configure Webhook (Optional but Recommended)

Eliminates 1-5 minute Flux polling delay.

**Get webhook URL:**
```bash
kubectl -n montage-ai get receiver montage-ai -o jsonpath='{.status.webhookPath}'
# Example output: /hook/montage-ai/abc123...
```

**In your Git repo (GitHub/GitLab):**
1. Go to Settings → Webhooks
2. Add webhook:
   - URL: `https://flux.fluxibri.lan/hook/montage-ai/<webhook-id>/<token>`
   - Content-Type: `application/json`
   - Events: `push`
3. Test webhook

**Result:** Every Git push triggers immediate Flux reconcile (no polling delay).

---

## 🔄 Daily Workflow

### Developer Workflow

```bash
# 1. Make changes to montage-ai code
git add .
git commit -m "feat: add new feature"
git push origin main
```

**What happens automatically:**

1. **Webhook triggers Tekton** (if configured) or CI/CD starts
2. **Tekton builds image:**
   ```
   - Git SHA: abc1234
   - Image: 192.168.1.16:30500/montage-ai:abc1234
   - Cached layers reused → Build time: 2-5 min (vs 10-15 min)
   ```
3. **Flux detects new image:**
   ```
   - ImageRepository scans registry (1min interval)
   - ImagePolicy selects abc1234 (latest alphabetical)
   ```
4. **Flux updates deployment:**
   ```
   - ImageUpdateAutomation commits to Git:
     "Update montage-ai image to :abc1234"
   - Kustomization reconciles
   ```
5. **K3s deploys:**
   ```
   - RollingUpdate: maxSurge=0, maxUnavailable=1 (for single GPU)
   - Old pod terminated, new pod starts
   - Liveness probe checks /api/status
   ```

**Total Time:** ~5-10 minutes (Git push → Production)

---

## 🎛️ Configuration Options

### Image Policy Strategies

**Current:** Alphabetical (for Git SHA tags)
```yaml
policy:
  alphabetical:
    order: asc
```

**Alternative:** Semantic Versioning (if you tag releases)
```yaml
policy:
  semver:
    range: '>=1.0.0'
```

### Build Cache Tuning

**Increase cache size** if building many large images:
```yaml
# In kaniko-cache-pvc.yaml
resources:
  requests:
    storage: 100Gi  # Default: 50Gi
```

**Disable cache** for debugging:
```yaml
# In build-with-git-sha.yaml
- name: CACHE_ENABLED
  value: "false"
```

### Deployment Strategy (Single GPU)

**Current:** RollingUpdate (safe for single GPU)
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 0          # Don't create extra pod (GPU conflict)
    maxUnavailable: 1    # Allow 1 pod down during update
```

**Alternative:** Blue/Green (zero downtime, requires 2 GPUs)
```yaml
# Deploy to "blue" environment
# Switch Service selector to "green" when ready
# Requires 2 GPU nodes or time-based switching
```

---

## 📊 Monitoring & Troubleshooting

### Check Image Automation Status

```bash
# Is Flux detecting new images?
kubectl -n montage-ai get imagerepository montage-ai -o yaml

# Which image is selected by policy?
kubectl -n montage-ai get imagepolicy montage-ai -o yaml

# When was deployment last updated?
kubectl -n montage-ai get imageupdateautomation montage-ai -o yaml
```

### Check Deployment Status

```bash
# Current image running
kubectl -n montage-ai get deploy montage-ai-web -o jsonpath='{.spec.template.spec.containers[0].image}'

# Rollout status
kubectl -n montage-ai rollout status deploy/montage-ai-web

# Recent events
kubectl -n montage-ai get events --sort-by='.lastTimestamp' | head -20
```

### Check Build Logs

```bash
# List recent builds
tkn pr list -n tekton-pipelines

# Get logs for latest build
tkn pr logs -f -n tekton-pipelines $(kubectl -n tekton-pipelines get pr -o name | tail -1)

# Check cache usage
kubectl -n tekton-pipelines exec -it <kaniko-pod> -- du -sh /cache/*
```

### Common Issues

**Issue:** Image not updating in deployment
```bash
# Check if ImageRepository is scanning
kubectl -n montage-ai describe imagerepository montage-ai

# Force reconcile
flux reconcile image repository montage-ai -n montage-ai
```

**Issue:** Build failing with "disk full"
```bash
# Check cache PVC usage
kubectl -n tekton-pipelines exec -it <kaniko-pod> -- df -h /cache

# Clean old cache layers
kubectl -n tekton-pipelines exec -it <kaniko-pod> -- rm -rf /cache/*
```

**Issue:** Webhook not triggering
```bash
# Check receiver status
kubectl -n montage-ai describe receiver montage-ai

# Test webhook manually
curl -X POST https://flux.fluxibri.lan/hook/montage-ai/<id>/<token> \
  -H "Content-Type: application/json" \
  -d '{"ref":"refs/heads/main"}'
```

---

## 🔒 Security Best Practices

### Image Signing (Optional but Recommended)

```bash
# Sign images with Cosign
cosign sign --key cosign.key 192.168.1.16:30500/montage-ai:abc1234

# Verify in deployment
kubectl patch deploy montage-ai-web -n montage-ai --type='json' \
  -p='[{"op": "add", "path": "/spec/template/metadata/annotations/cosign.sigstore.dev~1signature", "value": "true"}]'
```

### Registry Authentication

If using private registry with authentication:
```yaml
# In imagerepository.yaml
spec:
  secretRef:
    name: registry-credentials
```

### Webhook Security

```bash
# Rotate webhook token regularly
NEW_TOKEN=$(openssl rand -hex 32)
kubectl -n montage-ai patch secret webhook-token \
  --type='json' \
  -p='[{"op": "replace", "path": "/data/token", "value": "'$(echo -n $NEW_TOKEN | base64)'"}]'
```

---

## 📚 References

- [Flux Image Automation Guide](https://fluxcd.io/flux/guides/image-update/)
- [Tekton Kaniko Cache Best Practices](https://tekton.dev/blog/2023/11/02/speeding-up-container-image-builds-in-tekton-pipelines/)
- [Kaniko Documentation](https://github.com/GoogleContainerTools/kaniko)
- [Flux Receiver Webhooks](https://fluxcd.io/flux/components/notification/receivers/)

---

## 🎯 Next Steps

1. **Enable notifications:**
   ```bash
   # Create Slack/Matrix alert for failed builds
   kubectl apply -f /path/to/notification-provider.yaml
   ```

2. **Multi-environment setup:**
   ```bash
   # Separate dev/staging/prod with different ImagePolicies
   # dev: any SHA tag
   # staging: tags matching "v*-rc*"
   # prod: tags matching "v*" (no rc)
   ```

3. **Automated rollback:**
   ```bash
   # If new deployment fails health check, Flux can auto-revert
   # Configure in Kustomization spec
   ```

---

**Last Updated:** 2025-12-04
**Maintainer:** Matthias Fahsold
