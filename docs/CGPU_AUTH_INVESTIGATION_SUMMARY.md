# CGPU Authentication Investigation Summary

**Date**: February 10, 2026  
**Status**: ✅ Investigation Complete  

---

## Executive Summary

Investigated how Montage AI can authenticate with cgpu (Cloud GPU) from both the **host machine** and the **web application**. Created comprehensive guides for both scenarios with implementation roadmaps.

**Key Finding**: Two-layer authentication model is optimal:

1. **Host Layer** (local machine): User runs `cgpu connect` once, stores credentials
2. **Cluster Layer** (Kubernetes): Webapp automates secret creation and pod restart

---

## What Was Investigated

### 1. Current State Analysis

**Host-based CGPU**:
- ✅ `cgpu` CLI tool available
- ✅ `cgpu connect` runs OAuth flow successfully
- ✅ Session persisted in `~/.config/cgpu/`
- ❌ Credentials missing from current environment

**Cluster-based CGPU**:
- ✅ Job manifests configured to mount `cgpu-credentials` secret
- ✅ Init containers set up to copy config/session files
- ✅ cgpu-server pod ready to run
- ❌ Secret doesn't exist (no credentials ever created)

**Web UI Integration**:
- ✅ `/api/cgpu/status` endpoint exists
- ✅ CGPU availability checks working
- ✅ Job submission endpoints prepared (transcribe, upscale, stabilize)
- ❌ No OAuth UI or credential management in webapp

### 2. Code Architecture Review

**Key Files**:
- `src/montage_ai/cgpu_utils.py` – Availability detection, session validation
- `src/montage_ai/web_ui/app.py` – API endpoints for CGPU operations (lines 1825+)
- `deploy/k3s/base/cgpu-server.yaml` – Pod definition with secret mounts
- `deploy/k3s/base/worker.yaml` – Worker pods with cgpu support
- `scripts/ops/cgpu-refresh-session.sh` – Helper script for secret refresh
- `docs/cgpu-setup.md` – Setup documentation (already exists)

**Authentication Flow**:
```
OAuth Credentials (Google Cloud)
        ↓
Local config.json (~/.config/cgpu/)
        ↓
cgpu CLI performs OAuth + creates session.json
        ↓
Kubernetes secret (config.json + session.json mounted)
        ↓
Pod reads files → starts cgpu serve → ready for LLM/GPU tasks
```

### 3. Security Model

**Existing Protections**:
- ✅ Credentials stored in Kubernetes secret (encrypted at rest)
- ✅ Secret mounted read-only into pods
- ✅ RBAC can restrict who views the secret
- ✅ OAuth scope limited to Colab (not full Google Cloud)

**Gaps Identified**:
- ⚠️ Client ID/secret visible to anyone with `kubectl` access
- ⚠️ No automatic session refresh (7-day expiry)
- ⚠️ No monitoring/alerts for expired credentials

---

## Deliverables Created

### 1. CGPU Authentication Guide

**File**: [docs/CGPU_AUTHENTICATION.md](docs/CGPU_AUTHENTICATION.md)

**Covers**:
- Part 1: Local host authentication (step-by-step)
- Part 2: Cluster deployment (manual + automatic)
- Part 3: Environment variables and configuration
- Part 4: Web UI architecture (current + proposed)
- Session lifecycle and best practices
- Security considerations
- Troubleshooting guide

**Length**: ~450 lines, complete reference document

### 2. Web UI Implementation Guide

**File**: [docs/CGPU_WEBAPP_AUTH_IMPLEMENTATION.md](docs/CGPU_WEBAPP_AUTH_IMPLEMENTATION.md)

**Covers**:
- Backend OAuth routes implementation (Flask blueprints)
- Frontend UI components (settings page)
- OAuth flow with callback handling
- Kubernetes secret creation automation
- Pod restart orchestration
- Testing checklist
- Security considerations

**Code Examples**:
- Full Python implementation for `routes/settings.py`
- JavaScript/HTML for settings template
- Integration points with existing code

**Length**: ~550 lines, ready-to-implement

### 3. Updated Documentation

**Enhanced Files**:
- [docs/cgpu-setup.md](docs/cgpu-setup.md) – (already comprehensive, no changes needed)
- [docs/KUBERNETES_RUNBOOK.md](docs/KUBERNETES_RUNBOOK.md) – Added cgpu secret refresh guidance
- [docs/operations/deployment-2026-02-10.md](docs/operations/deployment-2026-02-10.md) – Documented cgpu credential requirement

### 4. Code Improvements

**File**: [src/montage_ai/cgpu_utils.py](src/montage_ai/cgpu_utils.py)

Added:
- One-time warning for missing credentials (no spam)
- stderr logging on cgpu status failures
- Helper function `_maybe_log_missing_cgpu_credentials()`

**Result**: Better diagnostics when CGPU fails in cluster

---

## Authentication Methods Compared

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Host CLI** (`cgpu connect`) | Simple, interactive, well-tested | Requires local machine access | Dev environments |
| **Script Automation** (`cgpu-refresh-session.sh`) | Hands-off after setup, documented | Doesn't handle expiry | Ops/DevOps teams |
| **Web UI (Proposed)** | User-friendly, no CLI needed | Requires implementation | End users, clusters |
| **Service Account** (Future) | Most secure for production | Complex setup, Google Cloud billing | Production clusters |

---

## Current Blockers (Cluster)

1. **No credentials created yet**
   - Solution: Run `./scripts/ops/cgpu-refresh-session.sh` from host with valid cgpu setup

2. **7-day session expiry**
   - Solution: Refresh monthly or implement auto-refresh in web UI

3. **No web UI for authentication**
   - Solution: Implement using guide in CGPU_WEBAPP_AUTH_IMPLEMENTATION.md

---

## Implementation Roadmap

### Immediate (This Week)
- [ ] Run `./scripts/ops/cgpu-refresh-session.sh` to create initial secret
- [ ] Verify pods restart and CGPU becomes available
- [ ] Test CGPU jobs (transcribe/upscale) end-to-end

### Short-term (This Month)
- [ ] Implement backend routes (`routes/settings.py`)
- [ ] Add frontend OAuth UI to settings page
- [ ] Test OAuth flow in dev cluster
- [ ] Document user-facing authentication process

### Medium-term (Q1 2026)
- [ ] Session refresh automation (alert 7 days before expiry)
- [ ] Monitoring/observability for CGPU health
- [ ] Service account authentication (deprecate user OAuth)

### Long-term (Q2+ 2026)
- [ ] Vault integration for secret management
- [ ] CGPU-server high availability (replicas)
- [ ] Custom credential provider plugins

---

## Quick Start: Enable CGPU Today

### For Cluster:

```bash
# 1. Setup local cgpu (one-time)
npm install -g cgpu
mkdir -p ~/.config/cgpu
# Follow docs/CGPU_AUTHENTICATION.md Part 1

# 2. Create cluster secret
./scripts/ops/cgpu-refresh-session.sh

# 3. Verify
kubectl -n montage-ai logs -f deploy/cgpu-server | grep "OAuth\|session\|error"

# 4. Test
kubectl -n montage-ai exec deploy/montage-ai-worker -- cgpu status
```

### For Web UI (Future):

After implementing the guide:

```bash
# User navigates to: http://localhost:5000/settings
# Clicks: "🔐 Authenticate with Google"
# Browser redirects to Google OAuth
# On success: Secret created, pods restarted
# User sees: "✅ Authenticated (expires: 2026-02-17)"
```

---

## Files Reference

| Document | Purpose | Status |
|----------|---------|--------|
| [CGPU_AUTHENTICATION.md](../docs/CGPU_AUTHENTICATION.md) | Complete auth guide (host + cluster) | ✅ Complete |
| [CGPU_WEBAPP_AUTH_IMPLEMENTATION.md](../docs/CGPU_WEBAPP_AUTH_IMPLEMENTATION.md) | Web UI implementation guide | ✅ Complete |
| [cgpu-setup.md](../docs/cgpu-setup.md) | Original setup docs | ✅ Existing |
| [KUBERNETES_RUNBOOK.md](../docs/KUBERNETES_RUNBOOK.md) | Cluster operations | ✅ Updated |
| [deployment-2026-02-10.md](../docs/operations/deployment-2026-02-10.md) | Deployment notes | ✅ Updated |

---

## Next Actions

1. **Immediate**: User provides Google OAuth credentials (from Part 1)
2. **Short-term**: Run `cgpu-refresh-session.sh` to create secret
3. **Testing**: Verify CGPU works end-to-end in cluster
4. **Implementation**: Follow web UI guide to add authentication UI
5. **Monitoring**: Set up alerts for session expiry (7-day lifecycle)

---

## Questions Answered

### Q: How do we authenticate cgpu on a host machine?
**A**: Run `cgpu connect` locally, which:
1. Opens browser to Google OAuth
2. Creates `~/.config/cgpu/config.json` (OAuth credentials)
3. Creates `~/.config/cgpu/state/session.json` (session token)

### Q: How do we use those credentials in the cluster?
**A**: Create a Kubernetes secret from those files:
```bash
kubectl create secret generic cgpu-credentials \
  --from-file=config.json=~/.config/cgpu/config.json \
  --from-file=session.json=~/.config/cgpu/state/session.json
```

### Q: Can users authenticate from the web UI?
**A**: Yes, but requires implementation. Full guide provided in CGPU_WEBAPP_AUTH_IMPLEMENTATION.md

### Q: How long are sessions valid?
**A**: ~7 days. After expiry, run `cgpu connect` again to refresh.

### Q: What happens when credentials are missing?
**A**: CGPU jobs silently fall back to local processing. Logs show warnings.

### Q: Can multiple users share cgpu credentials?
**A**: Yes, one secret per cluster. Consider service account auth for production.

---

## Conclusion

CGPU authentication is well-architected with multiple layers:

1. **Host Layer**: User OAuth (already working, documented)
2. **Cluster Layer**: Secret distribution (existing, just needs credentials)
3. **Web Layer**: Proposed UI to automate credential management

The investigation has produced:
- ✅ Complete authentication guides
- ✅ Ready-to-implement web UI code
- ✅ Troubleshooting documentation
- ✅ Security best practices
- ✅ Roadmap for future enhancements

Next step: Apply the guides to enable CGPU end-to-end.
