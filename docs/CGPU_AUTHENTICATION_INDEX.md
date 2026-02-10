# CGPU Authentication Documentation Index

**Investigation Date**: February 10, 2026  
**Status**: ✅ Complete with Implementation Ready Code

---

## Quick Links

### For Users (Operators)
- **[CGPU_AUTHENTICATION.md](CGPU_AUTHENTICATION.md)** – Complete setup guide
  - Part 1: Local host authentication (15 min)
  - Part 2: Cluster deployment (5 min)
  - Troubleshooting guide included

### For Developers (Implementation)
- **[CGPU_WEBAPP_AUTH_IMPLEMENTATION.md](CGPU_WEBAPP_AUTH_IMPLEMENTATION.md)** – Ready-to-code guide
  - Flask backend routes with full code
  - HTML/JavaScript frontend templates
  - Kubernetes integration steps
  - Testing checklist

### For Decision Makers
- **[CGPU_AUTH_INVESTIGATION_SUMMARY.md](CGPU_AUTH_INVESTIGATION_SUMMARY.md)** – Executive summary
  - Architecture overview
  - Security assessment
  - Implementation roadmap
  - Q&A section

---

## What These Documents Cover

### CGPU_AUTHENTICATION.md (447 lines)

**Part 1: Local Host Authentication**
- Step-by-step Google Cloud project setup
- OAuth credential creation
- Local `cgpu connect` flow
- Session management and refresh

**Part 2: Cluster Deployment**
- Automated secret creation via script
- Manual kubectl commands
- Secret mounting and pod configuration
- Verification steps

**Part 3: Configuration Reference**
- Environment variables (CGPU_ENABLED, CGPU_HOST, etc.)
- Default values and overrides
- Timeout configurations

**Part 4: Web UI Architecture**
- Current state (API endpoints exist)
- Proposed authentication UI
- OAuth flow via web interface
- Automatic secret management

**Session Lifecycle**
- 7-day session validity period
- Refresh requirements
- Cluster considerations

**Security & Troubleshooting**
- Credential scope limitations
- Pod security best practices
- Common errors and fixes

---

### CGPU_WEBAPP_AUTH_IMPLEMENTATION.md (622 lines)

**Architecture Diagram**
- OAuth flow with state tracking
- Callback handling
- Secret creation and pod restart orchestration

**Backend Implementation (Full Python Code)**
```python
# routes/settings.py - 250+ lines
- POST /api/settings/cgpu/authenticate/start
- GET /api/settings/cgpu/authenticate/callback
- GET /api/settings/cgpu/authenticate/status/<session_id>
- GET /api/settings/cgpu/status
```

**Frontend Implementation (HTML + JavaScript)**
```html
<!-- settings.html snippet -->
- "🔐 Authenticate with Google" button
- Status display (authenticated/not authenticated)
- Popup-based OAuth flow
- Real-time status polling
```

**Integration Points**
1. Kubernetes secret management via kubectl
2. Pod restart orchestration
3. Session tracking
4. Error handling and recovery

**Testing Checklist**
- Frontend button appearance
- OAuth URL generation
- Callback capture
- Secret creation
- Pod restart verification
- Error scenarios

---

### CGPU_AUTH_INVESTIGATION_SUMMARY.md (286 lines)

**Key Findings**
- Two-layer authentication model (host + cluster)
- Multiple deployment methods available
- Current blockers identified with solutions

**Implementation Roadmap**
- Immediate (this week)
- Short-term (this month)
- Medium-term (Q1 2026)

**Security Model**
- What's protected ✅
- What needs consideration ⚠️

**Q&A Section**
- How to authenticate locally?
- How to use in cluster?
- Can users authenticate from web UI?
- Session validity and refresh?
- Missing credentials behavior?

---

## Files Modified

### Code Enhancements
- `src/montage_ai/cgpu_utils.py`
  - Added: `_maybe_log_missing_cgpu_credentials()` function
  - Added: Better error logging for `cgpu status` failures
  - Result: Better diagnostics in cluster logs

### Documentation Updates
- `docs/KUBERNETES_RUNBOOK.md`
  - Added: CGPU secret refresh instructions
  - Added: Link to `cgpu-refresh-session.sh`

- `docs/operations/deployment-2026-02-10.md`
  - Added: CGPU credentials requirement note
  - Added: Refresh workflow documentation

---

## Usage Scenarios

### Scenario 1: Enable CGPU in Cluster (Operations)
**Time Required**: 5 minutes

```bash
# Step 1: Ensure local cgpu setup (one-time)
npm install -g cgpu
cgpu connect  # Interactive OAuth flow

# Step 2: Create cluster secret
./scripts/ops/cgpu-refresh-session.sh

# Step 3: Verify
kubectl -n montage-ai logs -f deploy/cgpu-server | grep "session\|error"

# Step 4: Test
kubectl -n montage-ai exec deploy/montage-ai-worker -- cgpu status
```

→ **Read**: CGPU_AUTHENTICATION.md, Parts 1-2

---

### Scenario 2: Add Web UI Authentication (Development)
**Time Required**: 4-6 hours for full implementation

```bash
# Step 1: Review architecture
# → Read: CGPU_WEBAPP_AUTH_IMPLEMENTATION.md

# Step 2: Create backend routes
# → Copy: Python code from guide → src/montage_ai/web_ui/routes/settings.py

# Step 3: Update settings page
# → Copy: HTML/JS from guide → src/montage_ai/web_ui/templates/settings.html

# Step 4: Register blueprint in app
# → Update: src/montage_ai/web_ui/app.py (register settings_bp)

# Step 5: Test OAuth flow
# → Follow: Testing checklist in guide

# Step 6: Deploy and verify
# → Run: Local test → Cluster test → Production
```

→ **Read**: CGPU_WEBAPP_AUTH_IMPLEMENTATION.md (all sections)

---

### Scenario 3: Troubleshoot Missing CGPU (Support)
**Time Required**: 10 minutes

```bash
# Check pod logs
kubectl -n montage-ai logs deploy/cgpu-server

# Check secret exists
kubectl -n montage-ai get secret cgpu-credentials

# Check worker can access cgpu
kubectl -n montage-ai exec deploy/montage-ai-worker -- cgpu status

# Refresh if expired
./scripts/ops/cgpu-refresh-session.sh
```

→ **Read**: CGPU_AUTHENTICATION.md, Troubleshooting section

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Montage AI + CGPU                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌─────────────────────────┐  │
│  │   Host Machine   │         │  Kubernetes Cluster     │  │
│  ├──────────────────┤         ├─────────────────────────┤  │
│  │ $ cgpu connect   │ ─────→  │ cgpu-credentials secret │  │
│  │                  │         │ (config.json +          │  │
│  │ ~/.config/cgpu   │         │  session.json)          │  │
│  │ ├─ config.json   │         │                         │  │
│  │ └─ session.json  │         └──────┬──────────────────┘  │
│  └──────────────────┘                │                      │
│                                      ↓                      │
│                          ┌──────────────────┐              │
│                          │  cgpu-server pod │              │
│                          ├──────────────────┤              │
│                          │ - Reads secret   │              │
│                          │ - Starts serve   │              │
│                          │ - Ready for jobs │              │
│                          └──────────────────┘              │
│                                      ↑                      │
│                          ┌──────────────────┐              │
│                          │  worker pods     │              │
│                          ├──────────────────┤              │
│                          │ - Use cgpu-serve │              │
│                          │ - Run LLM tasks  │              │
│                          │ - GPU jobs       │              │
│                          └──────────────────┘              │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Web UI (Optional - Proposed)                 │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  Settings → "🔐 Authenticate with Google"           │  │
│  │    ↓ (Opens OAuth popup)                             │  │
│  │  Backend creates secret + restarts pods             │  │
│  │  User sees: "✅ Authenticated (expires: 2026-02-17)" │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Session Lifecycle

```
Day 0 (Create)     cgpu connect
       │            ↓
       │       Session created
       │       ~/.config/cgpu/state/session.json
       │       Kubernetes secret cgpu-credentials
       │
Days 1-6           ✅ Active - All CGPU features work
       │            • CGPU LLM tasks
       │            • GPU upscaling
       │            • Transcription
       │            • Stabilization
       │
Day 6  (Alert)     ⚠️  Approaching expiry (reminder)
       │
Day 7  (Expiry)    ❌ Session invalid
       │            cgpu status: "Missing OAuth credentials"
       │
Action             Run: cgpu connect (local) OR
                   Run: ./scripts/ops/cgpu-refresh-session.sh
```

---

## Implementation Checklist

### Phase 1: Enable CGPU (Immediate)
- [ ] Read CGPU_AUTHENTICATION.md Part 1
- [ ] Run `cgpu connect` locally
- [ ] Read CGPU_AUTHENTICATION.md Part 2
- [ ] Run `./scripts/ops/cgpu-refresh-session.sh`
- [ ] Verify: `kubectl exec deploy/montage-ai-worker -- cgpu status`

### Phase 2: Test CGPU Jobs (This Week)
- [ ] Submit transcription job via API
- [ ] Submit upscaling job via API
- [ ] Verify jobs complete successfully
- [ ] Check pod logs for any warnings

### Phase 3: Add Web UI (This Month)
- [ ] Read CGPU_WEBAPP_AUTH_IMPLEMENTATION.md
- [ ] Create `routes/settings.py` with OAuth handlers
- [ ] Update `settings.html` with UI components
- [ ] Register blueprint in `app.py`
- [ ] Follow testing checklist

### Phase 4: Production Hardening (Q1 2026)
- [ ] Implement session refresh automation
- [ ] Set up monitoring for CGPU health
- [ ] Plan migration to service account auth
- [ ] Document runbooks for operations team

---

## Support & References

### Official Resources
- **cgpu GitHub**: https://github.com/RohanAdwankar/cgpu
- **Colab API Docs**: https://colab.research.google.com/
- **Kubernetes Secrets**: https://kubernetes.io/docs/concepts/configuration/secret/

### Project Resources
- **Existing cgpu-setup.md**: `docs/cgpu-setup.md`
- **Kubernetes Runbook**: `docs/KUBERNETES_RUNBOOK.md`
- **Deployment Notes**: `docs/operations/deployment-2026-02-10.md`

---

## Summary

This investigation provides:

✅ **Complete Documentation** (1,355 lines across 3 files)
✅ **Ready-to-Implement Code** (Flask + JavaScript)
✅ **Security Analysis** (protections + considerations)
✅ **Implementation Roadmap** (3-phase plan)
✅ **Testing Guidance** (checklists + verification steps)

**Next Step**: Read [CGPU_AUTHENTICATION.md](CGPU_AUTHENTICATION.md) and follow the guide for your scenario.
