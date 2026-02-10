# CGPU Authentication Guide

## Overview

This document covers how to authenticate with cgpu (Cloud GPU) for both **local development** and **web-based cluster deployments**. Authentication enables Montage AI to offload computation (transcription, upscaling, stabilization, LLM tasks) to Google Colab via cgpu.

---

## Architecture

### CGPU Stack

```
User (Host or Webapp) → cgpu Client → cgpu serve (localhost:8080) → Colab/Google API
```

1. **cgpu Client**: Command-line tool installed locally that manages OAuth & session
2. **cgpu serve**: Local HTTP server that exposes LLM endpoints (Gemini) and compute jobs
3. **Colab/Google API**: Remote backend managed by cgpu

### Authentication Flow

1. **One-time Setup**: Create Google Cloud project + OAuth credentials
2. **Interactive Login**: Run `cgpu connect` → browser OAuth → session stored locally
3. **Session Persistence**: Session file (`~/.config/cgpu/state/session.json`) reused for 7 days
4. **Cluster Deployment**: Secret mounted into worker pods → session propagated

---

## Part 1: Local Host Authentication

### Prerequisites

1. cgpu installed (Node.js CLI):
   ```bash
   npm install -g cgpu
   ```

2. Google Account (any account works; doesn't need special permissions)

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create **New Project** → Name: `montage-ai-cgpu`
3. Wait for project creation (2-3 seconds)

### Step 2: Enable Colab API

1. In project, search for **Colaboratory API**
2. Click **Enable**
3. (Optional) Enable **Google AI for Developers** if using Gemini API directly

### Step 3: Create OAuth Credentials

1. Go to **APIs & Services → OAuth consent screen**
2. Choose **External** (allows any Google account)
3. Fill app info:
   - **App name**: `cgpu`
   - **User support email**: Your email
   - **Developer contact**: Your email
4. Click **Save and Continue**
5. **Scopes** tab:
   - Click **Add or Remove Scopes**
   - Add: `https://www.googleapis.com/auth/colaboratory`
   - Also add: `.../auth/userinfo.email` and `.../auth/userinfo.profile`
   - Click **Update** → **Save and Continue**
6. **Test Users** tab:
   - Click **Add Users**
   - Enter your Google email
   - Click **Save and Continue**

7. Go to **APIs & Services → Credentials**
8. Click **Create Credentials → OAuth client ID**
9. **Application type**: Desktop app
10. **Name**: `cgpu-client`
11. Click **Create**
12. **Copy** the **Client ID** and **Client Secret**

### Step 4: Create Local Config

Create `~/.config/cgpu/config.json`:

```bash
mkdir -p ~/.config/cgpu
cat > ~/.config/cgpu/config.json << 'EOF'
{
  "clientId": "YOUR_CLIENT_ID.apps.googleusercontent.com",
  "clientSecret": "YOUR_CLIENT_SECRET",
  "colabApiDomain": "https://colab.research.google.com",
  "colabGapiDomain": "https://colab.googleapis.com"
}
EOF
```

Replace `YOUR_CLIENT_ID` and `YOUR_CLIENT_SECRET` with values from Step 3.

### Step 5: Authenticate via OAuth

```bash
cgpu connect
```

This will:
1. Print a browser link
2. Open your browser to Google OAuth
3. Ask for permission to use Colab
4. Create session file at `~/.config/cgpu/state/session.json` (valid for ~7 days)

**Troubleshooting**:
- If callback fails: ensure localhost:8000 is not in use; cgpu will retry
- Session expired? Run `cgpu connect` again

### Step 6: Verify

```bash
cgpu status
```

Expected output:
```
✅ cgpu status: OK
Authenticated: yes
Expires: 2026-02-17 10:23:45 UTC
```

### Step 7: Start cgpu serve (Optional Local Server)

For local testing:

```bash
cgpu serve --host 127.0.0.1 --port 8080
```

This exposes an HTTP server that Montage AI can call directly (instead of via CLI).

---

## Part 2: Web-based Cluster Authentication

### Prerequisite: Valid Session on Host

Ensure you've completed Part 1 (local `cgpu connect` succeeded).

### Option A: Refresh Secret from Host

**Recommended** – automates secret creation and pod restart.

```bash
./scripts/ops/cgpu-refresh-session.sh
```

This script:
1. Runs `cgpu connect` (interactive OAuth)
2. Reads config + session files from `~/.config/cgpu`
3. Creates Kubernetes secret `cgpu-credentials` in `montage-ai` namespace
4. Restarts worker + cgpu-server pods to pick up new session

**What's in the secret**:
- `config.json`: OAuth client credentials
- `session.json`: Current session token (expires in ~7 days)

### Option B: Manual Secret Creation

If you have config/session files but `cgpu-refresh-session.sh` isn't available:

```bash
kubectl -n montage-ai create secret generic cgpu-credentials \
  --from-file=config.json=$HOME/.config/cgpu/config.json \
  --from-file=session.json=$HOME/.config/cgpu/state/session.json \
  --dry-run=client -o yaml | kubectl apply -f -
```

Then restart pods:

```bash
kubectl -n montage-ai rollout restart deploy/montage-ai-worker deploy/cgpu-server
```

### Option C: Web UI Configuration Page (Future Enhancement)

**Status**: Not yet implemented.

**Proposed flow**:
1. User navigates to **Settings → Cloud GPU**
2. Click **"Authenticate with Google"**
3. Opens popup with `cgpu connect` flow
4. On completion, webapp creates secret automatically
5. Workers updated in background

---

## Part 3: Cluster Deployment

### Architecture

When pods start, they expect a secret at `/cgpu-secrets` with:
- `config.json` → copied to `/root/.config/cgpu/config.json` (or `/home/montage/.config/cgpu/`)
- `session.json` → copied to `/root/.config/cgpu/state/session.json`

Then they:
1. Run `cgpu connect` internally (if session invalid)
2. Start `cgpu serve` on port 8080
3. Expose service at `cgpu-server:8080` for worker pods

### Files Modified by Deployment

- [deploy/k3s/base/cgpu-server.yaml](deploy/k3s/base/cgpu-server.yaml) – cgpu-server pod
- [deploy/k3s/base/worker.yaml](deploy/k3s/base/worker.yaml) – worker pods with cgpu support
- [deploy/k3s/base/kustomization.yaml](deploy/k3s/base/kustomization.yaml) – image substitution for init container

### Checking Cluster Status

```bash
# Check if secret exists
kubectl -n montage-ai get secret cgpu-credentials

# Check cgpu-server logs
kubectl -n montage-ai logs deploy/cgpu-server -f

# Check if cgpu is available in worker
kubectl -n montage-ai exec deploy/montage-ai-worker -- cgpu status

# Port-forward to cgpu-server (for testing)
kubectl -n montage-ai port-forward svc/cgpu-server 8080:8080
curl http://localhost:8080/status  # (if endpoint exists)
```

### Troubleshooting

**Issue**: `cgpu status` in pod returns "Missing Colab OAuth credentials"

**Solutions**:
1. Secret may not exist or not mounted correctly
2. Session may have expired (7-day limit)
3. Fix: Run `./scripts/ops/cgpu-refresh-session.sh` to refresh

**Issue**: Pod stuck trying to authenticate

**Solutions**:
1. Manually set `CGPU_ENABLED=false` in pod env to disable
2. Or: Delete pod → let Kubernetes recreate
3. Or: Run `cgpu-refresh-session.sh` again

---

## Part 4: Web UI Integration (Architecture)

### Current State

The web UI checks CGPU availability but does **not** provide authentication UI.

#### Endpoints

**Check CGPU status** (GET):
```
GET /api/cgpu/status
```

Response:
```json
{
  "cgpu_available": true,
  "cgpu_gpu_available": true,
  "gpu_info": "NVIDIA A100 (40GB)",
  "llm_available": true,
  "message": "OK"
}
```

**Submit transcription job** (POST):
```
POST /api/cgpu/transcribe
{
  "video_path": "/data/input/video.mp4"
}
```

**Submit upscaling job** (POST):
```
POST /api/cgpu/upscale
{
  "video_path": "/data/input/video.mp4"
}
```

**Submit stabilization job** (POST):
```
POST /api/cgpu/stabilize
{
  "video_path": "/data/input/video.mp4"
}
```

### Webapp Files

- [src/montage_ai/web_ui/app.py](src/montage_ai/web_ui/app.py#L1825) – `/api/cgpu/status` endpoint
- [src/montage_ai/cgpu_utils.py](src/montage_ai/cgpu_utils.py) – CGPU availability checks + logging
- [src/montage_ai/web_ui/templates/settings.html](src/montage_ai/web_ui/templates/settings.html) – Settings page (no CGPU auth UI yet)

### Proposed Web UI Authentication Flow

**File**: `src/montage_ai/web_ui/routes/settings.py` (to be created)

**Endpoint** (POST):
```
POST /api/cgpu/authenticate
```

**Request**:
```json
{
  "action": "start_oauth",  // or "refresh_session"
}
```

**Response**:
```json
{
  "oauth_url": "https://accounts.google.com/o/oauth2/auth?...",
  "session_id": "abc123",
  "status": "awaiting_callback"
}
```

**Callback** (after user authenticates in browser):
```
GET /api/cgpu/authenticate/callback?code=<auth_code>
```

**Backend**:
1. Exchange code for session via cgpu CLI
2. Create secret in cluster
3. Restart pods
4. Return success

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CGPU_ENABLED` | `false` | Enable LLM tasks (Story Engine, Creative Director) |
| `CGPU_GPU_ENABLED` | `false` | Enable GPU tasks (upscaling, stabilization, transcription) |
| `CGPU_HOST` | `127.0.0.1` | cgpu serve hostname |
| `CGPU_PORT` | `8090` | cgpu serve port |
| `CGPU_MODEL` | `gemini-2.0-flash` | LLM model to use |
| `CGPU_TIMEOUT` | `1200` | Timeout for cgpu operations (seconds) |
| `CGPU_MAX_CONCURRENCY` | `1` | Max concurrent cgpu jobs |
| `CGPU_STATUS_TIMEOUT` | `30` | Timeout for `cgpu status` check |
| `CGPU_GPU_CHECK_TIMEOUT` | `120` | Timeout for GPU availability check |

---

## Session File Lifecycle

### Location

```
~/.config/cgpu/state/session.json
```

### Format

```json
{
  "accessToken": "...",
  "refreshToken": "...",
  "tokenExpiry": "2026-02-17T10:23:45Z",
  "userId": "user@gmail.com"
}
```

### Expiry

- Default: **7 days** after creation
- Can refresh: Run `cgpu connect` again before expiry
- Kubernetes secret: Also expires after 7 days → needs refresh

### Best Practice

- Refresh monthly: `./scripts/ops/cgpu-refresh-session.sh`
- Monitor pod logs: Look for "Missing Colab OAuth credentials"
- Alert on failure: Set up monitoring on secret age

---

## Security Considerations

### Secrets Management

✅ **Good**:
- Credentials stored in Kubernetes secret (encrypted at rest with etcd)
- Secret mounted read-only into pods
- Session file not committed to Git (in `.gitignore`)

⚠️ **Considerations**:
- Client ID/secret visible to anyone with `kubectl` access
- Consider using external secret management (Vault, AWS Secrets Manager)
- In production: use service account impersonation instead of OAuth

### Credential Scope

- OAuth scope: **Colab only** (narrow, safer than full Google Cloud access)
- User data: Not accessed by cgpu (Colab is stateless)

### Pod Security

- Init container runs as root to set up config files
- Main container can run as unprivileged user (future enhancement)

---

## Roadmap

### Q1 2026

- [ ] Web UI authentication page (Settings → Cloud GPU)
- [ ] In-browser OAuth redirect handler
- [ ] Automatic secret creation from UI
- [ ] Session refresh button in UI

### Q2 2026

- [ ] Service account authentication (replace user OAuth)
- [ ] Vault integration for secret storage
- [ ] Monitoring + alerts for session expiry
- [ ] cgpu-server high availability (replicas)

---

## References

- **cgpu Docs**: https://github.com/RohanAdwankar/cgpu
- **Google Colab API**: https://colab.research.google.com/
- **Kubernetes Secrets**: https://kubernetes.io/docs/concepts/configuration/secret/
- **OAuth 2.0**: https://tools.ietf.org/html/rfc6749

---

## Support

For issues or questions:

1. Check logs: `kubectl -n montage-ai logs deploy/cgpu-server`
2. Verify config: `kubectl -n montage-ai describe secret cgpu-credentials`
3. Test locally: `cgpu status` in terminal
4. Open issue with logs: https://github.com/mfahsold/montage-ai/issues
