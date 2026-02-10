# CGPU Web UI Authentication Implementation Guide

## Goal

Enable users to authenticate with CGPU directly from the Montage AI web interface (settings page) without needing command-line access.

## Current State

### What Works

- ✅ CGPU availability detection (`/api/cgpu/status` endpoint)
- ✅ CGPU job submission (transcribe, upscale, stabilize)
- ✅ Status display in system dashboard

### What's Missing

- ❌ OAuth flow UI (login button, callback handling)
- ❌ Automatic secret creation in cluster
- ❌ Session refresh from web UI
- ❌ Credential validation feedback

---

## Architecture

### Flow Diagram

```text
User clicks "Authenticate" in Settings
        ↓
[1] Webapp generates OAuth link + session_id
        ↓
User redirected to Google OAuth (in new window)
        ↓
User grants permission to cgpu
        ↓
[2] Browser callback to /api/cgpu/authenticate/callback?code=...&session_id=...
        ↓
[3] Webapp exchanges code for session (via cgpu CLI)
        ↓
[4] Webapp creates/updates Kubernetes secret cgpu-credentials
        ↓
[5] Webapp signals pod restart
        ↓
[6] Pods pick up new credentials and restart cgpu-server
        ↓
Success notification in UI
```

### Components

1. **Frontend** (`settings.html`): OAuth button + status display
2. **Backend Routes** (`routes/settings.py`): OAuth endpoints
3. **Cluster Integration** (`cluster_integration.py`): Secret management
4. **CGPUManager** (`cgpu_jobs/base.py`): OAuth state management

---

## Implementation Steps

### Step 1: Backend OAuth Routes

**File**: `src/montage_ai/web_ui/routes/settings.py` (new file)

```python
"""
Settings routes including CGPU authentication.
"""
from flask import Blueprint, request, jsonify, redirect, url_for
from datetime import datetime
import uuid
import tempfile
import json
import os
import subprocess
from pathlib import Path

from ...logger import logger
from ...config import get_settings

settings_bp = Blueprint('settings', __name__, url_prefix='/api/settings')

# In-memory OAuth session tracking (replace with Redis in production)
_oauth_sessions = {}

@settings_bp.route('/cgpu/authenticate/start', methods=['POST'])
def start_cgpu_oauth():
    """
    Start CGPU OAuth flow.
    
    Returns OAuth URL and session ID for polling.
    """
    try:
        # Check if cgpu config exists
        config_path = Path.home() / ".config" / "cgpu" / "config.json"
        if not config_path.exists():
            return jsonify({
                "status": "error",
                "message": "cgpu config not found. Set up locally first via 'cgpu connect'."
            }), 400
        
        # Read config to get client ID
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        client_id = config.get('clientId')
        if not client_id:
            return jsonify({
                "status": "error",
                "message": "Invalid cgpu config: missing clientId"
            }), 400
        
        # Generate session tracking ID
        session_id = str(uuid.uuid4())
        
        # Build OAuth URL (cgpu's internal flow)
        oauth_url = (
            "https://accounts.google.com/o/oauth2/auth?"
            f"client_id={client_id}&"
            "response_type=code&"
            "scope=https://www.googleapis.com/auth/colaboratory&"
            f"redirect_uri={request.host_url.rstrip('/')}/api/settings/cgpu/authenticate/callback&"
            "state=" + session_id
        )
        
        # Track session
        _oauth_sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "status": "awaiting_callback"
        }
        
        return jsonify({
            "status": "ok",
            "session_id": session_id,
            "oauth_url": oauth_url,
            "message": "Redirect user to oauth_url. They will be redirected back after authentication."
        })
    
    except Exception as e:
        logger.exception("Failed to start CGPU OAuth: %s", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@settings_bp.route('/cgpu/authenticate/callback', methods=['GET'])
def cgpu_oauth_callback():
    """
    Handle OAuth callback from Google.
    
    Query params:
        code: Authorization code from Google
        state: Session ID for tracking
        error: Error code if auth failed
    """
    try:
        error = request.args.get('error')
        if error:
            logger.warning("OAuth error: %s", error)
            return redirect(f"/settings?cgpu_error=oauth_failed&detail={error}")
        
        code = request.args.get('code')
        state = request.args.get('state')
        
        if not code or not state:
            return redirect("/settings?cgpu_error=missing_params")
        
        session_record = _oauth_sessions.get(state)
        if not session_record:
            return redirect("/settings?cgpu_error=invalid_session")
        
        # Exchange code for session via cgpu CLI
        logger.info("Exchanging OAuth code for cgpu session (%s)", state)
        
        result = subprocess.run(
            ["cgpu", "connect", "--code", code],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error("cgpu connect failed: %s", result.stderr)
            return redirect(f"/settings?cgpu_error=connect_failed&detail={result.stderr[:100]}")
        
        # Session created; now update cluster secret
        session_file = Path.home() / ".config" / "cgpu" / "state" / "session.json"
        config_file = Path.home() / ".config" / "cgpu" / "config.json"
        
        if not session_file.exists():
            return redirect("/settings?cgpu_error=no_session_file")
        
        # Create/update Kubernetes secret
        logger.info("Creating Kubernetes secret cgpu-credentials")
        _update_cgpu_secret(config_file, session_file)
        
        # Restart pods
        _restart_cgpu_pods()
        
        # Mark session as complete
        session_record['status'] = 'completed'
        session_record['completed_at'] = datetime.now().isoformat()
        
        return redirect("/settings?cgpu_success=true")
    
    except Exception as e:
        logger.exception("OAuth callback failed: %s", e)
        return redirect(f"/settings?cgpu_error=callback_error&detail={str(e)[:100]}")


@settings_bp.route('/cgpu/authenticate/status/<session_id>', methods=['GET'])
def cgpu_oauth_status(session_id):
    """
    Poll for OAuth completion status.
    
    Used by frontend while user is in OAuth flow.
    """
    session_record = _oauth_sessions.get(session_id)
    
    if not session_record:
        return jsonify({
            "status": "invalid_session"
        }), 404
    
    return jsonify({
        "session_id": session_id,
        "status": session_record['status'],
        "created_at": session_record['created_at'],
        "completed_at": session_record.get('completed_at')
    })


@settings_bp.route('/cgpu/status', methods=['GET'])
def get_cgpu_auth_status():
    """
    Get current CGPU authentication status.
    
    Returns:
        - authenticated: bool (has valid session)
        - expires_at: ISO datetime (when session expires)
        - user_email: str (authenticated user, if available)
        - message: str (human-readable status)
    """
    try:
        session_file = Path.home() / ".config" / "cgpu" / "state" / "session.json"
        
        if not session_file.exists():
            return jsonify({
                "authenticated": False,
                "message": "No cgpu session found. Run 'cgpu connect' locally or authenticate via Settings."
            })
        
        with open(session_file, 'r') as f:
            session = json.load(f)
        
        expires_at = session.get('tokenExpiry')
        user_email = session.get('userId')
        
        # Check if expired
        if expires_at:
            exp_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            is_expired = exp_time < datetime.now(tz=exp_time.tzinfo)
        else:
            is_expired = False
        
        return jsonify({
            "authenticated": not is_expired,
            "expires_at": expires_at,
            "user_email": user_email,
            "message": "CGPU authenticated and ready" if not is_expired else "Session expired; refresh authentication."
        })
    
    except Exception as e:
        logger.debug("Failed to check CGPU status: %s", e)
        return jsonify({
            "authenticated": False,
            "message": f"Error checking status: {str(e)}"
        }), 500


def _update_cgpu_secret(config_file, session_file):
    """
    Create or update Kubernetes secret with CGPU credentials.
    
    Uses kubectl to create/replace the secret in the cluster.
    """
    settings = get_settings()
    namespace = os.environ.get('CLUSTER_NAMESPACE', 'montage-ai')
    
    try:
        # Use kubectl to create secret
        cmd = [
            'kubectl', 'create', 'secret', 'generic', 'cgpu-credentials',
            f'--from-file=config.json={config_file}',
            f'--from-file=session.json={session_file}',
            '-n', namespace,
            '--dry-run=client', '-o', 'yaml'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise RuntimeError(f"kubectl failed: {result.stderr}")
        
        # Apply the secret
        apply_cmd = f"echo '{result.stdout}' | kubectl apply -f -"
        apply_result = subprocess.run(
            apply_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if apply_result.returncode != 0:
            raise RuntimeError(f"kubectl apply failed: {apply_result.stderr}")
        
        logger.info("Created/updated cgpu-credentials secret in namespace %s", namespace)
    
    except Exception as e:
        logger.error("Failed to update CGPU secret: %s", e)
        raise


def _restart_cgpu_pods():
    """
    Restart cgpu-server and worker pods to pick up new credentials.
    """
    settings = get_settings()
    namespace = os.environ.get('CLUSTER_NAMESPACE', 'montage-ai')
    
    try:
        # Restart cgpu-server
        subprocess.run(
            ['kubectl', 'rollout', 'restart', 'deploy/cgpu-server', '-n', namespace],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Restart workers
        subprocess.run(
            ['kubectl', 'rollout', 'restart', 'deploy/montage-ai-worker', '-n', namespace],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        logger.info("Triggered pod restart in namespace %s", namespace)
    
    except Exception as e:
        logger.error("Failed to restart pods: %s", e)
        # Don't raise; pods might restart anyway or user can restart manually
```

### Step 2: Register Blueprint in App

**File**: `src/montage_ai/web_ui/app.py`

```python
# In the routes registration section, add:
from .routes.settings import settings_bp

app.register_blueprint(settings_bp)
```

### Step 3: Frontend UI

**File**: `src/montage_ai/web_ui/templates/settings.html`

```html
{% extends "base.html" %}
{% import "components/macros.html" as m %}

{% block title %}Settings - Montage AI{% endblock %}

{% block extra_css %}
    <link rel="stylesheet" href="{{ url_for('static', filename='css/settings.css') }}">
    <style>
        .cgpu-status { padding: 1rem; border-radius: 0.5rem; }
        .cgpu-status.authenticated { background: #e8f5e9; border: 1px solid #4caf50; }
        .cgpu-status.unauthenticated { background: #fff3e0; border: 1px solid #ff9800; }
        .cgpu-button { margin-top: 0.5rem; }
    </style>
{% endblock %}

{% block content %}
    {% call m.section_header("SETTINGS", "System configuration and AI parameters.") %}{% endcall %}

    <div class="grid grid-cols-1 gap-4">
        {% call m.voxel_card("HARDWARE ACCELERATION", "Configure GPU usage for rendering and AI inference.") %}
            <div class="settings-item">
                <div class="settings-item-info">
                    <div class="text-sm font-bold">GPU Encoding</div>
                    <div class="text-xs text-muted">Intel QSV / NVIDIA NVENC / Apple Silicon</div>
                </div>
                {{ m.badge("ENABLED", "success") }}
            </div>
        {% endcall %}

        {% call m.voxel_card("CLOUD GPU (CGPU)", "Offload AI tasks (transcription, upscaling, LLM) to Google Colab.") %}
            <div id="cgpu-status-container" class="cgpu-status unauthenticated">
                <div class="text-sm">
                    <strong>Status:</strong> <span id="cgpu-status-text">Checking...</span>
                </div>
                <div class="text-xs text-muted" id="cgpu-status-details"></div>
            </div>
            <button id="cgpu-auth-button" class="cgpu-button btn btn-primary" onclick="startCGPUOAuth()">
                🔐 Authenticate with Google
            </button>
            <button id="cgpu-refresh-button" class="cgpu-button btn btn-secondary" onclick="refreshCGPUStatus()" style="display: none;">
                🔄 Refresh Status
            </button>
        {% endcall %}

        {% call m.voxel_card("AI MODELS", "Active LLM and Vision models for analysis and creative direction.") %}
            <div class="settings-item">
                <div class="settings-item-info">
                    <div class="text-sm font-bold">Creative Director</div>
                    <div class="text-xs text-muted">Gemini 2.0 Flash (Cloud)</div>
                </div>
                {{ m.badge("CONNECTED", "secondary") }}
            </div>
        {% endcall %}
    </div>

    <script>
        let currentOAuthSessionId = null;

        async function checkCGPUStatus() {
            try {
                const response = await fetch('/api/settings/cgpu/status');
                const data = await response.json();
                
                const container = document.getElementById('cgpu-status-container');
                const statusText = document.getElementById('cgpu-status-text');
                const statusDetails = document.getElementById('cgpu-status-details');
                const authButton = document.getElementById('cgpu-auth-button');
                const refreshButton = document.getElementById('cgpu-refresh-button');

                if (data.authenticated) {
                    container.classList.remove('unauthenticated');
                    container.classList.add('authenticated');
                    statusText.textContent = '✅ Authenticated';
                    statusDetails.innerHTML = `<div>User: ${data.user_email}</div><div>Expires: ${new Date(data.expires_at).toLocaleString()}</div>`;
                    authButton.style.display = 'none';
                    refreshButton.style.display = 'inline-block';
                } else {
                    container.classList.add('unauthenticated');
                    container.classList.remove('authenticated');
                    statusText.textContent = '❌ Not Authenticated';
                    statusDetails.textContent = data.message;
                    authButton.style.display = 'inline-block';
                    refreshButton.style.display = 'none';
                }
            } catch (error) {
                console.error('Failed to check CGPU status:', error);
                document.getElementById('cgpu-status-text').textContent = 'Error checking status';
            }
        }

        async function startCGPUOAuth() {
            try {
                document.getElementById('cgpu-auth-button').disabled = true;
                
                const response = await fetch('/api/settings/cgpu/authenticate/start', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.status !== 'ok') {
                    alert(`Error: ${data.message}`);
                    document.getElementById('cgpu-auth-button').disabled = false;
                    return;
                }
                
                currentOAuthSessionId = data.session_id;
                
                // Open OAuth in new window
                const popup = window.open(data.oauth_url, 'cgpu_oauth', 'width=500,height=700');
                
                // Poll for completion
                pollOAuthCompletion(data.session_id, popup);
            } catch (error) {
                console.error('Failed to start OAuth:', error);
                alert('Failed to start authentication. Check console for details.');
                document.getElementById('cgpu-auth-button').disabled = false;
            }
        }

        function pollOAuthCompletion(sessionId, popup) {
            const pollInterval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/settings/cgpu/authenticate/status/${sessionId}`);
                    const data = await response.json();
                    
                    if (data.status === 'completed') {
                        clearInterval(pollInterval);
                        if (popup) popup.close();
                        alert('✅ CGPU authentication successful! Restarting pods...');
                        
                        // Wait a few seconds for pods to restart
                        setTimeout(() => {
                            checkCGPUStatus();
                        }, 3000);
                    }
                } catch (error) {
                    console.debug('Polling OAuth status:', error);
                }
            }, 2000);
            
            // Stop polling after 10 minutes
            setTimeout(() => clearInterval(pollInterval), 600000);
        }

        async function refreshCGPUStatus() {
            document.getElementById('cgpu-refresh-button').disabled = true;
            checkCGPUStatus();
            setTimeout(() => {
                document.getElementById('cgpu-refresh-button').disabled = false;
            }, 1000);
        }

        // Check status on page load
        window.addEventListener('load', checkCGPUStatus);
    </script>
{% endblock %}
```

---

## Integration Points

### 1. Kubernetes Secret Management

The backend automatically:- Creates secret via `kubectl create secret`
- Updates existing secret if already present
- Mounts secret into pod via volume (existing yaml)

### 2. Pod Restart

```bash
kubectl rollout restart deploy/cgpu-server -n montage-ai
kubectl rollout restart deploy/montage-ai-worker -n montage-ai
```

Pods will:

- Read new credentials from secret
- Run cgpu connect (if session invalid)
- Start cgpu-serve on 8080

### 3. Monitoring

Check pod startup logs:

```bash
kubectl -n montage-ai logs -f deploy/cgpu-server
```

Look for:

```text
✅ CGPU serve started on 0.0.0.0:8080
✅ OAuth session valid (expires: 2026-02-17)
```

---

## Security Considerations

### ✅ Good Practices

1. **Secret encryption**: Kubernetes secrets are encrypted at rest (via etcd)
2. **RBAC**: Only cluster admins/service accounts can view secret
3. **HTTPS**: OAuth callback must use HTTPS in production
4. **Temporary states**: OAuth session ID expires after 10 minutes
5. **Input validation**: Code/state parameters validated before use

### ⚠️ Limitations

1. **Local cgpu required**: User must have `cgpu` CLI installed locally first
2. **No refresh button yet**: Users must re-authenticate after 7 days (TODO)
3. **No error recovery**: If pod restart fails, manual intervention needed

---

## Testing Checklist

- [ ] Frontend button appears in Settings
- [ ] OAuth URL opens in popup
- [ ] Callback URL captured correctly
- [ ] `cgpu connect` runs successfully in backend
- [ ] Secret created in `montage-ai` namespace
- [ ] Pods restart automatically
- [ ] Status updates to "Authenticated" after completion
- [ ] CGPU jobs (transcribe/upscale) work after auth
- [ ] Session file stored correctly: `~/.config/cgpu/state/session.json`
- [ ] Error handling: invalid code, missing config, kubectl not available

---

## Next Steps

1. Implement backend routes (Step 1)
2. Register blueprint in app (Step 2)
3. Update settings template (Step 3)
4. Test locally with `cgpu serve` running
5. Test in cluster with `/api/cgpu/transcribe` after auth
6. Add refresh/re-auth logic (7-day session expiry)
7. Monitor error logs and improve UX messaging

---

## References

- [CGPU Authentication Guide](CGPU_AUTHENTICATION.md)
- [Flask Blueprints](https://flask.palletsprojects.com/en/stable/blueprints/)
- [Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [OAuth 2.0 Flow](https://tools.ietf.org/html/rfc6749#section-1.3.1)
