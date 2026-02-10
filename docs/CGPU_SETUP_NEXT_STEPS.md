# CGPU Setup: Next Steps

## Current Status

✅ **Completed:**
- cgpu CLI installed globally via npm
- `~/.config/cgpu` directory created
- Documentation updated (npm installation vs pip)

❌ **Missing:**
- Google Cloud OAuth credentials configuration
- cgpu OAuth authentication (local session)
- Kubernetes secret creation

---

## Step 1: Create Google Cloud Project & OAuth Credentials

Follow the detailed instructions in [docs/CGPU_AUTHENTICATION.md](CGPU_AUTHENTICATION.md) **Part 1 - Local Host Authentication**, specifically:

### Part 1: Local Host Authentication

**Steps 1-3: Create Google Cloud OAuth Config**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project: `montage-ai-cgpu`
3. Enable the **Colaboratory API**
4. Create OAuth credentials (Desktop app type)
5. Note down your **Client ID** and **Client Secret**

### Step 2: Create cgpu Config File

Once you have the credentials, create this file:

```bash
~/.config/cgpu/config.json
```

With the following content (replace YOUR_CLIENT_ID and YOUR_CLIENT_SECRET):

```json
{
  "clientId": "YOUR_CLIENT_ID",
  "clientSecret": "YOUR_CLIENT_SECRET",
  "colabApiDomain": "https://colab.research.google.com",
  "colabGapiDomain": "https://colab.googleapis.com"
}
```

### Step 3: Authenticate with cgpu

Run the interactive OAuth flow:

```bash
cgpu connect
```

**Important**: Keep the terminal open during authentication. The browser will open an OAuth consent screen.

### Step 4: Create Kubernetes Secret

Once authentication succeeds, run:

```bash
./scripts/ops/cgpu-refresh-session.sh
```

This will:
1. Verify your cgpu session (`~/.config/cgpu/state/session.json`)
2. Create/update the `cgpu-credentials` Kubernetes secret
3. Restart the worker + cgpu-server deployments to pick up credentials

---

## Verification

After completing all steps, verify CGPU is available in the cluster:

```bash
# Check that cgpu status works in a worker pod
kubectl -n montage-ai exec deploy/montage-ai-worker -- cgpu status

# Expected output: Connected to Colab / Google Gemini APIs (no errors)
```

---

## References

- **Full Setup Guide**: [docs/CGPU_AUTHENTICATION.md](CGPU_AUTHENTICATION.md)
- **Web UI Implementation** (optional): [docs/CGPU_WEBAPP_AUTH_IMPLEMENTATION.md](CGPU_WEBAPP_AUTH_IMPLEMENTATION.md)
- **Troubleshooting**: See [docs/CGPU_AUTHENTICATION.md](CGPU_AUTHENTICATION.md) Troubleshooting section

---

## Estimated Time

- Creating Google Cloud project + credentials: **10-15 minutes**
- Running cgpu connect: **2-3 minutes**
- Running cgpu-refresh-session.sh: **1-2 minutes**

**Total**: ~15-20 minutes
