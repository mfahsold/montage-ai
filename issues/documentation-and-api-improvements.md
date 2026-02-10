# Improvement Suggestions: Documentation & API Experience

**Date:** February 10, 2026
**Source:** Verification Run (CodeAI)

## Context
During the installation and deployment verification process, several minor friction points were identified that could be improved to enhance developer experience and API usability.

## Suggestions

### 1. Documentation: API Access & Host Headers
**Problem:** accessing the API in a local cluster (`k3d`/Traefik) leads to `404 Not Found` errors if accessed via `localhost:8080/8081` without the correct Host header (`montage-ai.local`), due to Ingress routing rules.
**Recommendation:**
*   Add a prominent "API Note" in the `docs/getting-started.md` and `docs/cluster-deploy.md`.
*   Explicitly mention that `curl` commands need `-H "Host: montage-ai.local"`.
*   Consider adding a `NodePort` service or a default catch-all Ingress for local dev to allow simpler `localhost` access without host file modification.

### 2. API: Job Status Consistency
**Problem:** The API returns status `finished` for completed jobs, whereas many users/developers might expect standard terms like `completed` or `succeeded`.
**Recommendation:**
*   Standardize status enums (e.g., `queued`, `processing`, `completed`, `failed`).
*   Or, if `finished` is kept, strictly document all possible status values in `docs/api-reference.md` (or equivalent).

### 3. API: Upload Error Messages
**Problem:** Uploading an `.mp3` file to `/api/upload` without specifying `type=music` results in a generic `Invalid video format` error.
**Recommendation:**
*   Improve error handling in `app.py`. If a known audio extension is detected but `type` is default/video, suggest adding `type=music` in the error message.
*   Example: `"Invalid video format. Did you mean to upload music? Set type='music'."`

### 4. Tooling: Script Paths
**Problem:** The `montage-ai.sh` wrapper works fine, but direct script usage `scripts/check-deps.sh` failed in the test environment (possibly due to path assumptions or execution context).
**Recommendation:**
*   Ensure all scripts in `scripts/` are standalone executable or strictly document that they must be run via `montage-ai.sh`.

---
*Created by CodeAI during deployment verification.*
