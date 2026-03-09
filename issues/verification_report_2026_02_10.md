# Verification Report: Installation and Deployment

**Date:** February 10, 2026  
**Author:** GitHub Copilot (CodeAI)

## Summary

Installation, local-cluster deployment, and test-job execution were successfully verified. A few documentation and tooling integrity improvements were identified.

## Executed Checks

1. **Installation and setup (`scripts/setup.sh`)**
   - Ran successfully.
   - Correctly detects ARM64 environment (MediaPipe disabled path).
   - Test assets (`data/input`, `data/music`) were generated successfully.

2. **Cluster deployment (`deploy/k3s`)**
   - `k3d` CLI was missing locally, but a cluster was already reachable via `kubectl` (`k3d-montage-dev`).
   - Idempotence test (`scripts/test-cluster-idempotency.sh`) passed. Repeated deploys did not change PVCs and did not trigger unexpected pod restarts.

3. **Features (LLM and CGPU)**
   - `verify-deps` (through `montage-ai.sh`) reported missing local tools (`gemini-cli`), while equivalent functionality was available in cluster mode (`cgpu-server` pod).
   - Job submission with `"cgpu": true` was accepted and processed.

4. **Test job**
   - Media upload (video and audio) via `/api/upload` succeeded.
   - Note: audio upload requires `type=music` in form-data; otherwise upload is rejected by video validation.
   - Job start via API succeeded.
   - Output was downloaded successfully to local `downloads/` (`montage_20260210_133050.mp4`, ~258 KB).

## Findings and Improvements

### 1. Documentation and accessibility
- **Host header:** For local cluster API access (`k3d`/Traefik), `Host: montage-ai.local` is required. Direct IP/port calls return 404. This is documented for browser setup (`/etc/hosts`) but should be clearer for API usage.
- **Job status naming:** API currently returns `finished`, while users may expect `completed` or `succeeded`. Consider standardization or explicit status-enum documentation.

### 2. Tooling
- **check-deps path:** Direct call to `scripts/check-deps.sh` failed (file not found), while `montage-ai.sh check-deps` worked. The command pathing should be documented more clearly.

### 3. API UX
- **Upload type messaging:** `Invalid video format` for `.mp3` uploads without `type=music` is technically correct but could be more helpful (for example: "Did you mean to upload music?").

## Conclusion

The system is functional, idempotent, and core features are usable. Documentation quality is generally good, with small gaps around API details and operator guidance.
