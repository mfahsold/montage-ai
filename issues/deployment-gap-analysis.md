# Installation & Deployment Gap Analysis

**Date:** 2026-02-10
**Type:** Gap Analysis / Bug Report
**Scope:** Local K3d Deployment & Documentation Quality
**Environment:** Linux (ARM64), K3d v5.8.3

## Summary
The technical features of the application (`montage-ai` CLI and Docker container) are functional and verified on ARM64. However, the documentation and scripts for "Local K3s/K3d Setup" have significant gaps that prevent a successful fresh installation without undocumented manual interventions.

## 1. Documentation Gaps (`docs/k3s-local-setup.md`)

### Critical: Missing Dependency Instructions (KEDA)
The local setup guide fails to mention that **KEDA (Kubernetes Event-driven Autoscaling)** is a required dependency.
*   **Result:** Deployment fails with `no matches for kind "ScaledObject"`.
*   **Fix:** Add `make -C deploy/k3s install-keda` to the guide. The `Makefile` has the target, but the docs don't mention it.

### Critical: Config Placeholders & Registry Logic
The guide instructs to use `sed` to replace cluster namespace/domain/IPs but ignores **Registry placeholders** (`<REGISTRY_HOST>`, `<REGISTRY_URL>`, etc.) present in `config-global.yaml.example`.
*   **Result:** `make pre-flight` correctly fails due to unreplaced placeholders.
*   **Fix:** Update `docs/k3s-local-setup.md` to explain how to configure the file for a local no-registry setup (e.g., setting them to empty quotes or specific local values).

### CI/Makefile Discrepancies
`docs/llm-agents.md` references `make ci-local`, which does not exist.
*   **Fix:** References should point to `./scripts/ci-local.sh`.

## 2. Technical Issues (Scripts)

### `scripts/ops/render_cluster_config_env.sh` Logic Bug
When configuring for a local environment without a registry (implicitly desired for K3d image import workflow), the script generates an invalid image path with a leading slash.
*   **Condition:** `REGISTRY_URL` is empty.
*   **Current Logic:** `: "${IMAGE_FULL:=${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}}"` -> resolves to `/montage-ai:latest`.
*   **Result:** Pods fail with `InvalidImageName`.
*   **Fix:** Change logic to conditionally append slash: `: "${IMAGE_FULL:=${REGISTRY_URL:+$REGISTRY_URL/}${IMAGE_NAME}:${IMAGE_TAG}}"`.

### Idempotency
*   **K3d Create:** `k3d cluster create` fails if the cluster or port exists. Idempotency scripts/checks are missing in docs.
*   **Docs:** `config-global.yaml.example` warning is good ("You MUST replace ALL..."), but the "Quick Start" provided `sed` commands are insufficient to satisfy it.

## 3. Successful Verification (with Manual Fixes)
After applying the following manual fixes, deployment was **successful**:
1.  Installed KEDA (`make -C deploy/k3s install-keda`).
2.  Patched `scripts/ops/render_cluster_config_env.sh` to remove invalid leading slash.
3.  Manually configured `config-global.yaml` to handle empty registry values.
4.  Created cluster on non-conflicting port.

**Status:**
*   **Docker:** ✅ Functional (Preview mode tested).
*   **CI:** ✅ Passed (600+ tests).
*   **Cluster:** ⚠️ Broken out-of-the-box, ✅ Functional with fixes.

## Recommendations
1.  Update `docs/k3s-local-setup.md` to include KEDA installation.
2.  Fix logic in `render_cluster_config_env.sh`.
3.  Add a "Local Dev" profile to `init-config.sh` that pre-populates `config-global.yaml` with values suitable for K3d/Kind (no registry) to avoid manual `sed` errors.
