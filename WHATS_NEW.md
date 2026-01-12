# What's New (2026-01-13)

This page highlights recent operational and CI improvements that make builds and canary deploys more resilient and easier for contributors without access to private indexes.

## Key updates

- Resilient push fallbacks
  - The `scripts/build-and-deploy.sh` script now attempts push fallbacks when the configured registry is unreachable:
    1. Primary: configured registry (from `deploy/config.env` / `REGISTRY`)
    2. Fallback: GHCR (when `GHCR_PAT` is provided)
    3. Fallback: Node-import via `scripts/load-image-to-cluster.sh` (when `NODE_IMPORT_NODES` is set)
  - Use `SKIP_DEPLOY=1` for local/test runs to skip Kubernetes steps.

- Local CI stability
  - `scripts/ci-local.sh` now skips `uv sync` by default when private extras (e.g., `cloud-private`) are declared to avoid failing on developer machines without private indexes. Set `INCLUDE_PRIVATE_EXTRAS=1` to opt-in.

- Power-user tools
  - `scripts/check-registry.py` and `scripts/registry_check.sh` provide quick diagnostics for registry reachability and TLS checks.
  - `scripts/load-image-to-cluster.sh` provides an example node-import flow (scp tar to node(s) + `ctr images import`) for environments where registry push is not available.

- Tests and docs
  - Added integration tests that simulate registry push failure and verify GHCR and node-import fallbacks.
  - Updated `docs/REGISTRY_TROUBLESHOOTING.md` and `docs/DEPENDENCY_MANAGEMENT.md` with the new workflow and guidance.

## How this helps contributors
- Contributors without private index access can still run local CI and tests reliably.
- Ops can choose to enable the internal registry or accept fallbacks (GHCR or node-import) depending on policy.

For the full list of changes see `CHANGELOG.md` under **Unreleased**.

> Note: To avoid incurring GitHub Actions runner costs for routine docs updates we publish GitHub Pages locally using the `gh` CLI. See `scripts/publish-docs.sh` for instructions and automation. This repository still includes an Actions-based Pages workflow for convenience, but we prefer the local `gh pages publish` flow when possible.
