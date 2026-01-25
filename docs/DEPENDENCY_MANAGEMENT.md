# Dependency Management (recommendation: uv)

This repository adopts `uv` (Astral) as the recommended dependency and project manager for development and CI.

Why `uv`?
- Fast dependency resolution and install (10-100× faster than pip in many benchmarks).
- Supports `pyproject.toml` projects and produces a reproducible `uv.lock` lockfile.
- Integrates well with Docker and GitHub Actions via `astral-sh/setup-uv`.

Quick start

1. Install uv (locally):

```bash
# Preferred: standalone installer
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or via pipx
pipx install uv
```

2. Install Python versions and sync dependencies (dev):

```bash
uv python install
uv sync --locked --all-extras --dev  # if uv.lock exists
uv sync --all-extras --dev            # fallback when uv.lock not present
```

3. Generate uv.lock (developer, run locally):

```bash
# Run this locally; CI uses `uv sync` as a best-effort if uv.lock is missing
make deps-lock  # runs ./scripts/uv-lock.sh
# After verifying, commit `uv.lock` to the repository
```

CI integration

- We prefer *local CI* to avoid GitHub Actions costs: run `make ci-local` or `./scripts/ci-local.sh` on a developer machine or self-hosted runner. The local CI script installs `uv` (if missing), syncs dependencies and runs `uv run pytest`.
- A lightweight GitHub Actions workflow (`.github/workflows/uv-ci.yml`) exists but **automatic triggers are disabled** and it is set to `workflow_dispatch` only. This avoids incurring recurring GitHub Actions costs.
- The workflow currently uses `uv sync --dev` to avoid installing private extras during CI while `uv.lock` is being adopted.

Agent instructions

- Agents (automation) MUST run the full local CI via `make ci-local` and attach the output to PRs rather than creating/updating GitHub Actions that run on push/pr and produce cost. If a workflow change is proposed that would add recurring CI execution on push/PR, explicitly call out the cost impact and obtain maintainer approval.
- Agents should include the local CI output (e.g., `uv run pytest -q` result) in the PR description or comments.

Notes & troubleshooting

- Some optional extras (e.g., `cgpu`) are not available on public PyPI or require private registries.

Private optional extras (recommended):
- We moved sensitive/private extras (e.g., `cgpu`) into a dedicated optional group `cloud-private` (not included in `montage-ai[all]`). This avoids breaking `uv sync --all-extras` in CI and developer environments.
- To install private extras locally, either:
  - Configure a private index and run `uv lock --index https://private.example/simple`
  - Install the private extra manually: `uv pip install --index https://private.example/simple cgpu` or `pip install --extra-index-url https://private.example/simple cgpu`.

## Local CI behavior

- The local CI script (`./scripts/ci-local.sh`) skips `uv sync` by default when private extras are declared (e.g., `cloud-private` in `pyproject.toml`) to avoid failing on developer machines that don't have access to private indices.
- To opt-in and attempt resolving private extras during local runs, set `INCLUDE_PRIVATE_EXTRAS=1` in your environment (for example: `export INCLUDE_PRIVATE_EXTRAS=1`). Use this only when you have configured access to the private index.

If `uv lock` fails locally, either:

- Generate lock locally after installing optional/private dependencies and commit `uv.lock`.
- Run `uv lock --index <private-index>` if using private registries.
- Edit pyproject.toml to temporarily remove problematic extras and retry `uv lock`.

Deployment config centralization

- Build and deploy scripts read `deploy/k3s/config-global.yaml` (and the generated `deploy/k3s/base/cluster-config.env`) and respect `REGISTRY_URL`, `IMAGE_NAME`, `IMAGE_TAG`, and `CLUSTER_NAMESPACE`. Override via environment variables or by editing `deploy/k3s/config-global.yaml`.
- The `Dockerfile` accepts a build arg `SERVICE_PORT` (default `5000`) and sets `ENV SERVICE_PORT` so the port is configurable at build time. To build with a custom port:

```bash
docker build --build-arg SERVICE_PORT=8080 -t ${REGISTRY}/montage-ai:${TAG} .
```

- Update `deploy/k3s/config-global.yaml` to set `registry.url`, `images.montage_ai.tag`, or `cluster.namespace` as needed for your environment.

- Best practice: pin `UV_VERSION` in `deploy/config.env` (legacy fallback) and in CI workflows to ensure reproducible tooling.

Next steps

- Commit `uv.lock` once optional/private packages (like `cgpu`) are available in the resolved indexes, or when maintainers can generate it locally.
- Consider moving CI to `uv sync --locked` after `uv.lock` is in repo.
