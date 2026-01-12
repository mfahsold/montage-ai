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

- We add a lightweight GitHub Actions workflow (`.github/workflows/uv-ci.yml`) that installs `uv` and runs `uv sync` + `uv run pytest`.
- The workflow currently uses unlocked `uv sync --all-extras --dev` for compatibility while `uv.lock` is being adopted.

Notes & troubleshooting

- Some optional extras (e.g., `cgpu`) are not available on public PyPI or require private registries. If `uv lock` fails locally, either:
  - Install required private dependencies or set `--index` to a private index, or
  - Temporarily remove the problematic extras from `pyproject.toml` while generating a lockfile.

- Best practice: pin `UV_VERSION` in `deploy/config.env` and in CI workflows to ensure reproducible tooling.

Next steps

- Commit `uv.lock` once optional/private packages (like `cgpu`) are available in the resolved indexes, or when maintainers can generate it locally.
- Consider moving CI to `uv sync --locked` after `uv.lock` is in repo.
