# CI without GitHub Actions

This repository enforces a strict _no GitHub Actions_ policy. CI is vendor-agnostic and runs locally or in any CI system you choose (Jenkins, Drone, Woodpecker, GitLab CI, Buildkite, etc.). The canonical CI runner is `./scripts/ci.sh` and it will fail if any files exist in `.github/workflows/`.

## Quickstart (Local)

```bash
./scripts/ci.sh
```

This runs `scripts/ci.sh`, which:

- Creates a Python virtualenv and installs dependencies from `requirements.txt`.
- Runs unit tests with `pytest` using `PYTHONPATH=src` (no package install required).
- Validates Kubernetes manifests with `kubectl kustomize` if `kubectl` is available.
- Performs a fast Docker smoke build (no push) if `docker` is available.

### Seeding test media for smoke tests

Some dev smoke tests expect at least one small video in the cluster `montage-input` PVC. Use the included helper to seed a tiny preview clip:

```bash
# generates a 1s MP4 locally (requires ffmpeg) and copies it into the PVC via a running pod
./scripts/ops/seed-test-media.sh montage-ai test_data/preview.mp4
```

This makes autoscale and end-to-end preview workflows deterministic on small test clusters.
## Jenkins

A declarative `Jenkinsfile` is included. Minimal setup:

- Create a Jenkins pipeline job pointed at this repo.
- Ensure the agent has Python 3.10+, Docker (optional), and kubectl (optional).
- The pipeline runs `./scripts/ci.sh`.

## Other CI systems

For Drone / Woodpecker / GitLab CI, configure a job to run:

```bash
chmod +x scripts/ci.sh
./scripts/ci.sh
```

You can set environment variables to tweak behavior:

- `PY_VER_MINOR` (default: `3.10`) – Python minor version selector for local discovery.
- `VENV_DIR` – custom virtualenv path (default: `.venv-ci`).
- `CI=true` – quiet pytest output.

## Notes

- The CI flow avoids installing the package (no `pip install .`) to prevent optional dependencies from affecting test runs. It uses `PYTHONPATH=src`.
- Kubernetes validation is skipped if `kubectl` is absent.
- Docker smoke build is skipped if `docker` is absent.
