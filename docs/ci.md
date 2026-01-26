# CI without GitHub Actions

This repository supports a vendor-agnostic CI flow that you can run locally or wire into any CI system (Jenkins, Drone, Woodpecker, GitLab CI, Buildkite, etc.). No GitHub Actions are required.

## Quickstart (Local)

```bash
./scripts/ci.sh
```

This runs `scripts/ci.sh`, which:

- Creates a Python virtualenv and installs dependencies from `requirements.txt`.
- Runs unit tests with `pytest` using `PYTHONPATH=src` (no package install required).
- Validates Kubernetes manifests with `kubectl kustomize` if `kubectl` is available.
- Performs a fast Docker smoke build (no push) if `docker` is available.

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
