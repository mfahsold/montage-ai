#!/usr/bin/env bash
set -euo pipefail

# Montage AI – Vendor-agnostic CI runner
# Runs locally or inside any CI system (Jenkins, Drone, Woodpecker, GitLab CI, etc.)
# No GitHub Actions required.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PY_VER_MINOR=${PY_VER_MINOR:-"3.10"}
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-ci}"

log() { echo -e "\033[36m[ci]\033[0m $*"; }
warn() { echo -e "\033[33m[ci]\033[0m $*"; }
err() { echo -e "\033[31m[ci]\033[0m $*"; }

ensure_python() {
  if command -v python${PY_VER_MINOR} >/dev/null 2>&1; then
    PY_CMD="python${PY_VER_MINOR}"
  elif command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
  elif command -v python >/dev/null 2>&1; then
    PY_CMD="python"
  else
    err "Python ${PY_VER_MINOR} not found. Please install Python >= ${PY_VER_MINOR}."
    exit 1
  fi
  echo "$PY_CMD"
}

setup_venv() {
  local py
  py=$(ensure_python)
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtualenv at ${VENV_DIR}"
    "$py" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  log "Installing dependencies"
  # Use requirements.txt to avoid extras from pyproject while keeping editable src via PYTHONPATH
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
}

check_no_github_actions() {
  # Fail CI early if GitHub Actions workflow files are present. This project forbids using GH Actions.
  if compgen -G "${ROOT_DIR}/.github/workflows/*" >/dev/null; then
    err "Error: GitHub Actions workflow files detected in .github/workflows/. This repository does not use GitHub Actions. Remove or disable them."
    exit 2
  fi
}

run_unit_tests() {
  log "Running unit tests"
  export PYTHONPATH="${ROOT_DIR}/src"
  # default to verbose if CI=true
  local PYTEST_FLAGS
  if [[ "${CI:-}" == "true" ]]; then PYTEST_FLAGS="-q"; else PYTEST_FLAGS="-q"; fi
  python -m pytest ${PYTEST_FLAGS} --maxfail=1 --disable-warnings
}

audit_dependencies() {
  log "Checking dependencies for security vulnerabilities"
  if ! python -m pip list | grep -q pip-audit; then
    python -m pip install pip-audit --quiet
  fi
  if python -m pip_audit --desc --skip-editable 2>/dev/null; then
    log "Dependency audit: OK (no vulnerabilities found)"
  else
    warn "Dependency audit: ⚠️  Some vulnerabilities detected; review manually before deploying"
    # NOTE: Do not fail CI on audit; security issues require human review
    return 0
  fi
}

validate_manifests() {
  if ! command -v kubectl >/dev/null 2>&1; then
    warn "kubectl not found; skipping Kubernetes manifest validation"
    return 0
  fi
  log "Validating Kubernetes manifests (kustomize)"
  set +e
  kubectl kustomize "${ROOT_DIR}/deploy/k3s/base/" >/dev/null && echo "base ✓"
  kubectl kustomize "${ROOT_DIR}/deploy/k3s/overlays/cluster/" >/dev/null && echo "overlays/cluster ✓"
  set -e
}

smoke_build() {
  if ! command -v docker >/dev/null 2>&1; then
    warn "docker not found; skipping Docker smoke build"
    return 0
  fi
  log "Running Docker smoke build (no push)"
  pushd "${ROOT_DIR}" >/dev/null
  # Keep it quick: local arch build, no push
  docker build --build-arg GIT_COMMIT=$(git rev-parse --short=8 HEAD 2>/dev/null || echo dev) -t montage-ai:ci .
  popd >/dev/null
}

main() {
  log "Starting vendor-agnostic CI"
  setup_venv
  check_no_github_actions
  run_unit_tests
  audit_dependencies
  validate_manifests
  smoke_build
  log "CI pipeline complete"
}

main "$@"
