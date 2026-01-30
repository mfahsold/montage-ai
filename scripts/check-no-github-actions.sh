#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
if compgen -G "${ROOT_DIR}/.github/workflows/*" >/dev/null; then
  echo "ERROR: GitHub Actions workflow files detected in .github/workflows/. This repository forbids using GitHub Actions."
  echo "Please remove or disable these files."
  exit 1
else
  echo "OK: No GitHub Actions workflows detected."
fi
