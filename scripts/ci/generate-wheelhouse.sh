#!/usr/bin/env bash
# Helper: build a small internal wheelhouse for flaky binary wheels used in CI
# Usage: ./scripts/ci/generate-wheelhouse.sh /tmp/wheelhouse
set -euo pipefail
OUT_DIR="${1:-/tmp/wheelhouse}"
mkdir -p "$OUT_DIR"
pip download --dest "$OUT_DIR" --only-binary=:all: --requirement requirements.txt || true
echo "Wheelhouse populated at: $OUT_DIR"