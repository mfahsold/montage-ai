#!/usr/bin/env bash
set -euo pipefail

# Grep for common hardcoded registry patterns in PUBLIC files.
# Excludes: private docs, git, venvs, this script itself, generated configs,
# and example/template files (which legitimately contain placeholder values).

echo "Scanning for hardcoded registry references (excludes private docs, examples, generated configs)..."

PATTERNS=("192\.168\.1\.12" "YOUR_REGISTRY")

grep_args=(
  --line-number --recursive
  --exclude-dir=".git" --exclude-dir="private" --exclude-dir="venv"
  --exclude-dir=".venv" --exclude-dir="__pycache__"
  --exclude="check-hardcoded-registries.sh"
  --exclude="*.example"
  --exclude="config-global.yaml"
  --exclude="cluster-config.env"
)

FOUND=0
for p in "${PATTERNS[@]}"; do
  if grep "${grep_args[@]}" -E "${p}" . 2>/dev/null; then
    FOUND=1
  fi
done

if [ "$FOUND" -eq 1 ]; then
  echo "WARNING: Found potential hardcoded registry strings in public files." >&2
  echo "Registry config should be centralized in deploy/k3s/config-global.yaml" >&2
  exit 1
else
  echo "No obvious hardcoded registry strings found in public files."
fi
