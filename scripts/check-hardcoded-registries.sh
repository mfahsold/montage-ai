#!/usr/bin/env bash
set -euo pipefail

# Grep for common hardcoded registry patterns in PUBLIC files
EXCLUDE_DIRS=(private .git .venv venv .venv)
PATTERNS=("192.168.1.12" ":5000" ":30500" "YOUR_REGISTRY" "ghcr.io/mfahsold/montage-ai")

echo "Scanning for hardcoded registry references (this scan excludes private docs)..."

grep_args=(--line-number --recursive --exclude-dir=".git" --exclude-dir="private" --exclude-dir="venv" --exclude-dir=".venv" --exclude-dir="__pycache__")

FOUND=0
for p in "${PATTERNS[@]}"; do
  if grep "${grep_args[@]}" -E "${p}" .; then
    FOUND=1
  fi
done

if [ "$FOUND" -eq 1 ]; then
  echo "WARNING: Found potential hardcoded registry strings in public files. Consider centralizing to deploy/k3s/config-global.yaml" >&2
  exit 1
else
  echo "No obvious hardcoded registry strings found in public files."
fi
