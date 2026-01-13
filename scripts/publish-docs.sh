#!/usr/bin/env bash
set -euo pipefail

# Publish docs to GitHub Pages using the gh CLI (local, no GitHub Actions cost)
# Usage: ./scripts/publish-docs.sh [--branch BRANCH] [--message MSG]

BRANCH=${1:-gh-pages}
MSG=${2:-"Publish docs: $(date -u +%Y-%m-%dT%H:%M:%SZ)"}

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: 'gh' CLI not found. Install GitHub CLI: https://cli.github.com/" >&2
  exit 1
fi

# Ensure authenticated
if ! gh auth status >/dev/null 2>&1; then
  echo "You are not logged in to GitHub CLI. Run 'gh auth login' to authenticate." >&2
  exit 1
fi

# Confirm docs directory exists
if [ ! -d "docs" ]; then
  echo "Error: docs/ directory not found. Ensure site is built into docs/" >&2
  exit 1
fi

# Publish using gh pages publish (this will push to the gh-pages branch)
echo "Publishing docs/ to branch: $BRANCH"
gh pages publish docs --branch "$BRANCH" --message "$MSG" || { echo "Failed to publish docs" >&2; exit 1; }

echo "Docs published to GitHub Pages (branch: $BRANCH)."