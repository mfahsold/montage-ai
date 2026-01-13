#!/usr/bin/env bash
set -euo pipefail

BRANCH=${1:-gh-pages}
COMMIT_MSG=${2:-"Publish docs: $(date -u +%Y-%m-%dT%H:%M:%SZ)"}
REPO_URL=$(git config --get remote.origin.url)

if [ -z "$REPO_URL" ]; then
  echo "Error: remote origin URL not found" >&2
  exit 1
fi

if [ ! -d "docs" ]; then
  echo "Error: docs/ directory not found" >&2
  exit 1
fi

REPO_ROOT=$(git -C "$(pwd)" rev-parse --show-toplevel || true)
if [ -z "$REPO_ROOT" ]; then
  REPO_ROOT=$(pwd)
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Cloning repo into $TMP_DIR..."
git clone --quiet "$REPO_URL" "$TMP_DIR"
cd "$TMP_DIR"

# Try to checkout the branch if it exists, otherwise create an orphan branch
if git ls-remote --heads origin "$BRANCH" | grep -q "$BRANCH"; then
  git checkout --quiet "$BRANCH"
else
  git checkout --orphan "$BRANCH"
fi

# Remove existing files (but keep .git)
find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +

# Copy docs content from source repo
cp -r "$REPO_ROOT/docs/." . || true

# Ensure there's something to commit
if [ -z "$(git status --porcelain)" ]; then
  echo "No changes to publish to $BRANCH"
  exit 0
fi

git add -A
git commit -m "$COMMIT_MSG"

echo "Pushing to origin/$BRANCH..."
# Force push to gh-pages to replace old content
git push origin "HEAD:$BRANCH" --force

echo "Published docs to branch: $BRANCH"