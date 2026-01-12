#!/usr/bin/env bash
set -euo pipefail

# Archive obsolete scripts into scripts/cleanup/obsolete
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE_DIR="$ROOT_DIR/scripts/cleanup/obsolete"
mkdir -p "$ARCHIVE_DIR"

echo "Scanning for obsolete patterns..."
PATTERNS=("build_with_cache.sh" "build-multiarch.sh" "build_with_cache" "legacy")

for p in "${PATTERNS[@]}"; do
  # Fallback to find if ripgrep (rg) is not available
  if command -v rg >/dev/null 2>&1; then
    files=$(rg --hidden --files -g 'scripts/**' -S --glob '!scripts/cleanup/**' -g "*${p}*" || true)
  else
    files=$(find scripts -maxdepth 2 -type f -iname "*${p}*" -not -path "scripts/cleanup/*" || true)
  fi
  for f in $files; do
    echo "Archiving $f -> $ARCHIVE_DIR"
    mv "$f" "$ARCHIVE_DIR/" || echo "Failed to move $f"
  done
done

echo "Archive complete. Review $ARCHIVE_DIR and commit if desired."