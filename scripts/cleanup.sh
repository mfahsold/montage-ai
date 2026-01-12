#!/usr/bin/env bash
set -euo pipefail

# Cleanup script for montage-ai
# - Archives proxies >SIZE_THRESHOLD and older than AGE_DAYS
# - Compresses monitoring JSONs older than MON_AGE_DAYS
# - Rotates render.log into data/output/archive

ARCHIVE_DIR="data/output/archive"
MON_ARCHIVE_DIR="data/output/monitoring_archive"
SIZE_THRESHOLD="200M"    # files larger than this
AGE_DAYS=1                # files older than this (days)
MON_AGE_DAYS=90           # monitoring JSONs older than this (days)

mkdir -p "$ARCHIVE_DIR"
mkdir -p "$MON_ARCHIVE_DIR"

ts=$(date -u +%Y%m%d_%H%M%SZ)

# Rotate render.log
if [ -f data/output/render.log ]; then
  cp data/output/render.log "$ARCHIVE_DIR/render.log.$ts"
  gzip -f "$ARCHIVE_DIR/render.log.$ts"
  : > data/output/render.log
  echo "Rotated render.log -> $ARCHIVE_DIR/render.log.$ts.gz"
else
  echo "No render.log to rotate"
fi

# Archive proxies
proxies=$(find /tmp -maxdepth 1 -type f -name '*proxy*.mp4' -size +$SIZE_THRESHOLD -mtime +$AGE_DAYS -print || true)
if [ -z "$proxies" ]; then
  echo "No proxies matched criteria (size>$SIZE_THRESHOLD and older than $AGE_DAYS days)."
else
  tar -czf "$ARCHIVE_DIR/proxies.$ts.tar.gz" $proxies
  echo "Archived proxies -> $ARCHIVE_DIR/proxies.$ts.tar.gz"
  rm -f $proxies
  echo "Removed archived proxies from /tmp"
fi

# Compress old monitoring JSONs
oldmon=$(find data/output -maxdepth 1 -type f -name 'monitoring_*.json' -mtime +$MON_AGE_DAYS -print || true)
if [ -z "$oldmon" ]; then
  echo "No monitoring JSONs older than $MON_AGE_DAYS days."
else
  for f in $oldmon; do
    gzip -c "$f" > "$MON_ARCHIVE_DIR/$(basename "$f").gz"
    rm -f "$f"
    echo "Compressed $f -> $MON_ARCHIVE_DIR/$(basename "$f").gz"
  done
fi

echo "Cleanup complete."