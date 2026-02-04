#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <input-file> [output-file]" >&2
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-}"
IMAGE_FULL="$(${ROOT_DIR}/scripts/resolve-image.sh)"

if [ -n "$OUTPUT_FILE" ]; then
  sed "s|<IMAGE_FULL>|${IMAGE_FULL}|g" "$INPUT_FILE" > "$OUTPUT_FILE"
else
  sed "s|<IMAGE_FULL>|${IMAGE_FULL}|g" "$INPUT_FILE"
fi
