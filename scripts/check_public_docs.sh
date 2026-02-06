#!/usr/bin/env bash
set -euo pipefail

fail=false

echo "[docs] Checking for internal docs in public areas..."

if find . -maxdepth 1 -type f \( \
  -name "*STATUS*" -o \
  -name "*STRATEGY*" -o \
  -name "*DEPLOYMENT*" -o \
  -name "*SESSION*" -o \
  -name "*AUDIT*" -o \
  -name "*VERIFICATION*" -o \
  -name "*REPORT*" -o \
  -name "*SUMMARY*" \
\) -print -quit 2>/dev/null | grep -q .; then
  echo "[docs] Error: internal docs detected in repository root."
  fail=true
fi

if find docs/ -type f \( \
  -name "*STATUS*" -o \
  -name "*STRATEGY*" -o \
  -name "*DEPLOYMENT*" -o \
  -name "*SESSION*" -o \
  -name "*AUDIT*" -o \
  -name "*VERIFICATION*" -o \
  -name "*REPORT*" -o \
  -name "*SUMMARY*" \
\) -print -quit 2>/dev/null | grep -q .; then
  echo "[docs] Error: internal docs detected in docs/."
  fail=true
fi

if [ "$fail" = true ]; then
  echo "[docs] Public docs check failed."
  exit 1
fi

echo "[docs] Public docs check passed."
