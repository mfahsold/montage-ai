#!/bin/bash
# Template Consistency Check & Migration Tool

set -e

TEMPLATES_DIR="src/montage_ai/web_ui/templates"
ISSUES_FOUND=0

echo "🔍 Montage AI Web UI Template Consistency Check"
echo "================================================"
echo ""

# Check 1: Templates extending base.html
echo "Check 1️⃣  - Template Inheritance"
echo "Expected: All templates extend base.html (except base.html itself)"
echo ""

NOT_EXTENDING=$(find "$TEMPLATES_DIR" -name "*.html" -not -name "base.html" -not -path "*/components/*" \
    -exec grep -L "{% extends" {} \;)

if [ -z "$NOT_EXTENDING" ]; then
    echo "✅ All user-facing templates extend base.html"
else
    echo "❌ Templates NOT extending base.html:"
    echo "$NOT_EXTENDING" | while read file; do
        echo "   - $file"
    done
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# Check 2: Duplicate navbar definitions
echo "Check 2️⃣  - Duplicate Navigation Code"
echo "Expected: No <nav> tags in child templates (only in base.html)"
echo ""

DUPLICATE_NAV=$(find "$TEMPLATES_DIR" -name "*.html" -not -name "base.html" -not -path "*/components/*" \
    -exec grep -l "<nav" {} \;)

if [ -z "$DUPLICATE_NAV" ]; then
    echo "✅ No duplicate <nav> definitions found"
else
    echo "❌ Found duplicate <nav> in:"
    echo "$DUPLICATE_NAV" | while read file; do
        echo "   - $file"
    done
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# Check 3: Inline style tags
echo "Check 3️⃣  - Inline <style> Tags"
echo "Expected: Minimal inline styles (prefer CSS files)"
echo ""

INLINE_STYLES=$(find "$TEMPLATES_DIR" -name "*.html" \
    -exec grep -l "<style" {} \; | grep -v "base.html")

if [ -z "$INLINE_STYLES" ]; then
    echo "✅ No problematic inline <style> tags found"
else
    echo "⚠️  Found <style> tags in:"
    echo "$INLINE_STYLES" | while read file; do
        COUNT=$(grep -c "<style" "$file" || true)
        echo "   - $file ($COUNT inline style blocks)"
    done
    echo "   💡 Tip: Move styles to voxel-dark.css or component-specific CSS files"
fi
echo ""

# Check 4: Duplicate DOCTYPE/html/head tags
echo "Check 4️⃣  - Full HTML Structure"
echo "Expected: Only base.html should have <!DOCTYPE>, <html>, <head>"
echo ""

FULL_STRUCTURE=$(find "$TEMPLATES_DIR" -name "*.html" -not -name "base.html" -not -path "*/components/*" \
    -exec grep -l "<!DOCTYPE\|<html\|<body>" {} \;)

if [ -z "$FULL_STRUCTURE" ]; then
    echo "✅ No duplicate HTML structure found"
else
    echo "❌ Child templates have full HTML structure:"
    echo "$FULL_STRUCTURE" | while read file; do
        echo "   - $file"
    done
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# Check 5: Macro usage
echo "Check 5️⃣  - Component Macro Usage"
echo "Expected: Reusable patterns use macros from components/macros.html"
echo ""

TEMPLATES_WITHOUT_MACROS=$(find "$TEMPLATES_DIR" -name "*.html" -not -path "*/components/*" \
    -exec grep -L "{% import.*macros" {} \;)

echo "⚠️  Templates not using macros:"
echo "$TEMPLATES_WITHOUT_MACROS" | while read file; do
    if [ "$file" != "$TEMPLATES_DIR/base.html" ]; then
        echo "   - $(basename $file)"
    fi
done
echo "   💡 Tip: Use macros for voxel_card, workflow_card, job_card, section_header, etc."
echo ""

# Summary
echo "================================================"
echo "Summary:"
if [ $ISSUES_FOUND -eq 0 ]; then
    echo "✅ All checks passed! Templates are consistent."
    exit 0
else
    echo "⚠️  Found $ISSUES_FOUND issue(s) to fix"
    echo ""
    echo "See docs/WEB_UI_ARCHITECTURE.md for migration guide"
    exit 1
fi
