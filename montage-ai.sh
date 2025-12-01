#!/bin/bash
# Montage AI - Simple CLI
set -e

show_help() {
    cat << EOF
Montage AI - Video Montage Creator

Usage: ./montage-ai.sh [COMMAND] [OPTIONS]

Commands:
  run [STYLE]     Create montage (default: dynamic)
  preview         Quick preview (fast preset)
  hq              High quality render
  list            List available styles
  build           Build Docker image

Styles:
  dynamic         Position-aware pacing (default)
  hitchcock       Suspense - slow build, fast climax
  mtv             Fast 1-2 beat cuts
  action          Michael Bay rapid cuts
  documentary     Natural, observational
  minimalist      Long contemplative takes

Options:
  --stabilize     Enable video stabilization
  --no-enhance    Disable color enhancement
  --variants N    Generate N variants

Examples:
  ./montage-ai.sh run                    # Default dynamic style
  ./montage-ai.sh run hitchcock          # Hitchcock suspense
  ./montage-ai.sh preview mtv            # Quick MTV preview
  ./montage-ai.sh hq documentary         # High quality documentary
  ./montage-ai.sh run --stabilize        # With stabilization
EOF
}

list_styles() {
    echo "Available Styles:"
    echo "  dynamic      - Position-aware pacing (intro→build→climax→outro)"
    echo "  hitchcock    - Slow build-up, explosive climax"
    echo "  mtv          - Fast 1-2 beat cuts, high energy"
    echo "  action       - Michael Bay rapid cuts"
    echo "  documentary  - Natural pacing, observational"
    echo "  minimalist   - Long takes, contemplative"
    echo "  wes_anderson - Symmetrical, whimsical"
}

build_image() {
    echo "Building Montage AI..."
    docker build -t montage-ai:latest .
    echo "Done."
}

run_montage() {
    local STYLE="${1:-dynamic}"
    local PRESET="${2:-medium}"
    local STABILIZE="${3:-false}"
    local ENHANCE="${4:-true}"
    local VARIANTS="${5:-1}"

    echo "🎬 Montage AI"
    echo "   Style: $STYLE"
    echo "   Preset: $PRESET"
    echo "   Stabilize: $STABILIZE"
    echo ""

    docker compose run --rm \
        -e CREATIVE_PROMPT="$STYLE" \
        -e FFMPEG_PRESET="$PRESET" \
        -e STABILIZE="$STABILIZE" \
        -e ENHANCE="$ENHANCE" \
        -e NUM_VARIANTS="$VARIANTS" \
        montage-ai
}

# Parse arguments
STYLE="dynamic"
PRESET="medium"
STABILIZE="false"
ENHANCE="true"
VARIANTS="1"

case "${1:-run}" in
    run)
        shift
        [[ -n "$1" && "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
    preview)
        PRESET="fast"
        shift
        [[ -n "$1" && "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
    hq)
        PRESET="slow"
        STABILIZE="true"
        shift
        [[ -n "$1" && "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
    list)
        list_styles
        exit 0
        ;;
    build)
        build_image
        exit 0
        ;;
    help|--help|-h)
        show_help
        exit 0
        ;;
    *)
        [[ "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
esac

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stabilize) STABILIZE="true"; shift ;;
        --no-enhance) ENHANCE="false"; shift ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

run_montage "$STYLE" "$PRESET" "$STABILIZE" "$ENHANCE" "$VARIANTS"
