#!/bin/bash
# Montage AI - Simple CLI
set -e

# cgpu server PID tracking
CGPU_PID_FILE="/tmp/cgpu_serve.pid"

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
  cgpu-start      Start cgpu serve (Gemini LLM API)
  cgpu-stop       Stop cgpu serve
  cgpu-status     Check cgpu status

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
  --cgpu          Enable cgpu/Gemini for Creative Director
  --cgpu-gpu      Enable cgpu cloud GPU for upscaling

Examples:
  ./montage-ai.sh run                    # Default dynamic style
  ./montage-ai.sh run hitchcock          # Hitchcock suspense
  ./montage-ai.sh preview mtv            # Quick MTV preview
  ./montage-ai.sh hq documentary         # High quality documentary
  ./montage-ai.sh run --stabilize        # With stabilization
  ./montage-ai.sh run --cgpu             # Use Gemini via cgpu
  ./montage-ai.sh run --cgpu --cgpu-gpu  # Use cloud GPU for upscaling
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

# cgpu management functions
cgpu_start() {
    if [ -f "$CGPU_PID_FILE" ] && kill -0 "$(cat "$CGPU_PID_FILE")" 2>/dev/null; then
        echo "cgpu serve already running (PID: $(cat "$CGPU_PID_FILE"))"
        return 0
    fi
    
    if ! command -v cgpu &> /dev/null; then
        echo "❌ cgpu not installed. Install with: npm i -g cgpu"
        echo "   Also need gemini-cli: https://github.com/google-gemini/gemini-cli"
        return 1
    fi
    
    echo "🚀 Starting cgpu serve..."
    PORT=${CGPU_PORT:-8080}
    cgpu serve --port "$PORT" &
    echo $! > "$CGPU_PID_FILE"
    sleep 2
    
    if kill -0 "$(cat "$CGPU_PID_FILE")" 2>/dev/null; then
        echo "✅ cgpu serve started (PID: $(cat "$CGPU_PID_FILE"))"
        echo "   Gemini API available at http://localhost:$PORT/v1"
    else
        echo "❌ Failed to start cgpu serve"
        rm -f "$CGPU_PID_FILE"
        return 1
    fi
}

cgpu_stop() {
    if [ -f "$CGPU_PID_FILE" ]; then
        PID=$(cat "$CGPU_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "✅ cgpu serve stopped (PID: $PID)"
        fi
        rm -f "$CGPU_PID_FILE"
    else
        echo "cgpu serve not running"
    fi
}

cgpu_status() {
    echo "cgpu Status:"
    
    if command -v cgpu &> /dev/null; then
        echo "  ✅ cgpu installed: $(cgpu --version 2>/dev/null || echo 'version unknown')"
    else
        echo "  ❌ cgpu not installed"
    fi
    
    if [ -f "$CGPU_PID_FILE" ] && kill -0 "$(cat "$CGPU_PID_FILE")" 2>/dev/null; then
        echo "  ✅ cgpu serve running (PID: $(cat "$CGPU_PID_FILE"))"
    else
        echo "  ⚪ cgpu serve not running"
    fi
    
    # Check if gemini-cli is available
    if command -v gemini &> /dev/null; then
        echo "  ✅ gemini-cli installed"
    else
        echo "  ⚠️ gemini-cli not found (needed for cgpu serve)"
    fi
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
    local CGPU_ENABLED="${6:-false}"
    local CGPU_GPU_ENABLED="${7:-false}"

    echo "🎬 Montage AI"
    echo "   Style: $STYLE"
    echo "   Preset: $PRESET"
    echo "   Stabilize: $STABILIZE"
    echo "   cgpu LLM: $CGPU_ENABLED"
    echo "   cgpu GPU: $CGPU_GPU_ENABLED"
    echo ""
    
    # Auto-start cgpu serve if cgpu is enabled
    if [ "$CGPU_ENABLED" = "true" ]; then
        cgpu_start || echo "⚠️ Continuing without cgpu..."
    fi

    docker compose run --rm \
        -e CREATIVE_PROMPT="$STYLE" \
        -e FFMPEG_PRESET="$PRESET" \
        -e STABILIZE="$STABILIZE" \
        -e ENHANCE="$ENHANCE" \
        -e NUM_VARIANTS="$VARIANTS" \
        -e CGPU_ENABLED="$CGPU_ENABLED" \
        -e CGPU_GPU_ENABLED="$CGPU_GPU_ENABLED" \
        montage-ai
}

# Parse arguments
STYLE="dynamic"
PRESET="medium"
STABILIZE="false"
ENHANCE="true"
VARIANTS="1"
CGPU_ENABLED="false"
CGPU_GPU_ENABLED="false"

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
    cgpu-start)
        cgpu_start
        exit 0
        ;;
    cgpu-stop)
        cgpu_stop
        exit 0
        ;;
    cgpu-status)
        cgpu_status
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
        --cgpu) CGPU_ENABLED="true"; shift ;;
        --cgpu-gpu) CGPU_GPU_ENABLED="true"; shift ;;
        *) shift ;;
    esac
done

run_montage "$STYLE" "$PRESET" "$STABILIZE" "$ENHANCE" "$VARIANTS" "$CGPU_ENABLED" "$CGPU_GPU_ENABLED"
