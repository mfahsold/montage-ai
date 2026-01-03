#!/bin/bash
# Montage AI - Simple CLI
set -e

# Load environment variables from .env if present
if [ -f .env ]; then
    # Export variables so they are available to child processes (like cgpu serve)
    set -a
    source .env
    set +a
fi

# cgpu server PID tracking
CGPU_PID_FILE="/tmp/cgpu_serve.pid"

show_help() {
    cat << EOF
Montage AI - Video Montage Creator

Usage: ./montage-ai.sh [COMMAND] [OPTIONS]

Commands:
  run [STYLE]     Create montage (default: dynamic)
  shorts [STYLE]  Create vertical shorts (9:16) with smart reframing
  text-edit       Text-based editing (remove fillers, edit by transcript)
  web             Start Web UI
  preview         Quick preview (fast preset)
  hq              High quality render
  retrieve        Retrieve results from cluster
  list            List available styles
  build           Build Docker image
  cgpu-start      Start cgpu serve (Gemini LLM API)
  cgpu-stop       Stop cgpu serve
  cgpu-status     Check cgpu status
  cgpu-test       Test cgpu connection

Styles:
  dynamic         Position-aware pacing (default)
  hitchcock       Suspense - slow build, fast climax
  mtv             Fast 1-2 beat cuts
  action          Michael Bay rapid cuts
  documentary     Natural, observational
  minimalist      Long contemplative takes

Options:
  --stabilize     Enable video stabilization
  --shorts        Enable Shorts mode (9:16 vertical + smart reframing)
  --no-enhance    Disable color enhancement
  --variants N    Generate N variants
  --cgpu          Enable cgpu/Gemini for Creative Director
  --cgpu-gpu      Enable cgpu cloud GPU for upscaling
  --cloud-only    Force ALL heavy lifting to cgpu (fails if cgpu unavailable)
  --export        Export timeline to OTIO/EDL/XML
  --story-engine  Enable Story Engine (narrative arc optimization)
  --captions [STYLE]  Burn-in captions (styles: tiktok, youtube, minimal, karaoke, bold, cinematic)
  --isolate-voice     Clean audio via voice isolation (requires cgpu)

Examples:
  ./montage-ai.sh run                    # Default dynamic style
  ./montage-ai.sh run hitchcock          # Hitchcock suspense
  ./montage-ai.sh preview mtv            # Quick MTV preview
  ./montage-ai.sh hq documentary         # High quality documentary
  ./montage-ai.sh run --stabilize        # With stabilization
  ./montage-ai.sh run --cgpu             # Use Gemini via cgpu
  ./montage-ai.sh run --story-engine     # Use Story Engine
  ./montage-ai.sh run --cgpu --cgpu-gpu  # Use cloud GPU for upscaling
  ./montage-ai.sh run --captions tiktok  # With TikTok-style captions
  ./montage-ai.sh hq --isolate-voice     # HQ with voice isolation
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
    echo "  vlog         - Personal, face-centric storytelling"
    echo "  sport        - High-energy action sequences"
}

run_web() {
    echo "🚀 Starting Web UI..."
    
    # Auto-start cgpu serve (since Web UI defaults to using it)
    cgpu_start || echo "⚠️ Continuing without cgpu..."

    echo "   Open http://localhost:8080 in your browser"
    echo "   ℹ️  If using Ollama on Linux, run: OLLAMA_HOST=0.0.0.0 ollama serve"
    # Pass arguments like --build to docker compose
    docker compose -f docker-compose.web.yml up "$@"
}

run_script() {
    local SCRIPT="$1"
    shift
    echo "🚀 Running script: $SCRIPT"
    docker compose run --rm \
        --entrypoint "" \
        -v "$(pwd)/scripts:/app/scripts" \
        -v "$(pwd)/src:/app/src" \
        -e PYTHONPATH=/app/src \
        -e CGPU_ENABLED="true" \
        -e CGPU_GPU_ENABLED="true" \
        montage-ai python3 "$SCRIPT" "$@"
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
    PORT=${CGPU_PORT:-8090}
    # Bind to 0.0.0.0 to allow access from Docker containers
    # Ensure both keys are set for robustness (gemini-cli sometimes demands GEMINI_API_KEY)
    (export GEMINI_API_KEY="${GEMINI_API_KEY:-$GOOGLE_API_KEY}"; exec cgpu serve --host 0.0.0.0 --port "$PORT" > /tmp/cgpu.log 2>&1) &
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
    local CAPTIONS="${8:-false}"
    local CAPTIONS_STYLE="${9:-youtube}"
    local VOICE_ISOLATION="${10:-false}"
    local STORY_ENGINE="${11:-false}"
    local STRICT_CLOUD_COMPUTE="${12:-false}"
    local SHORTS_MODE="${13:-false}"
    local EXPORT_TIMELINE="${14:-false}"

    echo "🎬 Montage AI"
    echo "   Style: $STYLE"
    echo "   Preset: $PRESET"
    echo "   Stabilize: $STABILIZE"
    echo "   Shorts Mode: $SHORTS_MODE"
    echo "   Export Timeline: $EXPORT_TIMELINE"
    echo "   cgpu LLM: $CGPU_ENABLED"
    echo "   cgpu GPU: $CGPU_GPU_ENABLED"
    echo "   Story Engine: $STORY_ENGINE"
    echo "   Strict Cloud: $STRICT_CLOUD_COMPUTE"
    [ "$CAPTIONS" = "true" ] && echo "   Captions: $CAPTIONS_STYLE"
    [ "$VOICE_ISOLATION" = "true" ] && echo "   Voice Isolation: enabled"
    echo ""

    # Auto-start cgpu serve if cgpu is enabled or features requiring it are enabled
    if [ "$CGPU_ENABLED" = "true" ] || [ "$CAPTIONS" = "true" ] || [ "$VOICE_ISOLATION" = "true" ] || [ "$STORY_ENGINE" = "true" ]; then
        cgpu_start || echo "⚠️ Continuing without cgpu..."
        # Unset GOOGLE_API_KEY to avoid conflict with cgpu's GEMINI_API_KEY
        # gemini-cli throws error if both are present
        # unset GOOGLE_API_KEY
        CGPU_ENABLED="true"  # Enable cgpu if any feature needs it
    fi

    docker compose run --rm \
        -e CREATIVE_PROMPT="$STYLE" \
        -e FFMPEG_PRESET="$PRESET" \
        -e STABILIZE="$STABILIZE" \
        -e ENHANCE="$ENHANCE" \
        -e NUM_VARIANTS="$VARIANTS" \
        -e CGPU_ENABLED="$CGPU_ENABLED" \
        -e CGPU_PORT="${CGPU_PORT:-8090}" \
        -e CGPU_MODEL="${CGPU_MODEL:-gemini-2.0-flash}" \
        -e CGPU_GPU_ENABLED="$CGPU_GPU_ENABLED" \
        -e STRICT_CLOUD_COMPUTE="$STRICT_CLOUD_COMPUTE" \
        -e ENABLE_STORY_ENGINE="$STORY_ENGINE" \
        -e SHORTS_MODE="$SHORTS_MODE" \
        -e EXPORT_TIMELINE="$EXPORT_TIMELINE" \
        -e CAPTIONS="$CAPTIONS" \
        -e CAPTIONS_STYLE="$CAPTIONS_STYLE" \
        -e VOICE_ISOLATION="$VOICE_ISOLATION" \
        -e TARGET_DURATION="${TARGET_DURATION:-0}" \
        -e MUSIC_START="${MUSIC_START:-0}" \
        -e MUSIC_END="${MUSIC_END:-0}" \
        -e QUALITY_PROFILE="${QUALITY_PROFILE:-standard}" \
        montage-ai
}

# Parse arguments
STYLE="dynamic"
PRESET="medium"
QUALITY_PROFILE="standard"
STABILIZE="false"
ENHANCE="true"
VARIANTS="1"
CGPU_ENABLED="false"
CGPU_GPU_ENABLED="false"
STRICT_CLOUD_COMPUTE="false"
EXPORT_TIMELINE="false"
CAPTIONS="false"
CAPTIONS_STYLE="youtube"
VOICE_ISOLATION="false"
STORY_ENGINE="false"
SHORTS_MODE="false"

case "${1:-run}" in
    run)
        shift
        [[ -n "$1" && "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
    shorts)
        SHORTS_MODE="true"
        CAPTIONS="true"
        CAPTIONS_STYLE="tiktok"  # Default to TikTok style for shorts
        shift
        [[ -n "$1" && "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
    text-edit)
        shift
        echo "📝 Starting Text-Based Editor..."
        # Override input volume to be writable for transcript generation
        docker compose run --rm \
            -v "$(pwd)/data/input:/data/input" \
            -e PYTHONPATH=/app/src \
            montage-ai python3 -m montage_ai.text_editor "$@"
        exit 0
        ;;
    preview)
        PRESET="fast"
        QUALITY_PROFILE="preview"
        shift
        [[ -n "$1" && "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
    hq)
        PRESET="slow"
        QUALITY_PROFILE="high"
        STABILIZE="true"
        shift
        [[ -n "$1" && "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
    list)
        list_styles
        exit 0
        ;;
    web)
        shift
        run_web "$@"
        exit 0
        ;;
    build)
        build_image
        exit 0
        ;;
    run-script)
        shift
        run_script "$@"
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
    cgpu-test)
        CGPU_HOST=localhost python3 scripts/test_cgpu_connection.py
        exit 0
        ;;
    retrieve)
        python3 scripts/retrieve_results.py
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
        --shorts) SHORTS_MODE="true"; shift ;;
        --export) EXPORT_TIMELINE="true"; shift ;;
        --no-enhance) ENHANCE="false"; shift ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        --cgpu) CGPU_ENABLED="true"; shift ;;
        --cgpu-gpu) CGPU_GPU_ENABLED="true"; shift ;;
        --cloud-only)
            CGPU_ENABLED="true"
            CGPU_GPU_ENABLED="true"
            STRICT_CLOUD_COMPUTE="true"
            shift
            ;;
        --story-engine) STORY_ENGINE="true"; shift ;;
        --captions)
            CAPTIONS="true"
            # Check if next arg is a style (not another flag)
            if [[ -n "$2" && "$2" != --* ]]; then
                CAPTIONS_STYLE="$2"; shift
            fi
            shift ;;
        --isolate-voice) VOICE_ISOLATION="true"; shift ;;
        *) shift ;;
    esac
done

run_montage "$STYLE" "$PRESET" "$STABILIZE" "$ENHANCE" "$VARIANTS" "$CGPU_ENABLED" "$CGPU_GPU_ENABLED" "$CAPTIONS" "$CAPTIONS_STYLE" "$VOICE_ISOLATION" "$STORY_ENGINE" "$STRICT_CLOUD_COMPUTE" "$SHORTS_MODE" "$EXPORT_TIMELINE"
