#!/bin/bash
# Montage AI - Simple CLI
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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
${CYAN}Montage AI${NC} - Video Montage Creator

${YELLOW}Usage:${NC} ./montage-ai.sh [COMMAND] [OPTIONS]

${YELLOW}Commands:${NC}
  ${GREEN}run${NC} [STYLE]     Create montage (default: dynamic)
  ${GREEN}shorts${NC} [STYLE]  Create vertical shorts (9:16) with smart reframing
  ${GREEN}text-edit${NC}       Text-based editing (remove fillers, edit by transcript)
  ${GREEN}web${NC}             Start Web UI
  ${GREEN}status${NC}          Check running job status and logs
  ${GREEN}preview${NC}         Quick preview (fast preset, 360p)
  ${GREEN}finalize${NC}        Finalize render (high quality, 1080p, stabilized)
  ${GREEN}hq${NC}              High quality render (1080p/4K)
  ${GREEN}download${NC} JOB_ID  Download job artifacts (video + timeline + logs)
  ${GREEN}export-to-nle${NC}   Export timeline to NLE formats (OTIO/EDL/Premiere/AAF)
  ${GREEN}check-hw${NC}        Diagnose hardware acceleration (NVENC/VAAPI/QSV)
  ${GREEN}list${NC}            List available styles
  ${GREEN}build${NC}           Build Docker image
  ${GREEN}cgpu-start${NC}      Start cgpu serve (Gemini LLM API)
  ${GREEN}cgpu-stop${NC}       Stop cgpu serve
  ${GREEN}cgpu-status${NC}     Check cgpu status
  ${GREEN}cgpu-test${NC}       Test cgpu connection

${YELLOW}Styles:${NC}
  ${CYAN}dynamic${NC}         Position-aware pacing (default)
  ${CYAN}hitchcock${NC}       Suspense - slow build, fast climax
  ${CYAN}mtv${NC}             Fast 1-2 beat cuts
  ${CYAN}action${NC}          Michael Bay rapid cuts
  ${CYAN}documentary${NC}     Natural, observational
  ${CYAN}minimalist${NC}      Long contemplative takes

${YELLOW}Options:${NC}
  ${GREEN}Video Enhancement:${NC}
  --stabilize         Enable video stabilization
  --upscale           Enable AI 4K upscaling (Real-ESRGAN)
  --denoise           Enable AI denoising (best for low-light)
  --sharpen           Enable sharpening (unsharp mask)

  ${GREEN}Color & Look:${NC}
  --color-grade [PRESET]  Color grading preset (teal_orange, cinematic, warm, etc.)
  --film-grain [TYPE]     Add film grain (fine, medium, 35mm, 16mm, 8mm)

  ${GREEN}Audio:${NC}
  --dialogue-duck     Auto-detect speech and duck music
  --audio-normalize   Normalize to -14 LUFS
  --isolate-voice     Clean audio via voice isolation (requires cgpu)
  --captions [STYLE]  Burn-in captions (tiktok, youtube, minimal, karaoke, bold)

  ${GREEN}Story & Output:${NC}
  --story-engine      Enable Story Engine (narrative arc optimization)
  --story-arc [ARC]   Story arc preset (hero_journey, mtv_energy, documentary, thriller)
  --shorts            Enable Shorts mode (9:16 vertical + smart reframing)
  --export            Export timeline to OTIO/EDL/XML
  --export-recipe     Generate enhancement Recipe Card (Markdown)

  ${GREEN}Cloud & Workflow:${NC}
  --cgpu              Enable cgpu/Gemini for Creative Director
  --cgpu-gpu          Enable cgpu cloud GPU for upscaling
  --cloud-only        Force ALL heavy lifting to cgpu
  --variants N        Generate N variants
  --no-enhance        Disable color enhancement

${YELLOW}Examples:${NC}
  ./montage-ai.sh run                    # Default dynamic style
  ./montage-ai.sh run hitchcock          # Hitchcock suspense
  ./montage-ai.sh preview mtv            # Quick MTV preview (360p)
  ./montage-ai.sh hq documentary         # High quality documentary
  ./montage-ai.sh run --stabilize        # With stabilization
  ./montage-ai.sh run --cgpu             # Use Gemini via cgpu
  ./montage-ai.sh run --story-engine     # Use Story Engine
  ./montage-ai.sh run --cgpu --cgpu-gpu  # Use cloud GPU for upscaling
  ./montage-ai.sh run --captions tiktok  # With TikTok-style captions
  ./montage-ai.sh hq --isolate-voice     # HQ with voice isolation
  
  ${YELLOW}Download Artifacts:${NC}
  ./montage-ai.sh download 20260112_114010                       # Download from local
  ./montage-ai.sh download 20260112_114010 --zip                 # Download as ZIP
  ./montage-ai.sh download 20260112_114010 --api http://host:30080  # From cluster API
  ./montage-ai.sh download 20260112_114010 --output ./my_project # Custom output dir

  ${YELLOW}Export to NLE:${NC}
  ./montage-ai.sh export-to-nle --manifest /data/output/manifest.json
  ./montage-ai.sh export-to-nle --manifest /data/output/manifest.json --formats otio edl premiere
EOF
}

list_styles() {
    echo -e "${YELLOW}Available Styles:${NC}"
    echo -e "  ${CYAN}dynamic${NC}      - Position-aware pacing (intro→build→climax→outro)"
    echo -e "  ${CYAN}hitchcock${NC}    - Slow build-up, explosive climax"
    echo -e "  ${CYAN}mtv${NC}          - Fast 1-2 beat cuts, high energy"
    echo -e "  ${CYAN}action${NC}       - Michael Bay rapid cuts"
    echo -e "  ${CYAN}documentary${NC}  - Natural pacing, observational"
    echo -e "  ${CYAN}minimalist${NC}   - Long takes, contemplative"
    echo -e "  ${CYAN}wes_anderson${NC} - Symmetrical, whimsical"
    echo -e "  ${CYAN}vlog${NC}         - Personal, face-centric storytelling"
    echo -e "  ${CYAN}sport${NC}        - High-energy action sequences"
}

run_web() {
    echo -e "${GREEN}🚀 Starting Web UI...${NC}"
    
    # Auto-start cgpu serve (since Web UI defaults to using it)
    cgpu_start || echo -e "${YELLOW}⚠️ Continuing without cgpu...${NC}"

    echo -e "   Open ${BLUE}http://localhost:8080${NC} in your browser"
    echo -e "   ℹ️  If using Ollama on Linux, run: ${CYAN}OLLAMA_HOST=0.0.0.0 ollama serve${NC}"
    # Pass arguments like --build to docker compose
    docker compose -f docker-compose.web.yml up "$@"
}

run_serve() {
    echo -e "${GREEN}🚀 Starting Web UI (Production)...${NC}"
    
    # Auto-start cgpu serve
    cgpu_start || echo -e "${YELLOW}⚠️ Continuing without cgpu...${NC}"

    # Run the app directly from source to ensure templates/static are found
    # We use the installed package for dependencies but run from the source tree
    export PYTHONPATH="/app/src:$PYTHONPATH"
    python3 -m montage_ai.web_ui.app
}

run_script() {
    local SCRIPT="$1"
    shift
    echo -e "${GREEN}🚀 Running script: $SCRIPT${NC}"
    docker compose run --rm --user "$(id -u):$(id -g)" \
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
    echo "🎬 Montage AI"
    echo "   Style: $STYLE"
    echo "   Preset: $PRESET"
    echo "   Quality: $QUALITY_PROFILE"
    echo ""
    echo "   Video Enhancement:"
    echo "     Stabilize: $STABILIZE | Upscale: $UPSCALE"
    echo "     Denoise: $DENOISE | Sharpen: $SHARPEN"
    echo ""
    echo "   Color & Look:"
    [ -n "$COLOR_GRADING" ] && echo "     Color Grade: $COLOR_GRADING @ ${COLOR_INTENSITY}"
    [ "$FILM_GRAIN" != "none" ] && echo "     Film Grain: $FILM_GRAIN"
    echo ""
    echo "   Audio:"
    [ "$DIALOGUE_DUCK" = "true" ] && echo "     Dialogue Duck: enabled"
    [ "$AUDIO_NORMALIZE" = "true" ] && echo "     Normalize: -14 LUFS"
    [ "$VOICE_ISOLATION" = "true" ] && echo "     Voice Isolation: enabled"
    [ "$CAPTIONS" = "true" ] && echo "     Captions: $CAPTIONS_STYLE"
    echo ""
    echo "   Story & Output:"
    echo "     Story Engine: $STORY_ENGINE"
    [ -n "$STORY_ARC" ] && echo "     Story Arc: $STORY_ARC"
    echo "     Shorts Mode: $SHORTS_MODE"
    echo "     Export Timeline: $EXPORT_TIMELINE"
    [ "$EXPORT_RECIPE" = "true" ] && echo "     Recipe Card: enabled"
    echo ""
    echo "   Cloud:"
    echo "     cgpu LLM: $CGPU_ENABLED | cgpu GPU: $CGPU_GPU_ENABLED"
    echo ""

    # Auto-start cgpu serve if cgpu is enabled or features requiring it are enabled
    if [ "$CGPU_ENABLED" = "true" ] || [ "$CAPTIONS" = "true" ] || [ "$VOICE_ISOLATION" = "true" ] || [ "$STORY_ENGINE" = "true" ]; then
        cgpu_start || echo "⚠️ Continuing without cgpu..."
        # Unset GOOGLE_API_KEY to avoid conflict with cgpu's GEMINI_API_KEY
        # gemini-cli throws error if both are present
        # unset GOOGLE_API_KEY
        CGPU_ENABLED="true"  # Enable cgpu if any feature needs it
    fi

    docker compose run --rm --user "$(id -u):$(id -g)" \
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
        -e STORY_ARC="$STORY_ARC" \
        -e SHORTS_MODE="$SHORTS_MODE" \
        -e EXPORT_TIMELINE="$EXPORT_TIMELINE" \
        -e EXPORT_RECIPE="$EXPORT_RECIPE" \
        -e CAPTIONS="$CAPTIONS" \
        -e CAPTIONS_STYLE="$CAPTIONS_STYLE" \
        -e VOICE_ISOLATION="$VOICE_ISOLATION" \
        -e TARGET_DURATION="${TARGET_DURATION:-0}" \
        -e MUSIC_START="${MUSIC_START:-0}" \
        -e MUSIC_END="${MUSIC_END:-0}" \
        -e QUALITY_PROFILE="${QUALITY_PROFILE:-standard}" \
        -e COLOR_GRADING="${COLOR_GRADING:-}" \
        -e COLOR_INTENSITY="${COLOR_INTENSITY:-0.7}" \
        -e UPSCALE="${UPSCALE:-false}" \
        -e CLUSTER_MODE="$CLUSTER_MODE" \
        -e CLUSTER_PARALLELISM="$CLUSTER_PARALLELISM" \
        -e DENOISE="$DENOISE" \
        -e SHARPEN="$SHARPEN" \
        -e FILM_GRAIN="$FILM_GRAIN" \
        -e DIALOGUE_DUCK="$DIALOGUE_DUCK" \
        -e AUDIO_NORMALIZE="$AUDIO_NORMALIZE" \
        montage-ai \
        python3 -m montage_ai
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
UPSCALE="false"
COLOR_GRADING=""
COLOR_INTENSITY="0.7"
# New options
DENOISE="false"
SHARPEN="false"
FILM_GRAIN="none"
DIALOGUE_DUCK="false"
AUDIO_NORMALIZE="false"
STORY_ARC=""
EXPORT_RECIPE="false"
CLUSTER_MODE="false"
CLUSTER_PARALLELISM="4"

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
        docker compose run --rm --user "$(id -u):$(id -g)" \
            -v "$(pwd)/data/input:/data/input" \
            -e PYTHONPATH=/app/src \
            montage-ai python3 -m montage_ai.text_editor "$@"
        exit 0
        ;;
    preview)
        PRESET="fast"
        QUALITY_PROFILE="preview"
        STABILIZE="false"
        ENHANCE="false"
        shift
        [[ -n "$1" && "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
    finalize)
        PRESET="slow"
        QUALITY_PROFILE="high"
        STABILIZE="true"
        ENHANCE="true"
        shift
        [[ -n "$1" && "$1" != --* ]] && { STYLE="$1"; shift; }
        ;;
    hq)
        PRESET="slow"
        QUALITY_PROFILE="high"
        STABILIZE="true"
        UPSCALE="true"
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
    serve)
        shift
        run_serve "$@"
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
    status)
        echo -e "${YELLOW}📊 Checking job status...${NC}"
        CONTAINER_ID=$(docker ps -q --filter "label=com.docker.compose.service=montage-ai" | head -n 1)
        if [[ -n "$CONTAINER_ID" ]]; then
            docker logs "$CONTAINER_ID" | tail -n 50
        else
            echo -e "${RED}No running montage-ai container found.${NC}"
        fi
        exit 0
        ;;
    cgpu-test)
        CGPU_HOST=localhost python3 scripts/test_cgpu_connection.py
        exit 0
        ;;
    download)
        shift
        if [[ -z "$1" ]]; then
            echo -e "${RED}Error: Job ID required${NC}"
            echo -e "Usage: ./montage-ai.sh download JOB_ID [--zip] [--output DIR] [--api URL]"
            exit 1
        fi
        JOB_ID="$1"
        shift

        # Default to local output dir, or API if specified
        API_URL=""
        OUTPUT_DIR="./downloads"
        ZIP_FLAG=""

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --api) API_URL="$2"; shift 2 ;;
                --output|-o) OUTPUT_DIR="$2"; shift 2 ;;
                --zip|-z) ZIP_FLAG="--zip"; shift ;;
                *) shift ;;
            esac
        done

        if [[ -n "$API_URL" ]]; then
            echo -e "${GREEN}📥 Downloading job ${JOB_ID} from ${API_URL}...${NC}"
            python3 scripts/download_job.py --job-id "$JOB_ID" --api "$API_URL" --output "$OUTPUT_DIR" $ZIP_FLAG
        else
            echo -e "${GREEN}📥 Downloading job ${JOB_ID} from local output...${NC}"
            python3 scripts/download_job.py --job-id "$JOB_ID" --local "${OUTPUT_DIR:-/data/output}" --output "$OUTPUT_DIR" $ZIP_FLAG
        fi
        exit $?
        ;;
    export-to-nle)
        shift
        echo -e "${GREEN}📤 Exporting timeline to NLE formats...${NC}"
        python3 -m montage_ai.export.cli "$@"
        exit $?
        ;;
    check-hw)
        shift
        python3 -m montage_ai.cli check-hw "$@"
        exit $?
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
        --upscale) UPSCALE="true"; shift ;;
        --color-grade)
            if [[ -n "$2" && "$2" != --* ]]; then
                COLOR_GRADING="$2"; shift
            else
                COLOR_GRADING="cinematic"
            fi
            shift ;;
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
        --story-arc)
            if [[ -n "$2" && "$2" != --* ]]; then
                STORY_ARC="$2"; shift
            fi
            shift ;;
        --captions)
            CAPTIONS="true"
            # Check if next arg is a style (not another flag)
            if [[ -n "$2" && "$2" != --* ]]; then
                CAPTIONS_STYLE="$2"; shift
            fi
            shift ;;
        --isolate-voice) VOICE_ISOLATION="true"; shift ;;
        --denoise) DENOISE="true"; shift ;;
        --sharpen) SHARPEN="true"; shift ;;
        --dialogue-duck) DIALOGUE_DUCK="true"; shift ;;
        --audio-normalize) AUDIO_NORMALIZE="true"; shift ;;
        --film-grain)
            if [[ -n "$2" && "$2" != --* ]]; then
                FILM_GRAIN="$2"; shift
            else
                FILM_GRAIN="fine"
            fi
            shift ;;
        --export-recipe) EXPORT_RECIPE="true"; shift ;;
        --cluster) CLUSTER_MODE="true"; shift ;;
        --cluster-parallelism) CLUSTER_PARALLELISM="$2"; shift 2 ;;
        *) shift ;;
    esac
done

run_montage
