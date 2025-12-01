#!/bin/bash
# Montage AI - Quick Edit Script
# Usage: ./montage-ai.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
CREATIVE_PROMPT="${CREATIVE_PROMPT:-}"
CUT_STYLE="${CUT_STYLE:-dynamic}"
STABILIZE="${STABILIZE:-false}"
UPSCALE="${UPSCALE:-false}"
ENHANCE="${ENHANCE:-true}"
NUM_VARIANTS="${NUM_VARIANTS:-1}"
VERBOSE="${VERBOSE:-true}"
EXPORT_TIMELINE="${EXPORT_TIMELINE:-false}"
DEEP_ANALYSIS="${DEEP_ANALYSIS:-false}"

# Performance settings
CPUS="${CPUS:-6}"
MEMORY="${MEMORY:-24g}"
FFMPEG_PRESET="${FFMPEG_PRESET:-medium}"

# Help message
show_help() {
    echo "Montage AI - AI-Powered Video Montage Creation"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  edit      Run the video editor (default)"
    echo "  build     Build the Docker image"
    echo "  shell     Open a shell in the container"
    echo "  logs      Show container logs"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CREATIVE_PROMPT   Natural language editing prompt"
    echo "  CUT_STYLE         Legacy style: fast, hyper, slow, dynamic"
    echo "  STABILIZE         Enable stabilization (true/false)"
    echo "  UPSCALE           Enable AI upscaling (true/false)"
    echo "  ENHANCE           Enable color enhancement (true/false)"
    echo "  NUM_VARIANTS      Number of output variants"
    echo "  DEEP_ANALYSIS     Enable deep footage analysis"
    echo ""
    echo "Examples:"
    echo "  # Quick edit with default settings"
    echo "  ./montage-ai.sh"
    echo ""
    echo "  # Hitchcock style with stabilization"
    echo "  CREATIVE_PROMPT='Edit like Hitchcock' STABILIZE=true ./montage-ai.sh"
    echo ""
    echo "  # MTV fast-paced style"
    echo "  CREATIVE_PROMPT='MTV music video style' ./montage-ai.sh"
    echo ""
    echo "  # Full quality render"
    echo "  STABILIZE=true UPSCALE=true FFMPEG_PRESET=slow ./montage-ai.sh"
}

# Build Docker image
build_image() {
    echo "🔨 Building Montage AI Docker image..."
    docker build -t montage-ai:latest .
    echo "✅ Build complete"
}

# Run the editor
run_editor() {
    echo "🎬 Starting Montage AI..."
    echo "   Creative Prompt: ${CREATIVE_PROMPT:-'(using default)'}"
    echo "   Stabilize: $STABILIZE"
    echo "   Upscale: $UPSCALE"
    echo "   Enhance: $ENHANCE"
    echo "   Variants: $NUM_VARIANTS"
    echo ""
    
    # Check if data directories exist
    if [ ! -d "./data/input" ]; then
        echo "❌ Error: ./data/input directory not found"
        echo "   Please create it and add your video files"
        exit 1
    fi
    
    if [ ! -d "./data/music" ]; then
        echo "❌ Error: ./data/music directory not found"
        echo "   Please create it and add your music file"
        exit 1
    fi
    
    # Create output directory
    mkdir -p ./data/output
    
    # Run container
    docker compose run --rm \
        -e CREATIVE_PROMPT="$CREATIVE_PROMPT" \
        -e CUT_STYLE="$CUT_STYLE" \
        -e STABILIZE="$STABILIZE" \
        -e UPSCALE="$UPSCALE" \
        -e ENHANCE="$ENHANCE" \
        -e NUM_VARIANTS="$NUM_VARIANTS" \
        -e VERBOSE="$VERBOSE" \
        -e EXPORT_TIMELINE="$EXPORT_TIMELINE" \
        -e DEEP_ANALYSIS="$DEEP_ANALYSIS" \
        -e FFMPEG_PRESET="$FFMPEG_PRESET" \
        montage-ai
}

# Open shell in container
run_shell() {
    echo "🐚 Opening shell in Montage AI container..."
    docker compose run --rm --entrypoint /bin/bash montage-ai
}

# Show logs
show_logs() {
    docker compose logs -f montage-ai
}

# Main
case "${1:-edit}" in
    edit)
        run_editor
        ;;
    build)
        build_image
        ;;
    shell)
        run_shell
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
