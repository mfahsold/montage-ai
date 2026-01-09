#!/bin/bash
# Montage AI Job Status & Control Helper
# Simple utility to check job status, logs, and manage runs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/montage_run.log"
OUTPUT_DIR="/data/output"
PID_FILE="/tmp/montage.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_ok() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Check if job is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo $PID
            return 0
        fi
    fi
    return 1
}

# Show status
show_status() {
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  🎬 MONTAGE AI JOB STATUS"
    echo "════════════════════════════════════════════════════"
    
    if PID=$(is_running); then
        log_ok "Job running (PID: $PID)"
        
        # Show uptime
        UPTIME=$(($(date +%s) - $(stat -f %B "$PID_FILE" 2>/dev/null || stat -c %Y "$PID_FILE" 2>/dev/null || echo 0)))
        echo "  ⏱️  Running for: $((UPTIME / 60))m $((UPTIME % 60))s"
        
        # Show recent progress
        if [ -f "$LOG_FILE" ]; then
            LATEST=$(tail -5 "$LOG_FILE" | grep -E "📊|🚀|✓|Phase" | tail -1)
            if [ ! -z "$LATEST" ]; then
                echo "  📋 Latest: $LATEST"
            fi
        fi
    else
        log_warn "No running job"
    fi
    
    # Show output status
    if [ -d "$OUTPUT_DIR" ]; then
        COUNT=$(ls -1 "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
        if [ $COUNT -gt 0 ]; then
            log_ok "Output ready ($COUNT videos in $OUTPUT_DIR/)"
            ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | tail -3 | awk '{print "     " $9 " (" $5 ")"}'
        else
            log_warn "No output videos yet"
        fi
    fi
    
    echo "════════════════════════════════════════════════════"
    echo ""
}

# Show logs
show_logs() {
    local LINES=${1:-50}
    if [ -f "$LOG_FILE" ]; then
        echo ""
        log_info "Last $LINES lines of log:"
        echo "────────────────────────────────────────────────────"
        tail -$LINES "$LOG_FILE"
    else
        log_error "Log file not found: $LOG_FILE"
    fi
}

# Stop job
stop_job() {
    if PID=$(is_running); then
        log_warn "Stopping job (PID: $PID)..."
        kill -15 $PID 2>/dev/null || true
        sleep 2
        
        if ps -p $PID > /dev/null 2>&1; then
            log_warn "Force killing..."
            kill -9 $PID 2>/dev/null || true
        fi
        
        log_ok "Job stopped"
        rm -f "$PID_FILE"
    else
        log_warn "No running job to stop"
    fi
}

# Cleanup temp files
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/scene_detect_*.mp4 2>/dev/null || true
    rm -f /tmp/montage_* 2>/dev/null || true
    log_ok "Cleanup complete"
}

# Main command dispatch
CMD=${1:-status}

case "$CMD" in
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-50}"
        ;;
    stop)
        stop_job
        ;;
    cleanup)
        cleanup
        ;;
    restart)
        stop_job
        sleep 1
        cleanup
        log_ok "Ready to restart. Run: ./montage-ai.sh run"
        ;;
    *)
        cat << EOF
${BLUE}Montage AI Job Status Helper${NC}

Usage: $0 <command> [args]

Commands:
  status              Show job status (default)
  logs [LINES]        Show last N lines of logs (default: 50)
  stop                Stop running job
  cleanup             Clean temporary files
  restart             Stop, cleanup, and prepare for new run

Examples:
  $0 status           # Check current status
  $0 logs 100         # Show last 100 lines
  $0 restart          # Restart everything

EOF
        ;;
esac
