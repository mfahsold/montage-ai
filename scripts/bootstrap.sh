#!/usr/bin/env bash
#
# Montage AI - Bootstrap Script
# Idempotent prerequisite installer for development and production environments.
#
# Usage:
#   ./scripts/bootstrap.sh [--dev] [--check-only]
#
# Options:
#   --dev         Install development dependencies (pytest, pre-commit, etc.)
#   --check-only  Only check prerequisites, don't install anything
#
# Supports: Ubuntu/Debian, Fedora/RHEL, macOS (Homebrew), Arch Linux
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (for finding repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
DEV_MODE=false
CHECK_ONLY=false
for arg in "$@"; do
    case $arg in
        --dev) DEV_MODE=true ;;
        --check-only) CHECK_ONLY=true ;;
        -h|--help)
            echo "Usage: $0 [--dev] [--check-only]"
            echo "  --dev         Install development dependencies"
            echo "  --check-only  Only check prerequisites"
            exit 0
            ;;
    esac
done

# =============================================================================
# Utility Functions
# =============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

command_exists() {
    command -v "$1" &>/dev/null
}

check_version() {
    local cmd="$1"
    local min_version="$2"
    local current
    current=$("$cmd" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    
    if [[ -z "$current" ]]; then
        return 1
    fi
    
    # Simple version comparison (works for most cases)
    printf '%s\n%s' "$min_version" "$current" | sort -V -C
}

# =============================================================================
# OS Detection
# =============================================================================

detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_LIKE="${ID_LIKE:-$ID}"
    elif [[ "$(uname)" == "Darwin" ]]; then
        OS_ID="macos"
        OS_LIKE="macos"
    else
        OS_ID="unknown"
        OS_LIKE="unknown"
    fi
}

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_python() {
    if command_exists python3; then
        local version
        version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ "$version" == "3.10" || "$version" == "3.11" || "$version" == "3.12" || "$version" == "3.13" ]]; then
            log_ok "Python $version found"
            return 0
        else
            log_warn "Python $version found, but 3.10+ recommended"
            return 0
        fi
    else
        log_error "Python 3 not found"
        return 1
    fi
}

check_ffmpeg() {
    if command_exists ffmpeg; then
        local version
        version=$(ffmpeg -version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        log_ok "FFmpeg $version found"
        
        # Check for GPU encoders
        local has_gpu=false
        if ffmpeg -encoders 2>/dev/null | grep -q h264_nvenc; then
            log_info "  └─ NVENC (NVIDIA) available"
            has_gpu=true
        fi
        if ffmpeg -encoders 2>/dev/null | grep -q h264_vaapi; then
            log_info "  └─ VAAPI (AMD/Intel) available"
            has_gpu=true
        fi
        if ffmpeg -encoders 2>/dev/null | grep -q h264_qsv; then
            log_info "  └─ QuickSync (Intel) available"
            has_gpu=true
        fi
        if ffmpeg -encoders 2>/dev/null | grep -q h264_videotoolbox; then
            log_info "  └─ VideoToolbox (Apple) available"
            has_gpu=true
        fi
        if [[ "$has_gpu" == "false" ]]; then
            log_warn "  └─ No GPU encoders detected (CPU encoding only)"
        fi
        return 0
    else
        log_error "FFmpeg not found"
        return 1
    fi
}

check_docker() {
    if command_exists docker; then
        local version
        version=$(docker --version 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        log_ok "Docker $version found"
        
        # Check if running
        if docker info &>/dev/null; then
            log_info "  └─ Docker daemon is running"
        else
            log_warn "  └─ Docker daemon not running (start with: sudo systemctl start docker)"
        fi
        return 0
    else
        log_error "Docker not found"
        return 1
    fi
}

check_gpu_access() {
    log_info "Checking GPU access..."
    
    # NVIDIA
    if command_exists nvidia-smi; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        log_ok "  NVIDIA GPU: $gpu_name"
    fi
    
    # VAAPI (AMD/Intel on Linux)
    if [[ -d /dev/dri ]]; then
        log_ok "  /dev/dri exists (VAAPI possible)"
        if [[ -r /dev/dri/renderD128 ]]; then
            log_ok "  /dev/dri/renderD128 readable"
        else
            log_warn "  /dev/dri/renderD128 not readable (add user to 'render' or 'video' group)"
        fi
    fi
    
    # macOS GPU
    if [[ "$OS_ID" == "macos" ]]; then
        log_ok "  macOS detected (VideoToolbox available)"
    fi
}

check_python_deps() {
    log_info "Checking Python dependencies..."
    
    if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
        local missing=()
        while IFS= read -r line; do
            # Skip comments and empty lines
            [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
            # Extract package name (before any version specifier)
            local pkg
            pkg=$(echo "$line" | sed -E 's/([a-zA-Z0-9_-]+).*/\1/')
            if ! python3 -c "import $pkg" &>/dev/null; then
                # Try with underscores replaced by dashes and vice versa
                local pkg_alt="${pkg//-/_}"
                if ! python3 -c "import $pkg_alt" &>/dev/null; then
                    missing+=("$pkg")
                fi
            fi
        done < <(grep -v '^\s*#' "$REPO_ROOT/requirements.txt")
        
        if [[ ${#missing[@]} -eq 0 ]]; then
            log_ok "All Python dependencies installed"
        else
            log_warn "Missing Python packages: ${missing[*]}"
            return 1
        fi
    else
        log_warn "requirements.txt not found"
    fi
}

# =============================================================================
# Installation Functions
# =============================================================================

install_system_deps() {
    log_info "Installing system dependencies..."
    
    case "$OS_ID" in
        ubuntu|debian|pop)
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-pip python3-venv \
                ffmpeg \
                libsndfile1 \
                libgl1-mesa-glx \
                git
            ;;
        fedora|rhel|centos|rocky|alma)
            sudo dnf install -y \
                python3 python3-pip \
                ffmpeg \
                libsndfile \
                mesa-libGL \
                git
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm --needed \
                python python-pip \
                ffmpeg \
                libsndfile \
                mesa \
                git
            ;;
        macos)
            if ! command_exists brew; then
                log_error "Homebrew not found. Install from https://brew.sh"
                return 1
            fi
            brew install python ffmpeg libsndfile git
            ;;
        *)
            log_error "Unsupported OS: $OS_ID"
            log_info "Please install manually: python3, ffmpeg, libsndfile, git"
            return 1
            ;;
    esac
}

install_python_deps() {
    log_info "Installing Python dependencies..."
    
    cd "$REPO_ROOT"
    
    # Create venv if not exists
    if [[ ! -d .venv ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate and install
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if [[ "$DEV_MODE" == "true" && -f requirements-dev.txt ]]; then
        log_info "Installing development dependencies..."
        pip install -r requirements-dev.txt
    fi
    
    log_ok "Python dependencies installed"
}

install_docker() {
    log_info "Installing Docker..."
    
    case "$OS_ID" in
        ubuntu|debian|pop)
            # Add Docker's official GPG key
            sudo apt-get update
            sudo apt-get install -y ca-certificates curl gnupg
            sudo install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/$OS_ID/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            sudo chmod a+r /etc/apt/keyrings/docker.gpg
            
            # Add repository
            echo \
                "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS_ID \
                $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
                sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            # Install
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            
            # Add user to docker group
            sudo usermod -aG docker "$USER"
            log_warn "Log out and back in for docker group membership to take effect"
            ;;
        fedora|rhel|centos|rocky|alma)
            sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
            sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            sudo systemctl enable --now docker
            sudo usermod -aG docker "$USER"
            ;;
        macos)
            if command_exists brew; then
                brew install --cask docker
                log_info "Open Docker Desktop to complete installation"
            else
                log_error "Install Docker Desktop from https://docker.com/products/docker-desktop"
            fi
            ;;
        *)
            log_error "Please install Docker manually: https://docs.docker.com/engine/install/"
            ;;
    esac
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo ""
    echo "=================================================="
    echo "  Montage AI - Bootstrap Script"
    echo "=================================================="
    echo ""
    
    detect_os
    log_info "Detected OS: $OS_ID"
    echo ""
    
    # Run checks
    local checks_passed=true
    
    echo "--- Prerequisites Check ---"
    check_python || checks_passed=false
    check_ffmpeg || checks_passed=false
    check_docker || checks_passed=false
    check_gpu_access
    echo ""
    
    if [[ "$CHECK_ONLY" == "true" ]]; then
        if [[ "$checks_passed" == "true" ]]; then
            log_ok "All prerequisites met!"
            exit 0
        else
            log_error "Some prerequisites missing"
            exit 1
        fi
    fi
    
    # Install if needed
    if [[ "$checks_passed" == "false" ]]; then
        echo "--- Installing Missing Dependencies ---"
        read -p "Install missing dependencies? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            command_exists python3 || install_system_deps
            command_exists ffmpeg || install_system_deps
            command_exists docker || install_docker
        fi
    fi
    
    # Python deps
    echo ""
    echo "--- Python Dependencies ---"
    check_python_deps || {
        read -p "Install Python dependencies in venv? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_python_deps
        fi
    }
    
    echo ""
    echo "=================================================="
    log_ok "Bootstrap complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate venv:  source .venv/bin/activate"
    echo "  2. Build Docker:   ./montage-ai.sh build"
    echo "  3. Run montage:    ./montage-ai.sh run"
    echo "=================================================="
}

main "$@"
