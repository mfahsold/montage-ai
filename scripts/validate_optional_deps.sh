#!/bin/bash
# Validate Optional Dependencies Installation
# Tests each optional dependency group to ensure they install and work correctly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "════════════════════════════════════════════════════════════"
echo "Optional Dependencies Validator"
echo "════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test results
RESULTS=()

# Function to test installation
test_install() {
    local group=$1
    local packages=$2
    local description=$3
    
    echo -n "Testing [$group]: $description... "
    
    # Create temp venv
    VENV_DIR=$(mktemp -d)
    trap "rm -rf $VENV_DIR" EXIT
    
    python3 -m venv "$VENV_DIR" > /dev/null 2>&1
    source "$VENV_DIR/bin/activate"
    
    # Install core first
    pip install -q -e "$REPO_ROOT" 2>/dev/null || {
        deactivate
        echo -e "${RED}✗ Failed${NC} (core install failed)"
        RESULTS+=("$group:FAIL")
        return 1
    }
    
    # Install optional group
    if [ ! -z "$packages" ]; then
        pip install -q $packages 2>/dev/null || {
            deactivate
            echo -e "${RED}✗ Failed${NC} (optional install failed)"
            RESULTS+=("$group:FAIL")
            return 1
        }
    fi
    
    # Test import
    python3 -c "import montage_ai; from montage_ai.config import ProcessingSettings" 2>/dev/null || {
        deactivate
        echo -e "${RED}✗ Failed${NC} (import failed)"
        RESULTS+=("$group:FAIL")
        return 1
    }
    
    # Additional tests per group
    case $group in
        ai)
            python3 -c "import mediapipe; import librosa" 2>/dev/null || {
                deactivate
                echo -e "${RED}✗ Failed${NC} (AI imports failed)"
                RESULTS+=("$group:FAIL")
                return 1
            }
            ;;
        web)
            python3 -c "import flask; from redis import Redis" 2>/dev/null || {
                deactivate
                echo -e "${RED}✗ Failed${NC} (web imports failed)"
                RESULTS+=("$group:FAIL")
                return 1
            }
            ;;
        cloud)
            python3 -c "import soundfile" 2>/dev/null || {
                deactivate
                echo -e "${RED}✗ Failed${NC} (cloud imports failed)"
                RESULTS+=("$group:FAIL")
                return 1
            }
            ;;
    esac
    
    deactivate
    echo -e "${GREEN}✓ Passed${NC}"
    RESULTS+=("$group:PASS")
}

echo "1. Core Installation (no optional deps)"
test_install "core" "" "Basic montage-ai"

echo ""
echo "2. AI Dependencies"
test_install "ai" "mediapipe scipy librosa color-matcher" "ML & audio analysis"

echo ""
echo "3. Web Dependencies"
test_install "web" "flask redis rq msgpack" "Web UI & async jobs"

echo ""
echo "4. Cloud Dependencies"
test_install "cloud" "soundfile" "Cloud/GPU offloading"

echo ""
echo "5. All Combined"
test_install "all" "mediapipe scipy librosa color-matcher flask redis rq msgpack soundfile" "Complete install"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Installation Size Estimates"
echo "════════════════════════════════════════════════════════════"
echo ""

# Estimate sizes for each group
estimate_size() {
    local group=$1
    local packages=$2
    
    VENV_DIR=$(mktemp -d)
    trap "rm -rf $VENV_DIR" EXIT
    
    python3 -m venv "$VENV_DIR" > /dev/null 2>&1
    source "$VENV_DIR/bin/activate"
    
    pip install -q -e "$REPO_ROOT" > /dev/null 2>&1
    if [ ! -z "$packages" ]; then
        pip install -q $packages > /dev/null 2>&1
    fi
    
    SIZE=$(du -sh "$VENV_DIR" 2>/dev/null | cut -f1)
    echo "  $group: $SIZE"
    
    deactivate
}

echo "Virtual Environment Sizes:"
estimate_size "core" ""
estimate_size "core+ai" "mediapipe scipy librosa color-matcher"
estimate_size "core+web" "flask redis rq msgpack"
estimate_size "core+all" "mediapipe scipy librosa color-matcher flask redis rq msgpack soundfile"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Test Results Summary"
echo "════════════════════════════════════════════════════════════"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

for result in "${RESULTS[@]}"; do
    IFS=':' read -r group status <<< "$result"
    if [ "$status" = "PASS" ]; then
        echo -e "  ${GREEN}✓${NC} $group"
        ((PASS_COUNT++))
    else
        echo -e "  ${RED}✗${NC} $group"
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "Total: $PASS_COUNT passed, $FAIL_COUNT failed"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All optional dependencies validated successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some optional dependencies failed validation${NC}"
    exit 1
fi
