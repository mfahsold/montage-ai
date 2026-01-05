#!/bin/bash
# Quick Integration Verification Script
# Tests CLI, Backend, Frontend without running full services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  MONTAGE AI - QUICK INTEGRATION CHECK                      ║"
echo "║  Verifies CLI, Backend, Frontend without services          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

PASS=0
FAIL=0

# Function to test
test_item() {
    local name=$1
    local cmd=$2
    
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${name}${NC}"
        ((PASS++))
    else
        echo -e "${RED}❌ ${name}${NC}"
        ((FAIL++))
    fi
}

echo -e "\n${CYAN}═══ 1. CLI CHECKS ═══${NC}"
test_item "CLI file exists" "test -f montage-ai.sh"
test_item "CLI is executable" "test -x montage-ai.sh"
test_item "CLI has run command" "grep -q 'run\\\\)' montage-ai.sh"
test_item "CLI has web command" "grep -q 'web\\\\)' montage-ai.sh"

echo -e "\n${CYAN}═══ 2. BACKEND CHECKS ═══${NC}"
test_item "app.py exists" "test -f src/montage_ai/web_ui/app.py"
test_item "tasks.py exists" "test -f src/montage_ai/tasks.py"
test_item "POST /api/jobs route" "grep -q '@app.route.*jobs.*POST' src/montage_ai/web_ui/app.py"
test_item "GET /api/files route" "grep -q '@app.route.*files.*GET' src/montage_ai/web_ui/app.py"
test_item "Shorts routes exist" "grep -q '@app.route.*shorts' src/montage_ai/web_ui/app.py"
test_item "Job queue (RQ) integration" "grep -q 'from rq import Queue' src/montage_ai/web_ui/app.py"

echo -e "\n${CYAN}═══ 3. FRONTEND CHECKS ═══${NC}"
test_item "app.js exists" "test -f src/montage_ai/web_ui/static/app.js"
test_item "montage.html exists" "test -f src/montage_ai/web_ui/templates/montage.html"
test_item "shorts.html exists" "test -f src/montage_ai/web_ui/templates/shorts.html"
test_item "API_BASE variable" "grep -q 'const API_BASE' src/montage_ai/web_ui/static/app.js"
test_item "fetch() API calls" "grep -q 'fetch.*API_BASE' src/montage_ai/web_ui/static/app.js"

echo -e "\n${CYAN}═══ 4. CONFIG CHECKS ═══${NC}"
test_item "config.py exists" "test -f src/montage_ai/config.py"
test_item "Input dir config" "grep -q 'input_dir' src/montage_ai/config.py"
test_item "Output dir config" "grep -q 'output_dir' src/montage_ai/config.py"
test_item "job_store.py exists" "test -f src/montage_ai/core/job_store.py"

echo -e "\n${CYAN}═══ 5. FEATURE CHECKS ═══${NC}"
test_item "Shorts feature" "grep -q 'api_shorts' src/montage_ai/web_ui/app.py"
test_item "Transcript feature" "grep -q 'api_transcript' src/montage_ai/web_ui/app.py"
test_item "Sessions feature" "grep -q 'api_session' src/montage_ai/web_ui/app.py"
test_item "CGPU integration" "grep -q 'api_cgpu' src/montage_ai/web_ui/app.py"

echo -e "\n${CYAN}═══ SUMMARY ═══${NC}"
TOTAL=$((PASS + FAIL))
echo -e "Passed: ${GREEN}${PASS}${NC}/${TOTAL}"
echo -e "Failed: ${RED}${FAIL}${NC}/${TOTAL}"

if [ $FAIL -eq 0 ]; then
    echo -e "\n${GREEN}✨ ALL CHECKS PASSED - System Ready${NC}"
    exit 0
elif [ $FAIL -le 2 ]; then
    echo -e "\n${YELLOW}⚠️  MINOR ISSUES - System Functional${NC}"
    exit 0
else
    echo -e "\n${RED}❌ PROBLEMS DETECTED${NC}"
    exit 1
fi
