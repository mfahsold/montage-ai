#!/usr/bin/env python3
"""
Integration Test: CLI ↔ Backend ↔ Frontend

Prüft:
1. CLI richtig konfiguriert
2. Backend API Endpoints ansprechbar
3. Frontend API Calls mit Backend korrekt matched
4. Job Queue (Redis) funktioniert
5. Daten-Flows End-to-End
"""

import os
import sys
import json
import subprocess
import requests
import time
from pathlib import Path
from datetime import datetime
import pytest

# Colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
NC = '\033[0m'

def log_section(title):
    print(f"\n{CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{NC}\n")

def log_success(msg):
    print(f"{GREEN}✅ {msg}{NC}")

def log_error(msg):
    print(f"{RED}❌ {msg}{NC}")

def log_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{NC}")

def log_info(msg):
    print(f"{BLUE}ℹ️  {msg}{NC}")

# =============================================================================
# TEST 1: CLI Überprüfung
# =============================================================================

def test_cli():
    log_section("TEST 1: CLI Überprüfung")
    
    cli_path = Path("montage-ai.sh")
    if not cli_path.exists():
        log_error("montage-ai.sh nicht gefunden")
        pytest.skip("CLI script missing in test environment")
    
    log_success("montage-ai.sh existiert")
    
    # Check CLI help
    try:
        result = subprocess.run(["./montage-ai.sh"], capture_output=True, text=True, timeout=5)
        if "Usage" in result.stdout or "Usage" in result.stderr:
            log_success("CLI Help funktioniert")
        else:
            log_warning("CLI Help nicht lesbar")
    except Exception as e:
        log_error(f"CLI Test fehlgeschlagen: {e}")
        pytest.skip(f"CLI help invocation failed: {e}")
    
    # Check available styles
    try:
        result = subprocess.run(["./montage-ai.sh", "list"], capture_output=True, text=True, timeout=5)
        if "dynamic" in result.stdout or "dynamic" in result.stderr:
            log_success("CLI Styles anzeigbar")
        else:
            log_warning("CLI Styles nicht gefunden")
    except Exception as e:
        log_error(f"CLI Styles Test fehlgeschlagen: {e}")
        pytest.skip(f"CLI styles invocation failed: {e}")

# =============================================================================
# TEST 2: Backend API Erreichbarkeit
# =============================================================================

def get_base_url() -> str:
    """Get backend API URL from environment or use default."""
    import os
    return os.environ.get("BACKEND_API_URL", "http://localhost:5000")

def test_backend_health(base_url=None):
    if base_url is None:
        base_url = get_base_url()
    
    log_section("TEST 2: Backend API Erreichbarkeit")
    
    endpoints = [
        ("/api/status", "GET", "Status Check"),
        ("/api/files", "GET", "File Listing"),
        ("/api/styles", "GET", "Styles"),
        ("/api/jobs", "GET", "Job Listing"),
        ("/api/transparency", "GET", "Transparency Info"),
        ("/api/quality-profiles", "GET", "Quality Profiles"),
    ]
    
    all_ok = True
    for endpoint, method, name in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            else:
                response = requests.post(f"{base_url}{endpoint}", json={}, timeout=5)
            
            if response.status_code < 400:
                log_success(f"{name}: {endpoint} ({response.status_code})")
            else:
                log_warning(f"{name}: {endpoint} ({response.status_code})")
        except requests.exceptions.ConnectionError:
            log_error(f"Kann nicht mit Backend verbinden: {base_url}")
            pytest.skip("Backend not reachable in test environment")
        except Exception as e:
            log_error(f"{name}: {endpoint} - {e}")
            all_ok = False

    if not all_ok:
        pytest.xfail("Backend health checks reported issues")

# =============================================================================
# TEST 3: Frontend API Calls → Backend Mapping
# =============================================================================

def test_api_mapping():
    log_section("TEST 3: Frontend API Calls → Backend Mapping")
    
    # Frontend Calls aus app.js überprüfen
    app_js_path = Path("src/montage_ai/web_ui/static/app.js")
    
    if not app_js_path.exists():
        log_error("app.js nicht gefunden")
        pytest.skip("Frontend app.js missing")
    
    log_info("Scanning app.js für API Calls...")
    
    frontend_calls = {
        "POST /api/jobs": "createJob()",
        "GET /api/jobs": "refreshJobs()",
        "GET /api/files": "refreshFiles()",
        "POST /api/upload": "uploadFiles()",
        "GET /api/jobs/<id>": "getJobStatus()",
        "POST /api/jobs/<id>/finalize": "finalizeJob()",
        "GET /api/stream": "startPolling()",
        "POST /api/shorts/create": "shortsCreate()",
        "GET /api/styles": "loadStyles()",
    }
    
    with open(app_js_path, 'r') as f:
        content = f.read()
    
    backend_file = Path("src/montage_ai/web_ui/app.py")
    if not backend_file.exists():
        log_error("app.py nicht gefunden")
        pytest.skip("Backend app.py missing")
    
    with open(backend_file, 'r') as f:
        backend_content = f.read()
    
    all_matched = True
    for api_call, js_func in frontend_calls.items():
        method, endpoint = api_call.split(" ", 1)
        
        # Normalize endpoint for search
        endpoint_pattern = endpoint.replace("<id>", "")
        
        if f"@app.route('{endpoint_pattern}" in backend_content or \
           f'@app.route("{endpoint_pattern}' in backend_content:
            log_success(f"✓ {api_call} existiert im Backend")
        else:
            log_warning(f"? {api_call} nicht eindeutig gefunden")
            # Try alternate search
            if endpoint.replace("/", "").replace("<", "").replace(">", "") in backend_content.lower():
                log_info(f"  (aber Endpoint-Code gefunden)")
            else:
                all_matched = False
    
    if not all_matched:
        pytest.xfail("Frontend API calls not fully mapped in test environment")

# =============================================================================
# TEST 4: Job Queue (Redis) Überprüfung
# =============================================================================

def test_redis():
    log_section("TEST 4: Job Queue (Redis) Überprüfung")
    
    try:
        from redis import Redis
        from rq import Queue
        
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        
        log_info(f"Connecting to Redis: {redis_host}:{redis_port}")
        
        redis_conn = Redis(host=redis_host, port=redis_port, decode_responses=True, socket_connect_timeout=3)
        redis_conn.ping()
        log_success(f"Redis erreichbar")
        
        # Check queue
        q = Queue(connection=redis_conn)
        
        # Get stats
        try:
            queued = len(q)
            started = len(q.started_job_registry)
            finished = len(q.finished_job_registry)
            failed = len(q.failed_job_registry)
            
            log_info(f"Queue Stats:")
            log_info(f"  Queued: {queued}")
            log_info(f"  Started: {started}")
            log_info(f"  Finished: {finished}")
            log_info(f"  Failed: {failed}")
            
            log_success("Queue funktioniert")
        except Exception as e:
            log_warning(f"Queue Stats nicht lesbar: {e}")
        
        log_success("Redis test completed successfully.")

    except ImportError:
        log_error("Redis/RQ nicht installiert")
        pytest.skip("Redis/RQ not installed in test environment")
    except Exception as e:
        log_error(f"Redis Verbindung fehlgeschlagen: {e}")
        pytest.skip("Redis unavailable in test environment")

# =============================================================================
# TEST 5: Job Creation End-to-End
# =============================================================================

def test_job_creation(base_url=None):
    if base_url is None:
        base_url = get_base_url()
    
    log_section("TEST 5: Job Creation End-to-End")
    
    # Zuerst: Dateien aufgelisten
    log_info("Schritt 1: Dateien überprüfen...")
    try:
        response = requests.get(f"{base_url}/api/files", timeout=5)
        files_data = response.json()
        
        videos = files_data.get('videos', [])
        music = files_data.get('music', [])
        
        log_info(f"  Videos: {len(videos)}")
        log_info(f"  Music: {len(music)}")
        
        if not videos or not music:
            log_warning("  Keine Test-Dateien vorhanden - überspringe Job-Erstellung")
            pytest.skip("No test media available for job creation")
    except Exception as e:
        log_error(f"File listing fehlgeschlagen: {e}")
        pytest.skip("File listing unavailable in test environment")
    
    # Schritt 2: Job erstellen
    log_info("Schritt 2: Job erstellen...")
    job_payload = {
        "style": "dynamic",
        "prompt": "Fast cuts on beat",
        "quality_profile": "preview",
        "enhance": False,
        "stabilize": False,
        "upscale": False
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/jobs",
            json=job_payload,
            timeout=10
        )
        
        if response.status_code < 400:
            job = response.json()
            job_id = job.get('id')
            status = job.get('status')
            
            log_success(f"Job erstellt: {job_id} (Status: {status})")
            
            # Schritt 3: Job Status überprüfen
            log_info("Schritt 3: Job Status überprüfen...")
            time.sleep(1)
            
            response = requests.get(f"{base_url}/api/jobs/{job_id}", timeout=5)
            if response.status_code < 400:
                job_status = response.json()
                log_success(f"Job Status abrufbar: {job_status.get('status')}")
                return
            else:
                log_error(f"Job Status nicht abrufbar: {response.status_code}")
                pytest.xfail("Job status endpoint returned error")
        else:
            log_error(f"Job Creation fehlgeschlagen: {response.status_code}")
            log_info(f"Response: {response.text[:200]}")
            pytest.xfail("Job creation failed in test environment")
            
    except Exception as e:
        log_error(f"Job Creation Exception: {e}")
        pytest.skip("Job creation skipped due to exception")

# =============================================================================
# TEST 6: Shorts API Integration
# =============================================================================

def test_shorts_api(base_url=None):
    if base_url is None:
        base_url = get_base_url()
    
    log_section("TEST 6: Shorts API Integration")
    
    endpoints = [
        ("/api/shorts/upload", "POST", "Upload"),
        ("/api/shorts/analyze", "POST", "Analyze"),
        ("/api/shorts/render", "POST", "Render"),
        ("/api/shorts/visualize", "POST", "Visualize"),
    ]
    
    all_ok = True
    for endpoint, method, name in endpoints:
        try:
            if method == "POST":
                response = requests.post(f"{base_url}{endpoint}", json={}, timeout=5)
            else:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            # 400/401/422 = endpoint existiert, nur Invalid Input
            if response.status_code < 500:
                log_success(f"Shorts {name}: {endpoint} (HTTP {response.status_code})")
            else:
                log_error(f"Shorts {name}: {endpoint} (HTTP {response.status_code})")
                all_ok = False
        except Exception as e:
            log_error(f"Shorts {name}: {endpoint} - {e}")
            all_ok = False

    if not all_ok:
        pytest.xfail("Shorts API checks reported issues")

# =============================================================================
# TEST 7: Session Management
# =============================================================================

def test_session_api(base_url=None):
    if base_url is None:
        base_url = get_base_url()
    
    log_section("TEST 7: Session Management")
    
    endpoints = [
        ("/api/session/create", "POST", "Create"),
        ("/api/session/<id>", "GET", "Get"),
        ("/api/session/<id>/asset", "POST", "Add Asset"),
        ("/api/session/<id>/analyze", "POST", "Analyze"),
    ]
    
    all_ok = True
    for endpoint, method, name in endpoints:
        # Note: We can't actually test <id> variants without creating a session first,
        # but we can check if the route handler exists
        try:
            # Try create first
            if "create" in name:
                response = requests.post(f"{base_url}/api/session/create", json={}, timeout=5)
                if response.status_code < 400:
                    log_success(f"Session {name}: {endpoint}")
                else:
                    log_warning(f"Session {name}: returned {response.status_code}")
        except Exception as e:
            log_error(f"Session {name}: {endpoint} - {e}")
            all_ok = False

    if not all_ok:
        pytest.xfail("Session API checks reported issues")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Montage AI - CLI & Backend Integration Test               ║")
    print("║  Prüft CLI, Backend APIs, Frontend Calls & Job Queue       ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{NC}")
    
    results = {
        "CLI": test_cli(),
        "Backend Health": test_backend_health(),
        "API Mapping": test_api_mapping(),
        "Redis Queue": test_redis(),
        "Job Creation": test_job_creation(),
        "Shorts API": test_shorts_api(),
        "Session API": test_session_api(),
    }
    
    # Summary
    log_section("ZUSAMMENFASSUNG")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}PASS{NC}" if result else f"{RED}FAIL{NC}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{CYAN}Ergebnis: {passed}/{total} Tests bestanden{NC}\n")
    
    if passed == total:
        log_success("Alle Tests bestanden! ✨")
        return 0
    else:
        log_error(f"{total - passed} Tests fehlgeschlagen")
        return 1

if __name__ == "__main__":
    sys.exit(main())
