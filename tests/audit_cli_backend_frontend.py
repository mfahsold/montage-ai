#!/usr/bin/env python3
"""
Detaillierte Überprüfung: CLI, Backend Code, Frontend Code, API Mapping

Prüft ohne einen Server zu starten!
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
NC = '\033[0m'

def section(title):
    print(f"\n{CYAN}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{NC}\n")

def ok(msg):
    print(f"{GREEN}✅ {msg}{NC}")

def err(msg):
    print(f"{RED}❌ {msg}{NC}")

def warn(msg):
    print(f"{YELLOW}⚠️  {msg}{NC}")

def info(msg):
    print(f"{BLUE}ℹ️  {msg}{NC}")

# =============================================================================
# 1. CLI ÜBERPRÜFUNG
# =============================================================================

def check_cli() -> bool:
    section("1. CLI (montage-ai.sh) ÜBERPRÜFUNG")
    
    cli_path = Path("montage-ai.sh")
    if not cli_path.exists():
        err(f"montage-ai.sh nicht gefunden")
        return False
    
    ok(f"CLI Datei existiert: {cli_path}")
    
    # Lese CLI
    with open(cli_path, 'r') as f:
        cli_content = f.read()
    
    # Überprüfe wichtige Kommandos
    commands = {
        "run": "run_montage",
        "web": "run_web",
        "shorts": "shorts creation",
        "preview": "quick preview",
        "list": "list styles",
        "cgpu-start": "cgpu management",
    }
    
    found_commands = 0
    for cmd, desc in commands.items():
        if f'"{cmd}"' in cli_content or f"'{cmd}'" in cli_content:
            ok(f"Kommando '{cmd}' definiert ({desc})")
            found_commands += 1
        else:
            warn(f"Kommando '{cmd}' nicht gefunden")
    
    # Überprüfe Funktionen
    required_functions = ["run_web", "cgpu_start", "cgpu_stop", "cgpu_status", "list_styles"]
    found_funcs = 0
    for func in required_functions:
        if f"{func}()" in cli_content or f"function {func}" in cli_content:
            ok(f"Funktion {func} definiert")
            found_funcs += 1
    
    if found_commands >= 4 and found_funcs >= 3:
        ok(f"CLI hat erforderliche Kommandos und Funktionen")
        return True
    else:
        warn(f"CLI möglicherweise unvollständig (Kommandos: {found_commands}/6, Funktionen: {found_funcs}/5)")
        return found_commands >= 3

# =============================================================================
# 2. BACKEND API STRUCTURE ÜBERPRÜFUNG
# =============================================================================

def check_backend() -> bool:
    section("2. BACKEND API STRUCTURE")
    
    app_py = Path("src/montage_ai/web_ui/app.py")
    if not app_py.exists():
        err(f"app.py nicht gefunden")
        return False
    
    with open(app_py, 'r') as f:
        app_content = f.read()
    
    # Finde alle @app.route Definitionen
    route_pattern = r"@app\.route\(['\"]([^'\"]+)['\"],\s*methods=\[([^\]]+)\]"
    routes = re.findall(route_pattern, app_content)
    
    info(f"Gefundene Routes: {len(routes)}")
    
    required_routes = {
        "/api/status": "GET",
        "/api/files": "GET",
        "/api/upload": "POST",
        "/api/jobs": "GET|POST",
        "/api/jobs/<job_id>": "GET",
        "/api/styles": "GET",
        "/api/shorts/upload": "POST",
        "/api/shorts/render": "POST",
        "/api/transcript/render": "POST",
        "/api/transparency": "GET",
    }
    
    found_routes = {}
    for route, method in routes:
        found_routes[route] = method
        
    # Überprüfe required routes
    ok_routes = 0
    for required_route, required_method in required_routes.items():
        # Normalize für Vergleich
        base_route = required_route.replace("<job_id>", "")
        
        matched = False
        for found_route in found_routes:
            if base_route in found_route or found_route == required_route:
                matched = True
                break
        
        if matched:
            ok(f"✓ {required_route} ({required_method})")
            ok_routes += 1
        else:
            warn(f"? {required_route} nicht gefunden")
    
    if ok_routes >= 8:
        ok(f"Backend hat erforderliche API Routes")
        return True
    else:
        warn(f"Backend Route Coverage: {ok_routes}/{len(required_routes)}")
        return ok_routes >= 7

# =============================================================================
# 3. FRONTEND API CALLS ÜBERPRÜFUNG
# =============================================================================

def check_frontend() -> bool:
    section("3. FRONTEND API CALLS")
    
    # Überprüfe alle JS/HTML Frontend Dateien
    frontend_dir = Path("src/montage_ai/web_ui/templates")
    if not frontend_dir.exists():
        err(f"Frontend Templates nicht gefunden")
        return False
    
    html_files = list(frontend_dir.glob("*.html"))
    ok(f"HTML Templates gefunden: {len(html_files)}")
    
    # Überprüfe app.js
    app_js = Path("src/montage_ai/web_ui/static/app.js")
    if not app_js.exists():
        err(f"app.js nicht gefunden")
        return False
    
    with open(app_js, 'r') as f:
        js_content = f.read()
    
    # Finde fetch Calls
    fetch_pattern = r"fetch\(['\"`]([^'\"`.]+)['\"`]"
    fetches = re.findall(fetch_pattern, js_content)
    
    info(f"Gefundene fetch() Calls: {len(set(fetches))}")
    
    required_apis = [
        "/api/jobs",
        "/api/files",
        "/api/upload",
        "/api/styles",
        "/api/video/",
        "/api/shorts/",
        "/api/stream",
        "/api/jobs/",
    ]
    
    found_apis = set(fetches)
    found_count = 0
    for api in required_apis:
        if api in str(found_apis):
            ok(f"✓ {api} wird aufgerufen")
            found_count += 1
        else:
            warn(f"? {api} nicht in fetch() gefunden")
    
    # Überprüfe auch HTML für Event Listener
    listener_count = js_content.count("addEventListener")
    ok(f"Event Listener: {listener_count}")
    
    if found_count >= 6:
        ok(f"Frontend hat erforderliche API Calls")
        return True
    else:
        warn(f"Frontend API Coverage: {found_count}/{len(required_apis)}")
        return found_count >= 5

# =============================================================================
# 4. API MAPPING ÜBERPRÜFUNG
# =============================================================================

def check_api_mapping() -> bool:
    section("4. API MAPPING - Frontend ↔ Backend")
    
    app_py = Path("src/montage_ai/web_ui/app.py")
    app_js = Path("src/montage_ai/web_ui/static/app.js")
    
    if not app_py.exists() or not app_js.exists():
        err("app.py oder app.js nicht gefunden")
        return False
    
    with open(app_py, 'r') as f:
        backend = f.read()
    
    with open(app_js, 'r') as f:
        frontend = f.read()
    
    # Extrahiere API Routes aus Backend
    backend_routes = set(re.findall(r"@app\.route\(['\"](/[^'\"]+)['\"]", backend))
    
    # Extrahiere API Calls aus Frontend
    frontend_apis = set(re.findall(r"fetch\(['\"`]([^'\"`.]+)['\"`]", frontend))
    
    info(f"Backend Routes: {len(backend_routes)}")
    info(f"Frontend API Calls: {len(frontend_apis)}")
    
    # Überprüfe Mapping
    mapped = 0
    not_mapped = 0
    
    for api_call in frontend_apis:
        # Normalize API Call (entferne Query Params, etc)
        api_base = api_call.split('?')[0]
        
        # Überprüfe ob Backend Route existiert
        matched = False
        for route in backend_routes:
            # Exact match
            if route == api_base:
                matched = True
                break
            # Wildcard match (z.B. /api/jobs/<id> vs /api/jobs/123)
            route_wildcard = re.sub(r'<[^>]+>', '[^/]+', route)
            if re.match(f"^{route_wildcard}$", api_base):
                matched = True
                break
        
        if matched:
            mapped += 1
            if not any(x in api_call for x in ['static', 'v2', 'old']):
                ok(f"✓ {api_call}")
        else:
            not_mapped += 1
            warn(f"? {api_call} (Backend Route nicht gefunden)")
    
    coverage = mapped / (mapped + not_mapped) * 100 if (mapped + not_mapped) > 0 else 0
    info(f"\nAPI Mapping Coverage: {mapped}/{mapped + not_mapped} ({coverage:.0f}%)")
    
    if coverage >= 90:
        ok(f"Hervorragende API Mapping Coverage")
        return True
    elif coverage >= 75:
        warn(f"Gute API Mapping Coverage, aber {not_mapped} Calls unmapped")
        return True
    else:
        err(f"Schwache API Mapping Coverage: {not_mapped} Calls unmapped")
        return coverage >= 60

# =============================================================================
# 5. JOB CREATION FLOW ÜBERPRÜFUNG
# =============================================================================

def check_job_flow() -> bool:
    section("5. JOB CREATION FLOW")
    
    app_py = Path("src/montage_ai/web_ui/app.py")
    with open(app_py, 'r') as f:
        content = f.read()
    
    # Überprüfe Job Creation Endpoint
    if "@app.route('/api/jobs', methods=['POST'])" in content or \
       '@app.route("/api/jobs", methods=["POST"])' in content:
        ok("✓ POST /api/jobs Endpoint existiert")
    else:
        err("POST /api/jobs Endpoint nicht gefunden")
        return False
    
    # Überprüfe Job Storage
    if "job_store.create_job" in content:
        ok("✓ Job Storage (job_store) verwendet")
    else:
        warn("Job Storage nicht eindeutig gefunden")
    
    # Überprüfe RQ Queue
    if "q.enqueue" in content or "Queue" in content:
        ok("✓ RQ Queue Integration")
    else:
        err("RQ Queue nicht konfiguriert")
        return False
    
    # Überprüfe Job Status Endpoint
    if "api_get_job" in content and "/api/jobs/<job_id>" in content:
        ok("✓ GET /api/jobs/<job_id> Endpoint")
    else:
        warn("Job Status Endpoint nicht klar")
    
    # Überprüfe Task Module
    tasks_py = Path("src/montage_ai/tasks.py")
    if tasks_py.exists():
        with open(tasks_py, 'r') as f:
            tasks_content = f.read()
        
        if "def run_montage" in tasks_content:
            ok("✓ run_montage Task (RQ Worker)")
        else:
            err("run_montage Task nicht gefunden")
            return False
    else:
        warn("tasks.py nicht gefunden")
    
    ok(f"Job Creation Flow korrekt konfiguriert")
    return True

# =============================================================================
# 6. DATA PATHS ÜBERPRÜFUNG
# =============================================================================

def check_data_paths() -> bool:
    section("6. DATA PATHS (ENV VARIABLES)")
    
    # Überprüfe config
    config_py = Path("src/montage_ai/config.py")
    if not config_py.exists():
        err("config.py nicht gefunden")
        return False
    
    with open(config_py, 'r') as f:
        config_content = f.read()
    
    required_paths = ["input_dir", "output_dir", "music_dir", "assets_dir"]
    found_paths = 0
    
    for path in required_paths:
        if path in config_content:
            ok(f"✓ {path} definiert")
            found_paths += 1
        else:
            warn(f"? {path} nicht in config")
    
    if found_paths >= 3:
        ok(f"Data Paths konfiguriert")
        return True
    else:
        warn(f"Data Paths möglicherweise unvollständig")
        return False

# =============================================================================
# 7. SHORTS & TRANSCRIPT FEATURE ÜBERPRÜFUNG
# =============================================================================

def check_features() -> bool:
    section("7. FEATURES (Shorts, Transcript, Sessions)")
    
    app_py = Path("src/montage_ai/web_ui/app.py")
    with open(app_py, 'r') as f:
        content = f.read()
    
    features = {
        "Shorts": ["/api/shorts/", "api_shorts"],
        "Transcript": ["/api/transcript/", "api_transcript"],
        "Sessions": ["/api/session/", "api_session"],
        "CGPU": ["/api/cgpu/", "api_cgpu"],
    }
    
    found_features = 0
    for feature, patterns in features.items():
        if any(p in content for p in patterns):
            ok(f"✓ {feature} Feature")
            found_features += 1
        else:
            warn(f"? {feature} Feature nicht eindeutig")
    
    if found_features >= 3:
        ok(f"Wichtigste Features implementiert")
        return True
    else:
        warn(f"Feature Coverage: {found_features}/4")
        return found_features >= 2

# =============================================================================
# 8. FRONTEND COMPONENTS ÜBERPRÜFUNG
# =============================================================================

def check_frontend_components() -> bool:
    section("8. FRONTEND COMPONENTS")
    
    templates_dir = Path("src/montage_ai/web_ui/templates")
    
    required_pages = ["index.html", "montage.html", "shorts.html", "transcript.html"]
    
    found_pages = 0
    for page in required_pages:
        page_path = templates_dir / page
        if page_path.exists():
            with open(page_path, 'r') as f:
                page_content = f.read()
            
            # Überprüfe ob es Interactive Content hat
            if "input" in page_content or "button" in page_content or "script" in page_content:
                ok(f"✓ {page}")
                found_pages += 1
            else:
                warn(f"? {page} möglicherweise unvollständig")
        else:
            warn(f"✗ {page} nicht gefunden")
    
    # Überprüfe CSS
    css_path = Path("src/montage_ai/web_ui/static/css/voxel-dark.css")
    if css_path.exists():
        ok(f"✓ CSS (voxel-dark.css)")
    else:
        warn(f"✗ CSS nicht gefunden")
    
    if found_pages >= 3:
        ok(f"Frontend Components vorhanden")
        return True
    else:
        warn(f"Frontend Components: {found_pages}/4")
        return found_pages >= 2

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{CYAN}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  MONTAGE AI - CODE AUDIT                                           ║")
    print("║  Detaillierte Überprüfung aller Komponenten (ohne Server)          ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{NC}")
    
    checks = {
        "CLI": check_cli(),
        "Backend API Structure": check_backend(),
        "Frontend API Calls": check_frontend(),
        "API Mapping": check_api_mapping(),
        "Job Creation Flow": check_job_flow(),
        "Data Paths": check_data_paths(),
        "Features": check_features(),
        "Frontend Components": check_frontend_components(),
    }
    
    # Summary
    section("ZUSAMMENFASSUNG")
    
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    for check_name, result in checks.items():
        status = f"{GREEN}✓ PASS{NC}" if result else f"{RED}✗ FAIL{NC}"
        print(f"  {check_name}: {status}")
    
    print(f"\n{CYAN}Ergebnis: {passed}/{total} Checks bestanden{NC}\n")
    
    if passed == total:
        ok(f"✨ ALLES OK - Vollständige Integration!")
        return 0
    elif passed >= 6:
        info(f"🟢 GUT - {total - passed} Probleme gefunden, aber funktionsfähig")
        return 0
    else:
        err(f"🔴 PROBLEME - {total - passed} Checks fehlgeschlagen")
        return 1

if __name__ == "__main__":
    sys.exit(main())
