import os
import sys
import subprocess
import re
import json
from datetime import datetime
from pathlib import Path
from src.montage_ai.core.job_store import JobStore
from src.montage_ai.env_mapper import map_options_to_env
from src.montage_ai.config import get_settings

# We need to know paths.
settings = get_settings()
INPUT_DIR = settings.paths.input_dir
MUSIC_DIR = settings.paths.music_dir
OUTPUT_DIR = settings.paths.output_dir
ASSETS_DIR = settings.paths.assets_dir

def run_montage(job_id: str, style: str, options: dict):
    """Run montage creation in background (RQ worker)."""
    store = JobStore()
    
    # Ensure directories
    settings.paths.ensure_directories()
    
    # Update status
    store.update_job(job_id, {"status": "running", "started_at": datetime.now().isoformat()})
    
    try:
        cmd = [sys.executable, "-m", "montage_ai.editor"]
        env = map_options_to_env(style, options, job_id)
        env["INPUT_DIR"] = str(INPUT_DIR)
        env["MUSIC_DIR"] = str(MUSIC_DIR)
        env["OUTPUT_DIR"] = str(OUTPUT_DIR)
        env["ASSETS_DIR"] = str(ASSETS_DIR)
        
        log_path = Path(OUTPUT_DIR) / f"render_{job_id}.log"
        
        def set_low_priority():
            try:
                os.nice(10)
            except Exception:
                pass

        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=set_low_priority
            )
            
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                
                if "Phase" in line and "/" in line:
                    try:
                        match = re.search(r'Phase\s*(\d+)/(\d+):\s*(\w+)', line)
                        if match:
                            phase_num = int(match.group(1))
                            phase_total = int(match.group(2))
                            phase_name = match.group(3).lower()
                            phase_data = {
                                "name": phase_name,
                                "label": f"Phase {phase_num}/{phase_total}: {match.group(3)}",
                                "number": phase_num,
                                "total": phase_total,
                                "started_at": datetime.now().isoformat(),
                                "progress_percent": int((phase_num / phase_total) * 100)
                            }
                            store.update_job(job_id, {"phase": phase_data})
                    except Exception:
                        pass
            
            process.wait()
            result = process
            
        # Check success
        output_pattern = f"*{job_id}*.mp4"
        output_files = list(Path(OUTPUT_DIR).glob(output_pattern))
        
        if result.returncode == 0 or output_files:
            updates = {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "progress_percent": 100
            }
            if output_files:
                updates["output_file"] = str(output_files[0].name)
            
            store.update_job(job_id, updates)
        else:
            store.update_job(job_id, {"status": "failed", "error": f"Process exited with code {result.returncode}"})

    except Exception as e:
        store.update_job(job_id, {"status": "failed", "error": str(e)})

def run_transcript_render(job_data: dict):
    """Run transcript render in background."""
    # Similar to run_montage but for transcript
    # For now, just a stub or copy logic if needed
    pass
