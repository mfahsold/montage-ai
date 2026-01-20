import os
import sys
import subprocess
import re
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from montage_ai.core.job_store import JobStore
from montage_ai.config import get_settings
from montage_ai.logger import logger
from montage_ai.core.workflow import WorkflowOptions
from montage_ai.core.montage_workflow import MontageWorkflow
from montage_ai.core.shorts_workflow import ShortsWorkflow
from montage_ai import telemetry

# We need to know paths.
settings = get_settings()
INPUT_DIR = settings.paths.input_dir
MUSIC_DIR = settings.paths.music_dir
OUTPUT_DIR = settings.paths.output_dir
ASSETS_DIR = settings.paths.assets_dir

def _resolve_input_files(input_dir: Path, filenames: list[str]) -> list[str]:
    """Resolve user-selected filenames to safe absolute paths."""
    resolved: list[str] = []
    input_root = input_dir.resolve()

    for name in filenames:
        safe_name = os.path.basename(str(name))
        if not safe_name:
            continue
        candidate = (input_root / safe_name).resolve()
        if not str(candidate).startswith(str(input_root)):
            logger.warning("Ignoring unsafe input path: %s", name)
            continue
        if not candidate.exists():
            logger.warning("Input file not found: %s", safe_name)
            continue
        resolved.append(str(candidate))

    return resolved

def run_montage(job_id: str, style: str, options: dict):
    """Run montage creation in background (RQ worker)."""
    # Ensure directories
    settings.paths.ensure_directories()

    # Setup job logging - write to /tmp to avoid permission issues with /data/output PVC
    log_dir = Path("/tmp/montage_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"render_{job_id}.log"
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Start telemetry tracking
    with telemetry.job_context(job_id, "montage") as t:
        try:
            # Prepare options
            t.phase_start("setup")
            editing_instructions = options.get('editing_instructions', {})
            prompt = options.get('prompt')

            if prompt and not editing_instructions:
                try:
                    from montage_ai.creative_director import interpret_natural_language
                    logger.info(f"Interpreting creative prompt: {prompt}")
                    editing_instructions = interpret_natural_language(prompt)
                except Exception as e:
                    logger.warning(f"Failed to interpret prompt: {e}")
            t.phase_end("setup")

            selected_files = _resolve_input_files(INPUT_DIR, options.get("video_files") or [])

            workflow_options = WorkflowOptions(
                input_path=str(INPUT_DIR),
                output_dir=str(OUTPUT_DIR),
                job_id=job_id,
                quality_profile=options.get('quality_profile', 'standard'),
                stabilize=options.get('stabilize', False),
                upscale=options.get('upscale', False),
                enhance=options.get('enhance', False),
                extras={
                    "style": style,
                    "variant_id": options.get('variant_id', 1),
                    "editing_instructions": editing_instructions,
                    "beat_sync": options.get('beat_sync', True),
                    # Audio
                    "music_track": options.get('music_track'),
                    "music_start": options.get('music_start', 0),
                    "music_end": options.get('music_end'),
                    # Color Grading
                    "color_grading": options.get('color_grading'),
                    "color_intensity": options.get('color_intensity'),
                    # Enhancements
                    "denoise": options.get('denoise', False),
                    "sharpen": options.get('sharpen', False),
                    "film_grain": options.get('film_grain'),
                    # Misc
                    "target_duration": options.get('target_duration'),
                    "video_files": selected_files,
                }
            )

            # Run workflow with telemetry phases
            workflow = MontageWorkflow(workflow_options)

            t.phase_start("analyzing")
            # Note: workflow.execute() handles all phases internally
            # We track overall execution here
            t.phase_end("analyzing")

            t.phase_start("rendering")
            workflow.execute()
            t.phase_end("rendering")

            t.record_success()

        except Exception as e:
            logger.error(f"Critical error in run_montage: {e}")
            store = JobStore()
            store.update_job(job_id, {"status": "failed", "error": str(e)})
            t.record_failure(str(e))

        finally:
            # Teardown logging
            logger.removeHandler(file_handler)
            file_handler.close()

def run_transcript_render(_job_data: dict):
    """Run transcript render in background."""
    # Similar to run_montage but for transcript
    # For now, just a stub or copy logic if needed
    pass


def run_test_job(job_id: str, duration: int = 5):
    """Lightweight dev-only job that sleeps for `duration` seconds and updates job status.

    - Intended for dev/CI only. Does not run the full montage pipeline.
    - Keeps the same job lifecycle semantics so callers can poll /api/jobs/<id>.
    """
    store = JobStore()

    # Coerce/validate duration (defensive)
    try:
        duration = max(0, int(duration))
    except Exception:
        duration = 5

    with telemetry.job_context(job_id, "dev-test"):
        try:
            logger.info("[DevTestJob] Starting lightweight test job %s (duration=%ss)", job_id, duration)
            store.update_job(job_id, {"status": "started", "phase": {"name": "running", "label": "Running test job"}})

            # Sleep in small increments so the job can be more responsive to interrupts
            remaining = duration
            while remaining > 0:
                step = min(1, remaining)
                try:
                    time.sleep(step)
                except KeyboardInterrupt:
                    # Respect cancellations in interactive/dev runs
                    logger.info("[DevTestJob] Interrupted %s", job_id)
                    store.update_job(job_id, {"status": "failed", "error": "interrupted"})
                    return
                remaining -= step

            store.update_job(job_id, {"status": "finished", "phase": {"name": "finished", "label": "Completed test job"}})
            logger.info("[DevTestJob] Completed %s", job_id)
            try:
                telemetry.record_event("dev_test_job", {"status": "success", "job_id": job_id})
            except Exception:
                pass

        except Exception as exc:  # noqa: BLE001
            logger.exception("[DevTestJob] Failed %s: %s", job_id, exc)
            store.update_job(job_id, {"status": "failed", "error": str(exc)})
            try:
                telemetry.record_event("dev_test_job", {"status": "failure", "job_id": job_id, "error": str(exc)[:200]})
            except Exception:
                pass


def run_shorts_reframe(job_id: str, options: dict):
    """Run shorts reframing in background (RQ worker)."""
    # Ensure directories
    settings.paths.ensure_directories()

    # Setup job logging
    log_path = Path(OUTPUT_DIR) / f"shorts_{job_id}.log"
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Start telemetry tracking
    with telemetry.job_context(job_id, "shorts") as t:
        try:
            t.phase_start("setup")
            workflow_options = WorkflowOptions(
                input_path=options.get('video_path'),
                output_dir=str(OUTPUT_DIR),
                job_id=job_id,
                quality_profile='standard',
                extras={
                    "reframe_mode": options.get('reframe_mode', 'auto'),
                    "add_captions": options.get('add_captions', True),
                    "caption_style": options.get('caption_style', 'tiktok'),
                    "clean_audio": options.get('clean_audio', False),
                    "audio_aware": options.get('audio_aware', False),
                    "platform": options.get('platform', 'tiktok')
                }
            )
            t.phase_end("setup")

            # Run workflow
            t.phase_start("rendering")
            workflow = ShortsWorkflow(workflow_options)
            workflow.execute()
            t.phase_end("rendering")

            t.record_success()

        except Exception as e:
            logger.error(f"Critical error in run_shorts_reframe: {e}")
            store = JobStore()
            store.update_job(job_id, {"status": "failed", "error": str(e)})
            t.record_failure(str(e))

        finally:
            # Teardown logging
            logger.removeHandler(file_handler)
            file_handler.close()

