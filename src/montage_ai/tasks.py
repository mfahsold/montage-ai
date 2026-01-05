import os
import sys
import subprocess
import re
import json
from datetime import datetime
from pathlib import Path
from montage_ai.core.job_store import JobStore
from montage_ai.env_mapper import map_options_to_env
from montage_ai.config import get_settings
from montage_ai.logger import logger

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


def run_shorts_reframe(job_id: str, options: dict):
    """
    Run Shorts Studio reframing in background (RQ worker).
    
    Consolidates Shorts workflow into job queue for:
    - Consistent progress tracking
    - Async processing (no blocking Flask endpoints)
    - Cancel support
    - Same infrastructure as Creator workflow
    """
    from .auto_reframe import AutoReframeEngine
    from .transcriber import transcribe_audio
    from .caption_burner import CaptionBurner, CaptionStyle
    
    store = JobStore()
    
    # Update status
    store.update_job(job_id, {"status": "running", "started_at": datetime.now().isoformat()})
    
    try:
        video_path = options.get('video_path')
        reframe_mode = options.get('reframe_mode', 'auto')
        caption_style = options.get('caption_style', 'tiktok')
        add_captions = options.get('add_captions', True)
        clean_audio = options.get('clean_audio', False)
        platform = options.get('platform', 'tiktok')
        
        if not video_path or not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Phase 1: Audio Cleaning (if requested)
        if clean_audio:
            store.update_job(job_id, {
                "phase": {
                    "name": "audio_cleaning",
                    "label": "Cleaning Audio",
                    "number": 1,
                    "total": 4 if add_captions else 3
                },
                "progress_percent": 10
            })
            # TODO: Apply clean audio (voice isolation + noise reduction)
            # video_path = _apply_clean_audio(video_path)
        
        # Phase 2: Analysis & Reframing
        # Optionally analyze audio for beat-aware cutting (shared with Creator)
        audio_beats = None
        if options.get('audio_aware', False):
            try:
                from .audio_analysis import get_beat_times
                logger.info("Analyzing audio for beat-aware cuts...")
                audio_beats = get_beat_times(str(video_path))
                logger.debug(f"Detected {audio_beats.beat_count} beats @ {audio_beats.tempo:.1f} BPM")
            except Exception as e:
                logger.warning(f"Audio analysis failed: {e}, continuing without audio awareness")
        
        store.update_job(job_id, {
            "phase": {
                "name": "reframing",
                "label": "Analyzing & Reframing",
                "number": 2 if clean_audio else 1,
                "total": 4 if add_captions else 3
            },
            "progress_percent": 30
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reframed_path = OUTPUT_DIR / f"shorts_reframed_{timestamp}.mp4"
        output_path = OUTPUT_DIR / f"shorts_{timestamp}.mp4"
        
        reframer = AutoReframeEngine(target_aspect=9/16)
        
        # Analyze if needed
        if reframe_mode != 'center':
            crop_data = reframer.analyze(video_path)
        else:
            crop_data = None
        
        # Apply reframing
        reframer.apply(crop_data, video_path, str(reframed_path))
        
        # Phase 3: Transcription (if captions requested)
        if add_captions:
            store.update_job(job_id, {
                "phase": {
                    "name": "transcription",
                    "label": "Transcribing Audio",
                    "number": 3,
                    "total": 4
                },
                "progress_percent": 60
            })
            
            transcript = transcribe_audio(str(reframed_path), model='base', word_timestamps=True)
            
            # Phase 4: Caption Burning
            store.update_job(job_id, {
                "phase": {
                    "name": "captioning",
                    "label": "Burning Captions",
                    "number": 4,
                    "total": 4
                },
                "progress_percent": 80
            })
            
            # Map caption style
            style_map = {
                'default': CaptionStyle.TIKTOK,
                'tiktok': CaptionStyle.TIKTOK,
                'bold': CaptionStyle.BOLD,
                'minimal': CaptionStyle.MINIMAL,
                'gradient': CaptionStyle.KARAOKE,
                'karaoke': CaptionStyle.KARAOKE,
            }
            burner_style = style_map.get(caption_style, CaptionStyle.TIKTOK)
            
            # Generate SRT
            srt_path = reframed_path.with_suffix('.srt')
            _generate_srt_from_transcript(transcript, str(srt_path))
            
            # Burn captions
            burner = CaptionBurner(style=burner_style)
            burner.burn(str(reframed_path), str(srt_path), str(output_path))
            
            # Cleanup
            reframed_path.unlink(missing_ok=True)
            srt_path.unlink(missing_ok=True)
        else:
            # No captions - just rename
            reframed_path.rename(output_path)
        
        # Success
        store.update_job(job_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "progress_percent": 100,
            "output_file": output_path.name,
            "path": str(output_path)
        })
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Shorts reframe failed: {error_msg}")
        logger.error(traceback.format_exc())
        store.update_job(job_id, {"status": "failed", "error": error_msg})


def _generate_srt_from_transcript(transcript: dict, srt_path: str):
    """Generate SRT file from Whisper transcript."""
    segments = transcript.get('segments', [])
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()
            
            # Format timecodes: HH:MM:SS,mmm
            def tc(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                ms = int((seconds - int(seconds)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
            f.write(f"{i}\n")
            f.write(f"{tc(start)} --> {tc(end)}\n")
            f.write(f"{text}\n\n")
