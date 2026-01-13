"""
Montage AI - Live Monitoring System

Provides detailed real-time logging for:
- Decision tracking (why certain clips/cuts were chosen)
- Data flow visualization (input → processing → output)
- Performance metrics (timing, memory, throughput)
- Analysis results (scene detection, energy mapping, beat sync)

Usage:
    from monitoring import Monitor
    monitor = Monitor(job_id="preview_20231201_120000")
    monitor.log_decision("clip_selection", "Selected clip A over B", {"reason": "higher_energy"})
    monitor.log_metric("processing_time", 2.5, "seconds")

Version: 1.0.0
"""

import os
import sys
import json
import time
import psutil
import io
import atexit
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from .logger import logger
from .config import get_settings


class LogLevel(Enum):
    DEBUG = "🔬"
    INFO = "ℹ️"
    DECISION = "🧠"
    DATA = "📊"
    METRIC = "📈"
    WARNING = "⚠️"
    ERROR = "❌"
    SUCCESS = "✅"


@dataclass
class MonitorEvent:
    """Single monitoring event with full context"""
    timestamp: str
    level: str
    category: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None


class Monitor:
    """
    Central monitoring hub for the video editing pipeline.
    
    Tracks decisions, data flows, metrics, and provides live console output.
    """
    
    def __init__(self, job_id: str, verbose: bool = True):
        self.job_id = job_id
        self.verbose = verbose
        self.start_time = time.time()
        self.events: List[MonitorEvent] = []
        self.metrics: Dict[str, List[float]] = {}
        self.decisions: List[Dict] = []
        self.phase = "init"
        self.clip_count = 0
        self.processed_count = 0
        self._log_file = None
        self._tee_enabled = False
        self._mem_thread = None
        # Use centralized monitoring settings (env-aware via Settings)
        self._mem_interval = float(get_settings().monitoring.mem_interval_sec)
        self._stop_event = threading.Event()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        
        # Performance tracking
        self.phase_times: Dict[str, float] = {}
        self.phase_start: Optional[float] = None

        # KPI tracking
        self.kpis: Dict[str, float] = {}
        self.feature_usage: Dict[str, bool] = {}

        self._setup_tee()
        self._print_header()
        self._start_mem_probe()
    
    def __del__(self):
        """Cleanup resources on object destruction (memory leak prevention)."""
        self.cleanup()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, _exc_tb):
        """Ensure cleanup on context manager exit."""
        self.cleanup()
        return False

    def _start_mem_probe(self):
        """Periodically log process and system memory to stdout (and tee)."""
        if self._mem_interval <= 0:
            return

        # Import here to avoid potential circular imports
        try:
            from .cgpu_utils import get_cgpu_metrics
        except ImportError:
            get_cgpu_metrics = lambda: None

        def _loop():
            proc = psutil.Process()
            while not self._stop_event.is_set():
                try:
                    rss = proc.memory_info().rss / (1024 * 1024)
                    vms = proc.memory_info().vms / (1024 * 1024)
                    virt = psutil.virtual_memory()
                    
                    msg = (f"[monitor] mem: rss={rss:.1f}Mi vms={vms:.1f}Mi "
                           f"sys_used={virt.used/1024/1024:.1f}Mi "
                           f"sys_free={virt.available/1024/1024:.1f}Mi "
                           f"sys_pct={virt.percent:.1f}%")
                    
                    # Add cgpu metrics if available
                    cgpu_stats = get_cgpu_metrics()
                    if cgpu_stats:
                        msg += f" | {cgpu_stats}"
                        
                    logger.debug(msg)
                except ImportError:
                    pass  # psutil not available
                except Exception as e:
                    logger.debug(f"Memory monitoring error: {e}")
                if self._stop_event.wait(self._mem_interval):
                    break

        self._mem_thread = threading.Thread(target=_loop, daemon=True)
        self._mem_thread.start()

    def _setup_tee(self):
        """
        Mirror stdout/stderr to a log file when LOG_FILE is set.

        Keeps full pod logs on the output PVC so they are retrievable
        even after the Job has finished.
        """
        cfg = get_settings().monitoring
        log_path = cfg.log_file or str(get_settings().paths.output_dir / "render.log")

        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self._log_file = open(log_path, "a", buffering=1, encoding="utf-8")

            class _Tee(io.TextIOBase):
                def __init__(self, *streams):
                    self.streams = list(streams)

                def write(self, data):
                    for s in list(self.streams):
                        try:
                            s.write(data)
                        except Exception:
                            try:
                                self.streams.remove(s)
                            except ValueError:
                                pass
                    return len(data)

                def flush(self):
                    for s in list(self.streams):
                        try:
                            s.flush()
                        except Exception:
                            try:
                                self.streams.remove(s)
                            except ValueError:
                                pass

            sys.stdout = _Tee(sys.stdout, self._log_file)
            sys.stderr = _Tee(sys.stderr, self._log_file)
            self._tee_enabled = True
            logger.info(f"[monitor] tee logging enabled → {log_path}")

            # Register cleanup on normal exit
            atexit.register(self.cleanup)
        except Exception as exc:
            logger.warning(f"[monitor] ⚠️ Could not enable tee logging: {exc}")
    
    def cleanup(self):
        """
        Cleanup resources (file handles, threads, restore stdout/stderr).
        
        Called by:
        - __del__() on object destruction
        - __exit__() on context manager exit
        - atexit on normal program termination
        
        Safe to call multiple times (idempotent).
        """
        try:
            self._stop_event.set()
            if self._mem_thread and self._mem_thread.is_alive():
                self._mem_thread.join(timeout=1)
        except RuntimeError as e:
            logger.debug(f"Thread cleanup error: {e}")
        except Exception as e:
            logger.debug(f"Stop event error: {e}")

        try:
            if self._tee_enabled:
                sys.stdout = self._orig_stdout
                sys.stderr = self._orig_stderr
                self._tee_enabled = False
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Stdout restore error: {e}")

        try:
            if self._log_file:
                self._log_file.flush()
                self._log_file.close()
                self._log_file = None
        except (IOError, OSError) as e:
            logger.debug(f"Log file cleanup error: {e}")
    
    def _print_header(self):
        """Print startup banner with job info"""
        logger.info("\n" + "=" * 70)
        logger.info(f"🎬 FLUXIBRI VIDEO EDITOR - LIVE MONITOR")
        logger.info("=" * 70)
        logger.info(f"   Job ID:     {self.job_id}")
        logger.info(f"   Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   PID:        {os.getpid()}")
        logger.info(f"   Verbose:    {self.verbose}")
        logger.info("=" * 70 + "\n")
    
    def _log(self, level: LogLevel, category: str, message: str, data: Dict = None, duration_ms: float = None):
        """Internal logging with formatted output"""
        event = MonitorEvent(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            level=level.name,
            category=category,
            message=message,
            data=data or {},
            duration_ms=duration_ms
        )
        self.events.append(event)
        
        if self.verbose or level in [LogLevel.DECISION, LogLevel.WARNING, LogLevel.ERROR, LogLevel.SUCCESS]:
            # Format output
            time_str = f"[{event.timestamp}]"
            icon = level.value
            cat_str = f"[{category}]".ljust(20)
            
            # Duration suffix
            dur_str = ""
            if duration_ms is not None:
                if duration_ms >= 1000:
                    dur_str = f" ({duration_ms/1000:.1f}s)"
                else:
                    dur_str = f" ({duration_ms:.0f}ms)"
            
            logger.info(f"{time_str} {icon} {cat_str} {message}{dur_str}")
            
            # Print data details if present
            if data and self.verbose:
                for key, value in data.items():
                    if isinstance(value, float):
                        logger.info(f"           └─ {key}: {value:.3f}")
                    elif isinstance(value, list) and len(value) <= 5:
                        logger.info(f"           └─ {key}: {value}")
                    elif isinstance(value, dict):
                        logger.info(f"           └─ {key}: {json.dumps(value, indent=2)[:100]}...")
                    else:
                        val_str = str(value)[:80]
                        logger.info(f"           └─ {key}: {val_str}")
    
    # ==================== PHASE TRACKING ====================
    
    def start_phase(self, phase_name: str):
        """Mark the start of a processing phase"""
        if self.phase_start is not None:
            # Close previous phase
            elapsed = (time.time() - self.phase_start) * 1000
            self.phase_times[self.phase] = elapsed
        
        self.phase = phase_name
        self.phase_start = time.time()
        
        logger.info(f"\n{'─' * 60}")
        logger.info(f"📍 PHASE: {phase_name.upper()}")
        logger.info(f"{'─' * 60}")
    
    def end_phase(self, summary: Dict = None):
        """Mark the end of current phase with optional summary"""
        if self.phase_start is None:
            return
        
        elapsed = (time.time() - self.phase_start) * 1000
        self.phase_times[self.phase] = elapsed
        
        logger.info(f"\n   ✓ Phase '{self.phase}' completed in {elapsed/1000:.1f}s")
        if summary:
            for key, value in summary.items():
                logger.info(f"      • {key}: {value}")
        
        self.phase_start = None
    
    # ==================== DECISION LOGGING ====================
    
    def log_decision(self, decision_type: str, choice: str, alternatives: List[str] = None, 
                     reason: str = None, scores: Dict[str, float] = None):
        """
        Log a creative or algorithmic decision with full context.
        
        Examples:
            monitor.log_decision(
                "clip_selection", 
                "VID_001.mp4 @ 2.5s",
                alternatives=["VID_002.mp4 @ 1.0s", "VID_003.mp4 @ 4.2s"],
                reason="highest_energy_match",
                scores={"VID_001": 0.85, "VID_002": 0.72, "VID_003": 0.65}
            )
        """
        decision = {
            "type": decision_type,
            "choice": choice,
            "alternatives": alternatives or [],
            "reason": reason,
            "scores": scores,
            "timestamp": datetime.now().isoformat()
        }
        self.decisions.append(decision)
        
        self._log(
            LogLevel.DECISION, 
            f"decision:{decision_type}", 
            f"→ {choice}",
            {"reason": reason, "scores": scores} if scores else {"reason": reason}
        )
    
    def log_cut_decision(self, cut_index: int, source_clip: str, start_time: float, 
                         duration: float, beat_aligned: bool, reason: str):
        """Log detailed cut/edit decision"""
        self._log(
            LogLevel.DECISION,
            "cut",
            f"Cut #{cut_index}: {os.path.basename(source_clip)} [{start_time:.2f}s → {start_time+duration:.2f}s]",
            {
                "duration": f"{duration:.2f}s",
                "beat_aligned": beat_aligned,
                "reason": reason
            }
        )
    
    # ==================== DATA FLOW LOGGING ====================
    
    def log_data_flow(self, stage: str, input_desc: str, output_desc: str, 
                      transform: str = None, details: Dict = None):
        """
        Log data transformation between pipeline stages.
        
        Example:
            monitor.log_data_flow(
                "scene_detection",
                input_desc="VID_001.mp4 (1920x1080, 30s)",
                output_desc="12 scenes detected",
                transform="ContentDetector(threshold=27)",
                details={"avg_scene_length": 2.5}
            )
        """
        self._log(
            LogLevel.DATA,
            f"flow:{stage}",
            f"{input_desc} → {output_desc}",
            {"transform": transform, **(details or {})}
        )
    
    def log_input_analysis(self, filename: str, width: int, height: int, 
                           duration: float, fps: float, codec: str = None):
        """Log video file analysis results"""
        self._log(
            LogLevel.DATA,
            "input_analysis",
            f"{os.path.basename(filename)}",
            {
                "resolution": f"{width}x{height}",
                "duration": f"{duration:.1f}s",
                "fps": fps,
                "codec": codec or "unknown"
            }
        )
    
    # ==================== METRIC LOGGING ====================
    
    def log_metric(self, name: str, value: float, unit: str = None):
        """Log a performance or quality metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        unit_str = f" {unit}" if unit else ""
        self._log(LogLevel.METRIC, "metric", f"{name}: {value:.3f}{unit_str}")
    
    def log_timing(self, operation: str, duration_ms: float):
        """Log operation timing"""
        self._log(
            LogLevel.METRIC,
            "timing",
            f"{operation}",
            duration_ms=duration_ms
        )

    # ==================== KPI TRACKING ====================

    def log_time_to_first_preview(self, duration_s: float):
        """
        Track Time-to-First-Preview KPI.

        This is a critical SLA metric: how long from job submission
        until the user sees their first preview frame.
        """
        self.kpis["time_to_first_preview_s"] = duration_s
        self._log(
            LogLevel.METRIC,
            "kpi:ttfp",
            f"Time-to-First-Preview: {duration_s:.2f}s",
            {"target_s": 30.0, "met": duration_s <= 30.0}
        )

    def log_feature_usage(self, features: Dict[str, bool]):
        """
        Track which features are enabled for analytics.

        Args:
            features: Dict of feature_name -> enabled (e.g., {"stabilize": True})
        """
        self.feature_usage.update(features)
        enabled = [k for k, v in features.items() if v]
        if enabled:
            self._log(
                LogLevel.INFO,
                "feature_usage",
                f"Enabled: {', '.join(enabled)}"
            )

    def log_render_kpi(self, input_duration_s: float, render_time_s: float):
        """
        Track render efficiency KPI (realtime factor).

        A factor < 1.0 means faster than realtime.
        """
        factor = render_time_s / input_duration_s if input_duration_s > 0 else 0
        self.kpis["render_realtime_factor"] = factor
        self._log(
            LogLevel.METRIC,
            "kpi:render",
            f"Render {input_duration_s:.1f}s video in {render_time_s:.1f}s ({factor:.2f}x realtime)"
        )
    
    def log_progress(self, current: int, total: int, item: str = "items"):
        """Log progress update"""
        pct = (current / total * 100) if total > 0 else 0
        bar_len = 20
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"\r   [{bar}] {current}/{total} {item} ({pct:.0f}%)", end="", flush=True)
        if current >= total:
            print()  # Newline when complete
    
    # ==================== ANALYSIS RESULTS ====================
    
    def log_scene_detection(self, video_path: str, scenes: List[Dict], threshold: float):
        """Log scene detection results"""
        self._log(
            LogLevel.DATA,
            "scene_detection",
            f"{os.path.basename(video_path)}: {len(scenes)} scenes",
            {
                "threshold": threshold,
                "scene_count": len(scenes),
                "avg_duration": sum(s.get('duration', 0) for s in scenes) / max(len(scenes), 1)
            }
        )
    
    def log_beat_analysis(self, audio_path: str, bpm: float, beat_count: int, 
                          energy_profile: str = None):
        """Log music/beat analysis results"""
        self._log(
            LogLevel.DATA,
            "beat_analysis",
            f"{os.path.basename(audio_path)}: {bpm:.0f} BPM, {beat_count} beats",
            {
                "bpm": bpm,
                "beat_count": beat_count,
                "energy_profile": energy_profile or "mixed"
            }
        )
    
    def log_energy_mapping(self, time_point: float, energy: float, 
                           suggested_cut_length: float, reason: str):
        """Log energy-based editing decision"""
        self._log(
            LogLevel.DECISION,
            "energy_map",
            f"@{time_point:.1f}s: energy={energy:.2f} → cut={suggested_cut_length:.2f}s",
            {"reason": reason}
        )
    
    # ==================== CLIP PROCESSING ====================
    
    def log_clip_start(self, clip_index: int, source: str, operation: str):
        """Log start of clip processing"""
        self.clip_count = max(self.clip_count, clip_index)
        self._log(
            LogLevel.INFO,
            "clip_process",
            f"[{clip_index}] {operation}: {os.path.basename(source)}"
        )
    
    def log_clip_complete(self, clip_index: int, output: str, duration_ms: float):
        """Log completion of clip processing"""
        self.processed_count += 1
        self._log(
            LogLevel.SUCCESS,
            "clip_complete",
            f"[{clip_index}] → {os.path.basename(output)}",
            duration_ms=duration_ms
        )
    
    # ==================== ASSEMBLING PHASE ====================
    
    def log_assembling_start(self, total_clips: int, target_duration: float, tempo: float):
        """Log start of assembling phase"""
        self._log(
            LogLevel.INFO,
            "assembling",
            f"Starting assembly: {total_clips} clips → {target_duration:.1f}s @ {tempo:.0f} BPM"
        )
    
    def log_cut_placed(self, cut_num: int, total_cuts: int, clip_name: str, 
                       start: float, duration: float, beat_idx: int, beats_per_cut: int,
                       energy: float, score: int, reason: str):
        """Log each cut placement with full context"""
        # Progress bar
        pct = (cut_num / total_cuts * 100) if total_cuts > 0 else 0
        bar_len = 15
        filled = int(bar_len * cut_num / total_cuts) if total_cuts > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        
        logger.info(f"\n   ┌─ Cut #{cut_num} [{bar}] {pct:.0f}%")
        logger.info(f"   │  📹 {os.path.basename(clip_name)}")
        logger.info(f"   │  ⏱️  {start:.2f}s → {start+duration:.2f}s ({duration:.2f}s)")
        logger.info(f"   │  🎵 Beat {beat_idx} + {beats_per_cut} beats")
        logger.info(f"   │  ⚡ Energy: {energy:.2f} | Score: {score}")
        logger.info(f"   └─ 💡 {reason}")
    
    def log_cut_summary(self, cut_num: int, total_expected: int, timeline_pos: float):
        """Quick progress update for cuts"""
        if cut_num % 5 == 0 or cut_num == total_expected:  # Every 5th cut
            logger.info(f"   📍 Progress: {cut_num} cuts placed, timeline @ {timeline_pos:.1f}s")
    
    def log_clip_selection(self, selected: str, candidates: List[Dict], 
                           winning_score: int, selection_reason: str):
        """Log clip selection decision with alternatives"""
        alt_list = [f"{c['name']}({c['score']})" for c in candidates[:3]]
        self._log(
            LogLevel.DECISION,
            "clip_selection",
            f"→ {selected} (score={winning_score})",
            {
                "alternatives": alt_list,
                "reason": selection_reason
            }
        )
    
    def log_transition_applied(self, transition_type: str, duration: float, reason: str):
        """Log transition decision"""
        logger.info(f"   │  🔄 Transition: {transition_type} ({duration:.2f}s) - {reason}")
    
    def log_enhancement_applied(self, _clip_num: int, enhancements: List[str], duration_ms: float):
        """Log applied enhancements"""
        enh_str = " + ".join(enhancements) if enhancements else "none"
        logger.info(f"   │  ✨ Enhancements: {enh_str} ({duration_ms:.0f}ms)")
    
    # ==================== RENDERING PHASE ====================
    
    def log_render_start(self, output_path: str, settings: Dict):
        """Log render start"""
        logger.info(f"\n{'─' * 60}")
        logger.info(f"🚀 RENDERING")
        logger.info(f"{'─' * 60}")
        logger.info(f"   Output: {os.path.basename(output_path)}")
        for key, value in settings.items():
            logger.info(f"   • {key}: {value}")
    
    def log_render_progress(self, frame: int, total_frames: int, fps: float):
        """Log render progress"""
        pct = (frame / total_frames * 100) if total_frames > 0 else 0
        bar_len = 30
        filled = int(bar_len * frame / total_frames) if total_frames > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        eta = ((total_frames - frame) / fps) if fps > 0 else 0
        
        print(f"\r   [{bar}] {pct:.0f}% | {frame}/{total_frames} frames | ETA: {eta:.0f}s", end="", flush=True)
        if frame >= total_frames:
            print()
    
    def log_render_complete(self, output_path: str, file_size_mb: float, duration_ms: float):
        """Log render completion"""
        self._log(
            LogLevel.SUCCESS,
            "render",
            f"✅ Complete: {os.path.basename(output_path)} ({file_size_mb:.1f}MB)",
            duration_ms=duration_ms
        )
    
    # ==================== STATUS METHODS ====================
    
    def log_info(self, category: str, message: str, data: Dict = None):
        """General info logging"""
        self._log(LogLevel.INFO, category, message, data)
    
    def log_warning(self, category: str, message: str, data: Dict = None):
        """Warning logging"""
        self._log(LogLevel.WARNING, category, message, data)
    
    def log_error(self, category: str, message: str, data: Dict = None):
        """Error logging"""
        self._log(LogLevel.ERROR, category, message, data)
    
    def log_success(self, category: str, message: str, data: Dict = None):
        """Success logging"""
        self._log(LogLevel.SUCCESS, category, message, data)
    
    # ==================== RESOURCE MONITORING ====================
    
    def log_resources(self):
        """Log current system resource usage"""
        try:
            process = psutil.Process()
            mem = process.memory_info()
            cpu = process.cpu_percent()
            
            # System memory
            sys_mem = psutil.virtual_memory()
            
            self._log(
                LogLevel.METRIC,
                "resources",
                f"Process: {mem.rss / 1024**3:.1f}GB RAM, {cpu:.0f}% CPU",
                {
                    "process_rss_gb": mem.rss / 1024**3,
                    "process_cpu_pct": cpu,
                    "system_mem_pct": sys_mem.percent,
                    "system_available_gb": sys_mem.available / 1024**3
                }
            )
        except Exception as e:
            self._log(LogLevel.WARNING, "resources", f"Could not read resources: {e}")
    
    # ==================== SUMMARY ====================
    
    def print_summary(self):
        """Print final job summary"""
        total_time = time.time() - self.start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("📋 JOB SUMMARY")
        logger.info("=" * 70)
        logger.info(f"   Job ID:         {self.job_id}")
        logger.info(f"   Total Duration: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"   Clips Processed: {self.processed_count}")
        logger.info(f"   Decisions Made: {len(self.decisions)}")
        logger.info(f"   Events Logged:  {len(self.events)}")
        
        # Phase breakdown
        if self.phase_times:
            logger.info(f"\n   ⏱️ Phase Timing:")
            for phase, ms in self.phase_times.items():
                pct = (ms / (total_time * 1000)) * 100
                logger.info(f"      • {phase}: {ms/1000:.1f}s ({pct:.0f}%)")
        
        # Metric summaries
        if self.metrics:
            logger.info(f"\n   📈 Metrics:")
            for name, values in self.metrics.items():
                avg = sum(values) / len(values)
                logger.info(f"      • {name}: avg={avg:.2f}, count={len(values)}")

        # KPI summaries
        if self.kpis:
            logger.info(f"\n   🎯 KPIs:")
            for name, value in self.kpis.items():
                logger.info(f"      • {name}: {value:.2f}")

        # Feature usage
        if self.feature_usage:
            enabled = [k for k, v in self.feature_usage.items() if v]
            logger.info(f"\n   🔧 Features: {', '.join(enabled) if enabled else 'none'}")

        logger.info("=" * 70 + "\n")
    
    def export_json(self, filepath: str):
        """Export all monitoring data as JSON"""
        data = {
            "job_id": self.job_id,
            "total_duration_s": time.time() - self.start_time,
            "phase_times_ms": self.phase_times,
            "kpis": self.kpis,
            "feature_usage": self.feature_usage,
            "decisions": self.decisions,
            "metrics": self.metrics,
            "events": [asdict(e) for e in self.events]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._log(LogLevel.SUCCESS, "export", f"Monitoring data exported to {filepath}")


# Global monitor instance (set by main script)
_monitor: Optional[Monitor] = None


def get_monitor() -> Optional[Monitor]:
    """Get the global monitor instance"""
    return _monitor


def init_monitor(job_id: str, verbose: bool = True) -> Monitor:
    """Initialize global monitor"""
    global _monitor
    _monitor = Monitor(job_id, verbose)
    return _monitor
