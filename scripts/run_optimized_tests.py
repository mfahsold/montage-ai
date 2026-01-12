#!/usr/bin/env python3
"""
Optimized Test Suite for Montage-AI

Based on performance analysis from previous runs:
- 65-min 4K video causes proxy timeout (>1h)
- Scene detection is CPU-bound bottleneck
- Memory usage is stable (~100-250MB)
- Initialization is fast (~5-10s)

Test Categories:
1. Quick Smoke Test: Small videos, preview quality
2. Standard Test: Medium videos, standard quality
3. Stress Test: Large videos, high quality
4. Cluster Test: Distributed processing
"""

import argparse
import os
import subprocess
import sys
import time
import threading
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from subprocess import Popen, PIPE, TimeoutExpired

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_INPUT = PROJECT_ROOT / "data" / "input"
DATA_OUTPUT = PROJECT_ROOT / "data" / "output"


@dataclass
class TestConfig:
    """Test configuration."""
    name: str
    duration: int  # Target duration in seconds
    quality: str  # preview, standard, high
    cut_style: str
    max_input_duration: int  # Skip videos longer than this (seconds)
    timeout: int  # Timeout in seconds
    extra_env: dict = None

    def to_env(self) -> dict:
        """Convert to environment variables."""
        env = {
            "TARGET_DURATION": str(self.duration),
            "QUALITY_PROFILE": self.quality,
            "CUT_STYLE": self.cut_style,
            "EXPORT_TIMELINE": "true",
            "PYTHONUNBUFFERED": "1",
        }
        if self.extra_env:
            env.update(self.extra_env)
        return env


# Test configurations based on performance data
TESTS = {
    "smoke": TestConfig(
        name="Quick Smoke Test",
        duration=10,
        quality="preview",
        cut_style="hitchcock",
        max_input_duration=600,  # Skip videos > 10 min
        timeout=300,  # 5 min timeout
        extra_env={"LOW_MEMORY_MODE": "true", "PROXY_HEIGHT": "480"},
    ),
    "standard": TestConfig(
        name="Standard Quality Test",
        duration=30,
        quality="standard",
        cut_style="hitchcock",
        max_input_duration=1800,  # Skip videos > 30 min
        timeout=1200,  # 20 min timeout
    ),
    "4k-quick": TestConfig(
        name="4K Quick Test (5min clip)",
        duration=15,
        quality="standard",
        cut_style="documentary",
        max_input_duration=7200,  # Allow long videos
        timeout=1800,  # 30 min timeout
        extra_env={
            "MAX_INPUT_CLIP_DURATION": "300",  # Use max 5 min from each source
            "PROXY_HEIGHT": "720",
        },
    ),
    "stress": TestConfig(
        name="Stress Test (Full 4K)",
        duration=60,
        quality="high",
        cut_style="documentary",
        max_input_duration=7200,
        timeout=7200,  # 2h timeout
        extra_env={"PROXY_TIMEOUT": "7200"},  # 2h proxy timeout
    ),
    "parallel": TestConfig(
        name="Parallel Processing Test",
        duration=20,
        quality="preview",
        cut_style="mtv",
        max_input_duration=1200,
        timeout=600,
        extra_env={
            "MAX_PARALLEL_JOBS": "4",
            "MAX_CONCURRENT_JOBS": "4",
        },
    ),
}


def get_eligible_videos(max_duration: int) -> List[Path]:
    """Get videos that are under the duration limit."""
    videos = []
    for video in DATA_INPUT.glob("*.mp4"):
        # Get duration using ffprobe
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video)
                ],
                capture_output=True, text=True, timeout=10
            )
            duration = float(result.stdout.strip())
            if duration <= max_duration:
                videos.append(video)
                print(f"  ✓ {video.name}: {duration/60:.1f}min")
            else:
                print(f"  ✗ {video.name}: {duration/60:.1f}min (skipped, > {max_duration/60:.0f}min)")
        except Exception as e:
            print(f"  ? {video.name}: Could not probe ({e})")
    return videos


def run_test(config: TestConfig, videos: List[Path]) -> dict:
    """Run a single test configuration."""
    print(f"\n{'='*60}")
    print(f"🧪 {config.name}")
    print(f"{'='*60}")
    print(f"Duration: {config.duration}s | Quality: {config.quality}")
    print(f"Videos: {len(videos)} | Timeout: {config.timeout}s")
    print()

    if not videos:
        print("❌ No eligible videos found!")
        return {"status": "skipped", "reason": "no_videos"}

    # Create temporary input directory with symlinks
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / "input"
        tmp_input.mkdir()

        for video in videos:
            (tmp_input / video.name).symlink_to(video)

        # Setup environment
        env = os.environ.copy()
        env.update({
            "INPUT_DIR": str(tmp_input),
            "MUSIC_DIR": str(PROJECT_ROOT / "data" / "music"),
            "OUTPUT_DIR": str(DATA_OUTPUT),
            "ASSETS_DIR": str(PROJECT_ROOT / "data" / "assets"),
        })
        env.update(config.to_env())

        # Run montage
        start_time = time.time()
        cmd = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-m", "montage_ai"]

        # Lightweight monitoring thread (samples every 5s) to capture
        # CPU, memory and optional GPU usage while test is running.
        monitor_data = []
        monitor_stop = threading.Event()

        def monitor_loop():
            try:
                import psutil
            except Exception:
                print("⚠️ psutil not available; system monitoring disabled")
                return

            while not monitor_stop.is_set():
                ts = datetime.now(timezone.utc).isoformat()
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()._asdict()

                gpu = None
                try:
                    # Attempt to get nvidia-smi if present
                    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                    if smi.returncode == 0 and smi.stdout.strip():
                        parts = [p.strip() for p in smi.stdout.strip().split(',')]
                        gpu = {"util": parts[0], "mem_used": parts[1]} if len(parts) >= 2 else {"util": parts[0]}
                except Exception:
                    gpu = None

                monitor_data.append({"ts": ts, "cpu": cpu, "mem": mem, "gpu": gpu})
                monitor_stop.wait(5)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

        proc = None
        try:
            # Use Popen so we can handle signals and timeouts cleanly
            proc = Popen(cmd, env=env, stdout=PIPE, stderr=PIPE, text=True)
            try:
                stdout, stderr = proc.communicate(timeout=config.timeout)
                ret = proc.returncode
            except TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate(timeout=10)
                ret = proc.returncode
                elapsed = time.time() - start_time
                print(f"⏰ TIMEOUT after {elapsed:.1f}s")
                return {
                    "status": "timeout",
                    "elapsed_s": elapsed,
                    "timeout": config.timeout,
                    "stderr": stderr[-500:],
                }

            elapsed = time.time() - start_time
            if ret == 0:
                print(f"✅ SUCCESS in {elapsed:.1f}s")
                return {
                    "status": "success",
                    "elapsed_s": elapsed,
                    "videos": len(videos),
                }
            else:
                print(f"❌ FAILED (exit {ret}) after {elapsed:.1f}s")
                print(f"Stderr: {stderr[-500:]}")
                return {
                    "status": "failed",
                    "elapsed_s": elapsed,
                    "exit_code": ret,
                    "stderr": stderr[-500:],
                }

        except KeyboardInterrupt:
            # If the user interrupts, ensure we stop the child process and save monitoring
            if proc and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            elapsed = time.time() - start_time
            print(f"⏸️ Interrupted by user after {elapsed:.1f}s")
            return {
                "status": "interrupted",
                "elapsed_s": elapsed,
            }

        finally:
            # Stop monitor and dump data
            monitor_stop.set()
            monitor_thread.join(timeout=2)
            if monitor_data:
                outfile = DATA_OUTPUT / f"monitoring_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{config.name.replace(' ','_')}.json"
                try:
                    with open(outfile, 'w') as f:
                        json.dump(monitor_data, f, indent=2)
                    print(f"📈 Monitoring data saved to {outfile}")
                except Exception as e:
                    print(f"⚠️ Failed to write monitoring data: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run optimized montage-ai tests")
    parser.add_argument(
        "tests",
        nargs="*",
        default=["smoke"],
        help=f"Test(s) to run: {', '.join(list(TESTS.keys()) + ['all'])}"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available tests"
    )
    args = parser.parse_args()

    if args.list:
        print("\n📋 Available Tests:\n")
        for name, config in TESTS.items():
            print(f"  {name:12} - {config.name}")
            print(f"               Duration: {config.duration}s, Quality: {config.quality}")
            print(f"               Max input: {config.max_input_duration/60:.0f}min, Timeout: {config.timeout/60:.0f}min")
            print()
        return

    # Determine which tests to run
    if "all" in args.tests:
        test_names = list(TESTS.keys())
    else:
        test_names = args.tests

    print("\n🔬 MONTAGE-AI OPTIMIZED TEST SUITE")
    print("=" * 60)
    print(f"Tests: {', '.join(test_names)}")
    print()

    results = {}
    for name in test_names:
        config = TESTS[name]

        # Get eligible videos
        print(f"\n📁 Scanning videos (max {config.max_input_duration/60:.0f}min)...")
        videos = get_eligible_videos(config.max_input_duration)

        # Run test
        results[name] = run_test(config, videos)

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = result["status"]
        emoji = {"success": "✅", "failed": "❌", "timeout": "⏰", "skipped": "⏭️", "interrupted": "⏸️"}.get(status, "❓")
        elapsed = result.get("elapsed_s", 0)
        print(f"  {emoji} {name}: {status} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
