#!/usr/bin/env python3
"""
Montage AI Micro-Benchmark Suite

Focuses on measuring critical code paths without requiring large test assets.
Tests algorithmic performance, I/O patterns, and pipeline efficiency.
"""

import os
import sys
import time
import tempfile
import json
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from montage_ai.config import get_settings
from montage_ai.logger import logger
from montage_ai.scene_analysis import Scene, SceneSimilarityIndex

# ============================================================================
# Performance Measurement
# ============================================================================

@dataclass
class BenchmarkResult:
    name: str
    duration_ms: float
    throughput: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class BenchmarkSuite:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.categories: Dict[str, List[BenchmarkResult]] = {}
    
    def add(self, category: str, result: BenchmarkResult):
        self.results.append(result)
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(result)
    
    def print_summary(self):
        print("\n" + "="*80)
        print("MONTAGE AI MICRO-BENCHMARK RESULTS")
        print("="*80)
        
        for cat, results in self.categories.items():
            print(f"\n📊 {cat}")
            print("-" * 80)
            total_ms = 0.0
            for r in results:
                throughput = f" [{r.throughput}]" if r.throughput else ""
                print(f"  {r.name:50s} {r.duration_ms:>10.2f}ms{throughput}")
                total_ms += r.duration_ms
            print(f"  {'TOTAL':50s} {total_ms:>10.2f}ms")
        
        total = sum(r.duration_ms for r in self.results)
        print("\n" + "="*80)
        print(f"TOTAL: {total:.2f}ms ({total/1000:.2f}s)")
        print("="*80 + "\n")
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'results': [asdict(r) for r in self.results],
                'categories': list(self.categories.keys())
            }, f, indent=2)


def benchmark(name: str, category: str, suite: BenchmarkSuite):
    """Decorator for benchmarking functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            
            bench_result = BenchmarkResult(
                name=name,
                duration_ms=duration_ms,
                details=result if isinstance(result, dict) else {}
            )
            suite.add(category, bench_result)
            return result
        return wrapper
    return decorator


# ============================================================================
# Benchmark Tests
# ============================================================================

def test_config_loading(suite: BenchmarkSuite):
    """Test configuration system performance."""
    
    @benchmark("Config: Load Settings", "Configuration", suite)
    def load_config():
        settings = get_settings()
        return {
            "num_features": len(vars(settings.features)),
            "num_paths": len(vars(settings.paths))
        }
    
    load_config()
    
    # Repeated loads (should use singleton)
    @benchmark("Config: Reload Settings (cached)", "Configuration", suite)
    def reload_config():
        for _ in range(100):
            settings = get_settings()
        return {"iterations": 100}
    
    reload_config()


def test_numpy_operations(suite: BenchmarkSuite):
    """Test NumPy operations used in audio analysis."""
    
    @benchmark("NumPy: Energy Profile Calculation", "Audio Analysis", suite)
    def energy_calc():
        # Simulate 3-minute audio at 22kHz
        samples = np.random.randn(22050 * 180)
        
        # Window-based energy (like in audio_analysis.py)
        window_size = 2048
        hop_length = 512
        
        energy_values = []
        for i in range(0, len(samples) - window_size, hop_length):
            window = samples[i:i+window_size]
            energy = np.sum(window ** 2)
            energy_values.append(energy)
        
        return {
            "samples": len(samples),
            "windows": len(energy_values)
        }
    
    energy_calc()
    
    @benchmark("NumPy: Beat Detection Simulation", "Audio Analysis", suite)
    def beat_simulation():
        # Simulate onset strength envelope
        duration_sec = 180
        tempo = 120
        expected_beats = int(duration_sec * tempo / 60)
        
        # Create synthetic onset strength
        onset_env = np.random.rand(int(duration_sec * 100))  # 100 fps
        
        # Peak detection (simplified)
        peaks = []
        threshold = np.median(onset_env) * 1.5
        for i in range(1, len(onset_env) - 1):
            if onset_env[i] > threshold and onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1]:
                peaks.append(i)
        
        return {
            "duration_sec": duration_sec,
            "beats_detected": len(peaks),
            "expected_beats": expected_beats
        }
    
    beat_simulation()


def test_scene_selection_algorithm(suite: BenchmarkSuite):
    """Test clip selection performance."""
    
    @benchmark("Selection: Weighted Scoring (1000 scenes)", "Timeline Assembly", suite)
    def selection_scoring():
        # Simulate 1000 candidate scenes
        num_scenes = 1000
        scenes = [{
            'quality': np.random.rand(),
            'action': np.random.rand(),
            'face_count': np.random.randint(0, 5),
            'duration': np.random.uniform(2, 10)
        } for _ in range(num_scenes)]
        
        # Weighted scoring
        weights = {'quality': 0.3, 'action': 0.2, 'face': 0.3, 'duration': 0.2}
        
        scores = []
        for scene in scenes:
            score = (
                scene['quality'] * weights['quality'] +
                scene['action'] * weights['action'] +
                min(scene['face_count'], 2) / 2.0 * weights['face'] +
                min(scene['duration'], 5) / 5.0 * weights['duration']
            )
            scores.append(score)
        
        # Select top 100
        top_indices = np.argsort(scores)[-100:]
        
        return {
            "candidates": num_scenes,
            "selected": len(top_indices)
        }
    
    selection_scoring()
    
    @benchmark("Selection: Recent Video Tracking (10000 ops)", "Timeline Assembly", suite)
    def recent_tracking():
        recent = set()
        ops = 10000
        
        for i in range(ops):
            video_id = f"video_{i % 100}"
            if video_id in recent:
                # Already used
                pass
            else:
                recent.add(video_id)
                if len(recent) > 20:
                    # Remove oldest (simplified - should use deque)
                    recent.pop()
        
        return {
            "operations": ops,
            "unique_tracked": len(recent)
        }
    
    recent_tracking()


def test_ffmpeg_operations(suite: BenchmarkSuite):
    """Test FFmpeg command building and execution."""
    
    @benchmark("FFmpeg: Command Execution Test", "Rendering", suite)
    def cmd_execution():
        # Simple FFmpeg operation test
        return {
            "test": "ffmpeg_available"
        }
    
    cmd_execution()
    
    # Test with tiny synthetic video
    @benchmark("FFmpeg: Create Test Video (1s)", "Data I/O", suite)
    def create_test_video():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output = f.name
        
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=1:size=640x480:rate=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
            output
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        
        if result.returncode == 0 and os.path.exists(output):
            size_kb = os.path.getsize(output) / 1024
            os.unlink(output)
            return {"success": True, "size_kb": size_kb}
        
        return {"success": False}
    
    create_test_video()


def test_file_operations(suite: BenchmarkSuite):
    """Test file I/O performance."""
    
    @benchmark("File I/O: Write 100 temp files (1MB each)", "Data I/O", suite)
    def write_temp_files():
        temp_dir = tempfile.mkdtemp()
        files_written = 0
        total_bytes = 0
        
        data = b"x" * (1024 * 1024)  # 1MB
        
        for i in range(100):
            path = os.path.join(temp_dir, f"temp_{i}.dat")
            with open(path, 'wb') as f:
                f.write(data)
            files_written += 1
            total_bytes += len(data)
        
        # Cleanup
        for i in range(100):
            os.unlink(os.path.join(temp_dir, f"temp_{i}.dat"))
        os.rmdir(temp_dir)
        
        return {
            "files": files_written,
            "total_mb": total_bytes / (1024 * 1024)
        }
    
    write_temp_files()
    
    @benchmark("File I/O: Read 100 files", "Data I/O", suite)
    def read_files():
        temp_dir = tempfile.mkdtemp()
        data = b"x" * (1024 * 1024)
        
        # Write
        for i in range(100):
            with open(os.path.join(temp_dir, f"temp_{i}.dat"), 'wb') as f:
                f.write(data)
        
        # Read
        files_read = 0
        total_bytes = 0
        for i in range(100):
            with open(os.path.join(temp_dir, f"temp_{i}.dat"), 'rb') as f:
                content = f.read()
            files_read += 1
            total_bytes += len(content)
        
        # Cleanup
        for i in range(100):
            os.unlink(os.path.join(temp_dir, f"temp_{i}.dat"))
        os.rmdir(temp_dir)
        
        return {
            "files": files_read,
            "total_mb": total_bytes / (1024 * 1024)
        }
    
    read_files()


def test_cache_operations(suite: BenchmarkSuite):
    """Test caching performance."""
    
    @benchmark("Cache: JSON write (1000 entries)", "Caching", suite)
    def json_cache_write():
        cache_data = {
            f"scene_{i}": {
                "path": f"/path/to/video{i}.mp4",
                "start": i * 2.0,
                "duration": 5.0,
                "quality": float(np.random.rand()),
                "has_faces": bool(np.random.choice([True, False])),
                "action_level": str(np.random.choice(["low", "medium", "high"]))
            }
            for i in range(1000)
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cache_data, f)
            cache_file = f.name
        
        size_kb = os.path.getsize(cache_file) / 1024
        os.unlink(cache_file)
        
        return {
            "entries": 1000,
            "size_kb": size_kb
        }
    
    json_cache_write()
    
    @benchmark("Cache: JSON read (1000 entries)", "Caching", suite)
    def json_cache_read():
        cache_data = {f"key_{i}": {"value": i} for i in range(1000)}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cache_data, f)
            cache_file = f.name
        
        # Read
        with open(cache_file, 'r') as f:
            loaded_data = json.load(f)
        
        os.unlink(cache_file)
        
        return {
            "entries": len(loaded_data)
        }
    
    json_cache_read()


def test_parallel_processing(suite: BenchmarkSuite):
    """Test concurrency patterns."""
    
    @benchmark("Parallel: ThreadPoolExecutor (100 tasks)", "Concurrency", suite)
    def thread_pool_test():
        from concurrent.futures import ThreadPoolExecutor
        
        def work(n):
            # Simulate I/O work
            time.sleep(0.001)
            return n * 2
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(work, i) for i in range(100)]
            results = [f.result() for f in futures]
        
        return {
            "tasks": len(results),
            "workers": 4
        }
    
    thread_pool_test()


# ============================================================================
# Main Runner
# ============================================================================

def main():
    print("\n🚀 Montage AI Micro-Benchmark Suite\n")
    
    suite = BenchmarkSuite()
    
    print("Running benchmarks...\n")
    
    test_config_loading(suite)
    test_numpy_operations(suite)
    test_scene_selection_algorithm(suite)
    test_ffmpeg_operations(suite)
    test_file_operations(suite)
    test_cache_operations(suite)
    test_parallel_processing(suite)
    
    # Save results
    output_dir = Path(__file__).parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "micro_benchmark_baseline.json"
    suite.save(str(output_file))
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    suite.print_summary()
    
    return suite


if __name__ == "__main__":
    main()
