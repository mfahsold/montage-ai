#!/usr/bin/env python3
"""
Performance Baseline - Quick benchmarks for different configurations
Establishes baseline metrics for future performance optimization tracking
"""

import sys
import os
import time
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add repo to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

def run_command(cmd: List[str], timeout: int = 300) -> Tuple[bool, float, str]:
    """Run command and measure execution time."""
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        elapsed = time.time() - start
        success = result.returncode == 0
        return success, elapsed, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, timeout, "Command timed out"
    except Exception as e:
        return False, time.time() - start, str(e)

def benchmark_import() -> Dict[str, float]:
    """Benchmark import times for different modules."""
    print("📊 Benchmarking imports...")
    results = {}
    
    modules = [
        ("montage_ai", "Core package"),
        ("montage_ai.config", "Configuration"),
        ("montage_ai.core.montage_builder", "Montage builder"),
        ("montage_ai.broll_planner", "B-roll planner"),
    ]
    
    for module_name, desc in modules:
        times = []
        for _ in range(3):  # 3 runs, take average
            start = time.perf_counter()
            try:
                __import__(module_name)
            except ImportError:
                continue
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        if times:
            avg_time = sum(times) / len(times)
            results[f"{module_name}"] = round(avg_time, 2)
            print(f"  {desc}: {avg_time:.2f}ms")
    
    return results

def benchmark_config_load() -> Dict[str, float]:
    """Benchmark configuration loading."""
    print("📊 Benchmarking configuration loading...")
    results = {}
    
    from montage_ai.config import ProcessingSettings
    
    times = []
    for _ in range(5):  # 5 runs
        start = time.perf_counter()
        config = ProcessingSettings()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    avg_time = sum(times) / len(times)
    results["config_load"] = round(avg_time, 2)
    print(f"  Average: {avg_time:.2f}ms")
    
    return results

def benchmark_optional_features() -> Dict[str, bool]:
    """Check which optional features are available."""
    print("📊 Checking optional features...")
    features = {}
    
    optional_modules = {
        "mediapipe": "Face detection",
        "librosa": "Audio analysis",
        "scipy": "Signal processing",
        "flask": "Web UI",
        "redis": "Cache backend",
        "torch": "Torch available",
    }
    
    for module_name, desc in optional_modules.items():
        try:
            __import__(module_name)
            features[module_name] = True
            print(f"  ✓ {desc} ({module_name})")
        except ImportError:
            features[module_name] = False
            print(f"  ✗ {desc} ({module_name})")
    
    return features

def benchmark_test_suite() -> Dict[str, float]:
    """Benchmark test suite execution."""
    print("📊 Benchmarking test suite...")
    results = {}
    
    # Run pytest collection only (fastest)
    success, elapsed, output = run_command(
        ["python", "-m", "pytest", "tests/", "--collect-only", "-q"],
        timeout=60
    )
    
    results["test_collection"] = round(elapsed, 2)
    
    # Count tests
    test_count = output.count("\n")
    print(f"  Test collection: {elapsed:.2f}s ({test_count} tests found)")
    
    # Run subset of quick tests
    success, elapsed, output = run_command(
        ["python", "-m", "pytest", "tests/test_config.py", "-v", "--tb=short"],
        timeout=60
    )
    
    results["test_config"] = round(elapsed, 2)
    print(f"  Config tests: {elapsed:.2f}s")
    
    return results

def benchmark_installation() -> Dict[str, Tuple[float, str]]:
    """Benchmark package installation with different configurations."""
    print("📊 Benchmarking installation sizes...")
    results = {}
    
    # Create temp directory for venv
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Core installation
        venv_path = tmpdir_path / "venv_core"
        cmd = [
            "python3", "-m", "venv", str(venv_path)
        ]
        run_command(cmd, timeout=30)
        
        # Measure size
        try:
            import shutil
            size_mb = sum(
                f.stat().st_size for f in venv_path.rglob('*') if f.is_file()
            ) / (1024 * 1024)
            results["core_venv"] = (size_mb, f"{size_mb:.1f} MB")
            print(f"  Core venv: {size_mb:.1f} MB")
        except:
            pass
    
    return results

def main():
    """Run all benchmarks."""
    print("════════════════════════════════════════════════════════════")
    print("Performance Baseline - Quick Benchmarks")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("════════════════════════════════════════════════════════════")
    print("")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "benchmarks": {}
    }
    
    try:
        # Run benchmarks
        all_results["benchmarks"]["imports"] = benchmark_import()
        print("")
        
        all_results["benchmarks"]["config"] = benchmark_config_load()
        print("")
        
        all_results["benchmarks"]["features"] = benchmark_optional_features()
        print("")
        
        all_results["benchmarks"]["tests"] = benchmark_test_suite()
        print("")
        
        all_results["benchmarks"]["installation"] = {
            k: v[1] for k, v in benchmark_installation().items()
        }
        print("")
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    print("════════════════════════════════════════════════════════════")
    print("Results Summary")
    print("════════════════════════════════════════════════════════════")
    print("")
    
    results_file = REPO_ROOT / "benchmark_results" / "baseline.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Baseline saved to: {results_file}")
    print("")
    print("Key Metrics:")
    print(f"  • Import time: {all_results['benchmarks'].get('imports', {}).get('montage_ai', 'N/A')}ms")
    print(f"  • Config load: {all_results['benchmarks'].get('config', {}).get('config_load', 'N/A')}ms")
    print(f"  • Test suite: {all_results['benchmarks'].get('tests', {}).get('test_collection', 'N/A')}s")
    print("")
    print("Optional Features Available:")
    features = all_results['benchmarks'].get('features', {})
    for module, available in features.items():
        status = "✓" if available else "✗"
        print(f"  {status} {module}")
    print("")

if __name__ == "__main__":
    main()
