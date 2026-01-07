#!/usr/bin/env python3
"""
Montage AI Performance Baseline Benchmark

Comprehensive performance testing framework to identify bottlenecks
across all features, flows, and data transfers.

Tests:
1. Audio Analysis (Beat detection, energy profiling)
2. Scene Detection & Content Analysis
3. Clip Selection & Timeline Assembly
4. Enhancement Pipeline (Stabilization, Upscaling, Color)
5. Rendering (Progressive, Segment Writing, FFmpeg)
6. Data I/O (File reads, FFprobe, temp file management)
7. End-to-End Montage Creation (Full workflow)
"""

import os
import sys
import time
import tempfile
import shutil
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ============================================================================
# Performance Measurement Utilities
# ============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark test result."""
    name: str
    category: str
    duration_seconds: float
    memory_mb: float = 0.0
    throughput: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        return self.duration_seconds * 1000


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    timestamp: str
    system_info: Dict[str, str]
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add(self, result: BenchmarkResult):
        self.results.append(result)
    
    def get_category_total(self, category: str) -> float:
        return sum(r.duration_seconds for r in self.results if r.category == category)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def print_summary(self):
        print("\n" + "="*80)
        print("MONTAGE AI PERFORMANCE BASELINE BENCHMARK")
        print("="*80)
        
        # Group by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)
        
        for cat, results in categories.items():
            print(f"\n📊 {cat}")
            print("-" * 80)
            total = 0.0
            for r in results:
                throughput_str = f" ({r.throughput})" if r.throughput else ""
                print(f"  {r.name:50s} {r.duration_ms:>10.2f}ms{throughput_str}")
                total += r.duration_seconds
            print(f"  {'TOTAL':50s} {total*1000:>10.2f}ms")
        
        # Overall
        total_time = sum(r.duration_seconds for r in self.results)
        print("\n" + "="*80)
        print(f"TOTAL BENCHMARK TIME: {total_time:.2f}s ({total_time/60:.2f}min)")
        print("="*80 + "\n")


@contextmanager
def benchmark(name: str, category: str, suite: BenchmarkSuite):
    """Context manager for timing operations."""
    start = time.perf_counter()
    memory_before = get_memory_usage_mb()
    
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        memory_after = get_memory_usage_mb()
        memory_delta = memory_after - memory_before
        
        result = BenchmarkResult(
            name=name,
            category=category,
            duration_seconds=duration,
            memory_mb=memory_delta
        )
        suite.add(result)


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_system_info() -> Dict[str, str]:
    """Collect system information."""
    import platform
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": str(os.cpu_count()),
    }
    
    # FFmpeg version
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info["ffmpeg_version"] = result.stdout.split('\n')[0]
    except:
        info["ffmpeg_version"] = "unknown"
    
    return info


# ============================================================================
# Test Data Setup
# ============================================================================

class TestDataManager:
    """Manages test video and audio files."""
    
    def __init__(self):
        # Use /data directory if it exists, otherwise fall back to repo data
        if Path("/data").exists() and (Path("/data") / "input").exists():
            self.data_dir = Path("/data")
        else:
            self.data_dir = Path(__file__).parent / "data"
        
        self.input_dir = self.data_dir / "input"
        self.music_dir = self.data_dir / "music"
        self.output_dir = Path(__file__).parent / "benchmark_results"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="montage_benchmark_"))
        self._cached_videos: Optional[List[Path]] = None
        self._cached_music: Optional[Path] = None
        
    def setup(self):
        """Ensure test data exists."""
        if self._cached_videos is not None and self._cached_music is not None:
            return self._cached_videos, self._cached_music

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for test videos
        video_files = list(self.input_dir.glob("*.mp4")) + list(self.input_dir.glob("*.mov"))
        if not video_files:
            print(f"⚠️  No test videos found in {self.input_dir}. Baseline benchmark will be skipped.")
            self._cached_videos, self._cached_music = [], None
            return self._cached_videos, self._cached_music
        
        # Check for test music
        music_files = list(self.music_dir.glob("*.mp3")) + list(self.music_dir.glob("*.wav"))
        if not music_files:
            print(f"⚠️  No test music found in {self.music_dir}. Baseline benchmark will be skipped.")
            self._cached_videos, self._cached_music = [], None
            return self._cached_videos, self._cached_music
        
        print(f"✅ Found {len(video_files)} test videos")
        print(f"✅ Found {len(music_files)} test music files")
        
        self._cached_videos, self._cached_music = video_files[:5], music_files[0]
        return self._cached_videos, self._cached_music  # Use first 5 videos
    
    def cleanup(self):
        """Remove temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


# ============================================================================
# Individual Benchmark Tests
# ============================================================================

def benchmark_audio_analysis(suite: BenchmarkSuite, data: TestDataManager):
    """Test audio beat detection and energy profiling."""
    from montage_ai.audio_analysis import analyze_audio
    
    _, music_file = data.setup()
    if not music_file:
        print("⚠️  Skipping audio analysis benchmark (no test music).")
        return
    
    with benchmark("Audio: Beat Detection (librosa)", "Audio Analysis", suite):
        beat_info, energy_profile = analyze_audio(str(music_file))
    
    suite.results[-1].details = {
        "beats_detected": len(beat_info.beat_times),
        "tempo": beat_info.tempo,
        "duration": beat_info.duration
    }
    suite.results[-1].throughput = f"{beat_info.duration/suite.results[-1].duration_seconds:.1f}x realtime"


def benchmark_scene_detection(suite: BenchmarkSuite, data: TestDataManager):
    """Test video scene detection."""
    from montage_ai.scene_analysis import detect_scenes
    
    videos, _ = data.setup()
    if not videos:
        print("⚠️  Skipping scene detection benchmark (no test videos).")
        return
    
    for i, video in enumerate(videos[:3]):  # Test first 3 videos
        with benchmark(f"Scene: Detection #{i+1} ({video.name})", "Scene Detection", suite):
            scenes = detect_scenes(str(video))
        
        suite.results[-1].details = {
            "scenes_detected": len(scenes),
            "video_name": video.name
        }


def benchmark_content_analysis(suite: BenchmarkSuite, data: TestDataManager):
    """Test AI content analysis (faces, action, quality)."""
    from montage_ai.scene_analysis import analyze_scene_content, Scene
    
    videos, _ = data.setup()
    if not videos:
        print("⚠️  Skipping content analysis benchmark (no test videos).")
        return
    
    # Create test scene
    test_scene = Scene(
        path=str(videos[0]),
        start_time=0.0,
        duration=5.0,
        scene_number=1
    )
    
    with benchmark("Content: Scene Content Analysis", "Content Analysis", suite):
        try:
            result = analyze_scene_content(test_scene)
            suite.results[-1].details = {
                "has_faces": result.has_faces if hasattr(result, 'has_faces') else False,
                "action_level": str(result.action_level) if hasattr(result, 'action_level') else "unknown"
            }
        except Exception as e:
            print(f"  ⚠️  Content analysis test skipped: {e}")


def benchmark_clip_selection(suite: BenchmarkSuite, data: TestDataManager):
    """Test clip selection algorithm."""
    from montage_ai.core.selection_engine import SelectionEngine
    from montage_ai.core.context import MontageContext
    from montage_ai.config import get_settings
    
    # Create mock context
    settings = get_settings()
    ctx = MontageContext(settings=settings)
    
    # Mock data
    ctx.timeline.target_duration = 60.0
    ctx.media.audio_result = type('obj', (object,), {
        'beat_times': [i * 0.5 for i in range(120)],
        'energy_values': [0.5] * 120,
        'tempo': 120
    })()
    
    # Create fake scenes
    from montage_ai.scene_analysis import Scene
    ctx.media.all_scenes = [
        Scene(
            path=f"/fake/video{i}.mp4",
            start_time=0.0,
            duration=10.0,
            scene_number=i
        )
        for i in range(50)
    ]
    
    engine = SelectionEngine(ctx)
    
    with benchmark("Selection: Clip Selection Algorithm", "Timeline Assembly", suite):
        # Simulate selection loop (simplified)
        selected_count = 0
        for beat_idx in range(100):
            scene = engine.select_clip(
                current_time=beat_idx * 0.5,
                current_energy=0.5,
                recent_videos=set()
            )
            if scene:
                selected_count += 1
    
    suite.results[-1].details = {"clips_selected": selected_count}


def benchmark_clip_enhancement(suite: BenchmarkSuite, data: TestDataManager):
    """Test clip enhancement pipeline."""
    from montage_ai.clip_enhancement import stabilize_clip, enhance_clip
    from montage_ai.config import get_settings
    
    videos, _ = data.setup()
    if not videos:
        print("⚠️  Skipping enhancement benchmark (no test videos).")
        return
    test_video = videos[0]
    
    settings = get_settings()
    
    # Test individual enhancements
    output_stable = data.temp_dir / "test_stabilized.mp4"
    
    # Stabilization
    with benchmark("Enhancement: Stabilization (vidstab)", "Enhancement Pipeline", suite):
        try:
            stabilize_clip(str(test_video), str(output_stable), 0, 3.0, settings)
        except Exception as e:
            print(f"  ⚠️  Stabilization test failed: {e}")
    
    # Enhancement (brightness, contrast)
    output_enhanced = data.temp_dir / "test_enhanced.mp4"
    with benchmark("Enhancement: Brightness/Contrast", "Enhancement Pipeline", suite):
        try:
            enhance_clip(str(test_video), str(output_enhanced), 0, 3.0, settings)
        except Exception as e:
            print(f"  ⚠️  Enhancement test failed: {e}")


def benchmark_rendering(suite: BenchmarkSuite, data: TestDataManager):
    """Test rendering and segment writing."""
    from montage_ai.segment_writer import SegmentWriter
    from montage_ai.config import get_settings
    
    videos, _ = data.setup()
    if not videos:
        print("⚠️  Skipping rendering benchmark (no test videos).")
        return
    settings = get_settings()
    
    # Create segment writer
    output_path = data.output_dir / "test_render.mp4"
    
    with benchmark("Rendering: SegmentWriter Initialization", "Rendering", suite):
        writer = SegmentWriter(
            output_path=str(output_path),
            audio_path=None,
            target_duration=10.0,
            fps=30,
            width=1920,
            height=1080,
            settings=settings
        )
    
    # Add test segments
    with benchmark("Rendering: Add 5 Segments", "Rendering", suite):
        for i in range(5):
            writer.add_segment(str(videos[i % len(videos)]), 0, 2.0)
    
    with benchmark("Rendering: Finalize Output", "Rendering", suite):
        writer.finalize()


def benchmark_ffmpeg_operations(suite: BenchmarkSuite, data: TestDataManager):
    """Test FFmpeg operations."""
    videos, music = data.setup()
    if not videos:
        print("⚠️  Skipping FFmpeg operations benchmark (no test videos).")
        return
    test_video = videos[0]
    
    # FFprobe metadata extraction
    with benchmark("FFmpeg: Metadata Extraction (ffprobe)", "Data I/O", suite):
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(test_video)],
            capture_output=True,
            timeout=30
        )
    
    # Simple clip extraction
    output_clip = data.temp_dir / "test_clip.mp4"
    with benchmark("FFmpeg: Clip Extraction (3s)", "Data I/O", suite):
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "0", "-t", "3", "-i", str(test_video), "-c", "copy", str(output_clip)],
            capture_output=True,
            timeout=30
        )


def benchmark_end_to_end(suite: BenchmarkSuite, data: TestDataManager):
    """Test complete montage creation workflow."""
    from montage_ai.core.montage_builder import MontageBuilder
    from montage_ai.config import get_settings
    
    settings = get_settings()
    
    # Configure for fast preview
    settings.encoding.quality_profile = "preview"  # 360p, ultrafast
    settings.features.stabilize = False
    settings.features.upscale = False
    settings.features.enhance = False
    settings.processing.parallel_jobs = 2
    
    with benchmark("E2E: Complete Montage (Preview Mode)", "End-to-End", suite):
        builder = MontageBuilder(
            variant_id=999,  # Test variant
            settings=settings
        )
        result = builder.build()
    
    if result.success:
        suite.results[-1].details = {
            "output_duration": result.duration,
            "cut_count": result.cut_count,
            "render_time": result.render_time,
            "file_size_mb": result.file_size_mb
        }
        suite.results[-1].throughput = f"{result.duration/result.render_time:.2f}x realtime"


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_baseline_benchmark():
    """Run complete baseline benchmark suite."""
    print("\n🚀 Starting Montage AI Performance Baseline Benchmark\n")
    
    # Setup
    suite = BenchmarkSuite(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        system_info=get_system_info()
    )
    data = TestDataManager()
    videos, music = data.setup()
    if not videos or not music:
        print("⚠️  Baseline benchmark skipped. Add sample media to test_data/input and test_data/music (or /data) to enable.")
        data.cleanup()
        return
    
    try:
        # Print system info
        print("System Information:")
        for key, value in suite.system_info.items():
            print(f"  {key}: {value}")
        print()
        
        # Run benchmarks
        print("Running benchmarks...\n")
        
        benchmark_audio_analysis(suite, data)
        benchmark_scene_detection(suite, data)
        benchmark_content_analysis(suite, data)
        benchmark_clip_selection(suite, data)
        benchmark_clip_enhancement(suite, data)
        benchmark_rendering(suite, data)
        benchmark_ffmpeg_operations(suite, data)
        benchmark_end_to_end(suite, data)
        
        # Save results
        output_file = data.output_dir / "baseline_results.json"
        suite.save(str(output_file))
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print summary
        suite.print_summary()
        
    finally:
        data.cleanup()


if __name__ == "__main__":
    run_baseline_benchmark()
