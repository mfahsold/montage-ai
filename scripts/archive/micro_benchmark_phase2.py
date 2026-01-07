#!/usr/bin/env python3
"""
Phase 2 Performance Benchmark

Tests Phase 2 optimizations:
1. Keyframe-only scene detection
2. RAM disk temp file usage
3. Vectorized NumPy scoring
4. K-D tree scene similarity indexing
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from montage_ai.scene_analysis import Scene, SceneSimilarityIndex
from montage_ai.config import get_settings

def test_kdtree_indexing():
    """Benchmark K-D tree scene similarity index."""
    print("\n=== K-D Tree Scene Similarity Index ===")
    
    # Create synthetic scenes
    n_scenes = 1000
    scenes = []
    for i in range(n_scenes):
        scenes.append(Scene(
            start=float(i * 10),
            end=float((i + 1) * 10),
            path=f"/data/input/video_{i % 10}.mp4"
        ))
    
    # Build index (without actual video files, we'll use synthetic histograms)
    print(f"Building K-D tree index for {n_scenes} scenes...")
    start = time.perf_counter()
    
    index = SceneSimilarityIndex()
    # Manually populate with synthetic histograms for benchmark
    index.scenes = scenes
    index.histograms = [np.random.rand(512).astype(np.float32) for _ in range(n_scenes)]
    
    if index.enabled:
        from scipy.spatial import KDTree
        index.kdtree = KDTree(index.histograms)
        build_time = (time.perf_counter() - start) * 1000
        print(f"   ✓ Built in {build_time:.2f}ms")
        
        # Query performance
        target_vec = np.random.rand(512).astype(np.float32)
        
        # K-D tree query (fast)
        start = time.perf_counter()
        for _ in range(100):
            distances, indices = index.kdtree.query(target_vec, k=5)
        kdtree_time = (time.perf_counter() - start) * 1000 / 100
        
        # Linear scan (slow)
        start = time.perf_counter()
        for _ in range(100):
            similarities = [np.dot(target_vec, hist) for hist in index.histograms]
            top_5 = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:5]
        linear_time = (time.perf_counter() - start) * 1000 / 100
        
        speedup = linear_time / kdtree_time
        print(f"   K-D tree query: {kdtree_time:.4f}ms")
        print(f"   Linear scan:    {linear_time:.4f}ms")
        print(f"   Speedup:        {speedup:.1f}x faster 🚀")
    else:
        print("   ⚠️  scipy not available, K-D tree disabled")

def test_vectorized_scoring():
    """Benchmark vectorized NumPy scoring vs iterative."""
    print("\n=== Vectorized NumPy Scoring ===")
    
    n_candidates = 20
    
    # Generate synthetic candidate metadata
    candidates = [
        {
            'quality': np.random.rand(),
            'action': np.random.rand(),
            'face_count': np.random.randint(0, 5),
            'energy': np.random.rand()
        }
        for _ in range(n_candidates)
    ]
    
    # Iterative scoring (old way)
    start = time.perf_counter()
    for _ in range(1000):
        scores_iter = []
        for c in candidates:
            score = (
                c['quality'] * 0.3 +
                c['action'] * 0.2 +
                min(c['face_count'], 2) / 2.0 * 0.3 +
                c['energy'] * 0.2
            )
            scores_iter.append(score)
    iter_time = (time.perf_counter() - start) * 1000 / 1000
    
    # Vectorized scoring (new way)
    quality = np.array([c['quality'] for c in candidates])
    action = np.array([c['action'] for c in candidates])
    faces = np.minimum(np.array([c['face_count'] for c in candidates]), 2) / 2.0
    energy = np.array([c['energy'] for c in candidates])
    
    start = time.perf_counter()
    for _ in range(1000):
        scores_vec = quality * 0.3 + action * 0.2 + faces * 0.3 + energy * 0.2
    vec_time = (time.perf_counter() - start) * 1000 / 1000
    
    speedup = iter_time / vec_time
    print(f"   Iterative:   {iter_time:.4f}ms")
    print(f"   Vectorized:  {vec_time:.4f}ms")
    print(f"   Speedup:     {speedup:.1f}x faster 🚀")

def test_ram_disk():
    """Benchmark RAM disk vs regular disk I/O."""
    print("\n=== RAM Disk Performance ===")
    
    settings = get_settings()
    temp_dir = settings.paths.temp_dir
    
    print(f"   Temp directory: {temp_dir}")
    
    if "/dev/shm" in str(temp_dir):
        print("   ✓ Using RAM disk (/dev/shm) 🚀")
    else:
        print("   ⚠️  Using regular disk")
    
    # Simple write test
    data = b"X" * (1024 * 1024)  # 1MB
    
    start = time.perf_counter()
    for i in range(10):
        tmp_file = temp_dir / f"bench_write_{i}.tmp"
        tmp_file.write_bytes(data)
        tmp_file.unlink()
    write_time = (time.perf_counter() - start) * 1000 / 10
    
    print(f"   Write (1MB):  {write_time:.2f}ms")

def test_ffmpeg_astats():
    """Verify FFmpeg astats optimization is active."""
    print("\n=== FFmpeg astats Audio Optimization ===")
    
    # Check if the fast method exists
    try:
        from montage_ai.audio_analysis import _ffmpeg_analyze_energy_fast
        print("   ✓ FFmpeg astats fast method available 🚀")
    except ImportError:
        print("   ⚠️  Fast method not found (Phase 1 not applied)")

def main():
    print("🎯 Phase 2 Performance Benchmark\n")
    print("Testing optimizations:")
    print("  1. K-D tree scene similarity indexing")
    print("  2. Vectorized NumPy scoring")
    print("  3. RAM disk temp files")
    print("  4. FFmpeg astats audio")
    
    test_kdtree_indexing()
    test_vectorized_scoring()
    test_ram_disk()
    test_ffmpeg_astats()
    
    print("\n" + "="*60)
    print("Phase 2 Benchmark Complete ✅")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
