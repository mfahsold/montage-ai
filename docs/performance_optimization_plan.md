# Performance Optimization Implementation Plan

## Baseline Results Analysis

**Bottlenecks Identified:**
1. ⚠️ NumPy Energy Profile Calculation: **552ms** (53% of total time)
2. ⚠️ File I/O Operations: **189ms** (18%)
3. ⚠️ FFmpeg Video Creation: **185ms** (18%)

**Total Baseline Time:** 1044.86ms

---

## Optimization Strategy: Quick Wins (Phase 1)

### 1. Audio Analysis Optimization ✅ 
**Target:** 10-20x speedup on energy profiling

**Implementation:**
- Use FFmpeg `astats` filter instead of NumPy window iteration
- Downsample audio to 11kHz for beat detection
- Implement lazy loading for preview mode

**Expected Impact:** 552ms → 30-50ms (500ms saved)

### 2. Scene Detection Optimization ✅
**Target:** 5-10x speedup

**Implementation:**
- Keyframe-only detection (skip non-keyframes)
- Adaptive threshold based on video characteristics
- Progressive analysis (early exit when enough scenes found)
- Add size limits to scene cache (LRU eviction)

**Expected Impact:** Current ~2-3s per video → 300-500ms

### 3. File I/O Optimization ✅
**Target:** 30-40% improvement

**Implementation:**
- Use `/dev/shm` (RAM disk) for temporary files on Linux
- Implement asynchronous I/O for non-blocking operations
- Batch file operations where possible
- Explicit cleanup with context managers

**Expected Impact:** 189ms → 120ms (69ms saved)

### 4. Caching Improvements ✅
**Target:** Better hit rates, faster serialization

**Implementation:**
- Implement LRU cache with max size (100 entries)
- Use content-addressable keys (SHA256 of file+mtime+size)
- Add cache statistics/metrics
- Use msgpack instead of JSON for faster serialization

**Expected Impact:** 63ms → 25ms (38ms saved), better cache hits

### 5. Timeline Assembly Optimization ✅
**Target:** Vectorized operations

**Implementation:**
- Use NumPy vectorized scoring instead of Python loops
- Pre-compute scene indices with K-D tree for fast lookup
- Implement Bloom filter for "recently used" checks
- Batch candidate evaluation

**Expected Impact:** 17ms → 5ms (12ms saved)

---

## Implementation Code Changes

### File 1: `src/montage_ai/audio_analysis_optimized.py` (NEW)
Fast audio energy profiling using FFmpeg astats filter.

### File 2: `src/montage_ai/scene_analysis.py` (MODIFY)
Add keyframe-only detection and LRU cache size limits.

### File 3: `src/montage_ai/core/selection_engine.py` (MODIFY)
Vectorized scoring with NumPy, K-D tree indexing.

### File 4: `src/montage_ai/utils.py` (MODIFY)
Add RAM disk support for temp files, async I/O helpers.

### File 5: `src/montage_ai/core/analysis_cache.py` (MODIFY)
Content-addressable caching, LRU eviction, msgpack serialization.

---

## Expected Results After Phase 1

| Component | Baseline | Optimized | Savings |
|-----------|----------|-----------|---------|
| Audio Analysis | 552ms | 50ms | 502ms |
| File I/O | 189ms | 120ms | 69ms |
| Caching | 63ms | 25ms | 38ms |
| Selection | 17ms | 5ms | 12ms |
| **TOTAL** | **1045ms** | **~400ms** | **621ms (59%)** |

**Target Achievement:** 2.6x speedup on micro-benchmarks
**Full Pipeline Impact:** Estimated 3-5min → 1.5-2min for complete montage

---

## Validation Plan

1. ✅ Run micro-benchmark baseline (DONE)
2. ⏳ Implement optimizations
3. ⏳ Run micro-benchmark optimized
4. ⏳ Compare results and document gains
5. ⏳ Run end-to-end montage test (real workflow)
6. ⏳ Update telemetry to track improvements

---

## Implementation Status

- [x] Research SOTA best practices
- [x] Create baseline benchmark
- [x] Analyze bottlenecks
- [ ] Implement audio analysis optimization
- [ ] Implement scene detection optimization
- [ ] Implement file I/O optimization
- [ ] Implement caching improvements
- [ ] Implement selection optimization
- [ ] Run optimized benchmark
- [ ] Document results

---

**Next Action:** Start implementing optimizations in order of impact (Audio → File I/O → Selection → Caching)
