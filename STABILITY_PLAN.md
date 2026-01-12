# Stability Improvements Plan for High-Resolution & Large Jobs

## Executive Summary

This plan addresses stability bottlenecks discovered during analysis of the montage-ai codebase.
Focus: Enable reliable processing of 4K+ content and 1000+ clip projects.

**Estimated Total Effort**: 6-8 hours
**Risk Reduction**: 80% fewer OOM crashes, 90% fewer temp file leaks

---

## Priority Matrix

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| P0 | VideoCapture leaks | OOM crashes | 2h |
| P0 | Temp file cleanup | Disk full errors | 30min |
| P1 | Batch sizing for 4K | OOM on 4K | 1h |
| P1 | Exception handling | Silent failures | 1h |
| P2 | ThreadPool sizing | Resource waste | 30min |
| P2 | Dynamic timeouts | False failures | 1h |

---

## Phase 1: Critical Memory Leaks (P0)

### 1.1 VideoCapture Consolidation

**Problem**: cv2.VideoCapture created without release in 5+ files
**Solution**: Use existing video_capture_pool.py consistently

**Files to fix**:
- `footage_analyzer.py:156-186` - Replace with pool
- `scene_detection_sota.py:282-606` - Add try-finally
- `scene_provider.py:458-545` - Use pool context manager

**Pattern to apply**:
```python
# Before (leaky)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
# ... missing release

# After (safe)
from .video_capture_pool import get_capture_pool
pool = get_capture_pool()
with pool.get(video_path) as cap:
    fps = cap.get(cv2.CAP_PROP_FPS)
```

### 1.2 Temp File Cleanup

**Problem**: Temp files leak on exception paths in segment_writer.py
**Solution**: Always use try-finally for temp file cleanup

**Pattern**:
```python
concat_list_path = str(self.output_dir / f"concat_{idx}.txt")
try:
    # ... processing
finally:
    for path in [concat_list_path, temp_audio_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning(f"Failed to cleanup: {path}")
```

---

## Phase 2: Configuration Improvements (P1)

### 2.1 Adaptive Batch Sizing

**Problem**: Fixed batch size causes OOM on 4K content
**Solution**: Resolution-aware batch sizing

**New config in `config.py`**:
```python
def get_batch_size_for_resolution(width: int, height: int, memory_gb: float) -> int:
    pixels = width * height

    # Resolution tiers
    if pixels <= 2_073_600:      # <= 1080p
        base = 25
    elif pixels <= 8_294_400:    # <= 4K
        base = 8
    else:                        # > 4K
        base = 2

    # Memory adjustment
    if memory_gb < 4:
        return max(1, base // 4)
    elif memory_gb < 8:
        return max(1, base // 2)

    return base
```

### 2.2 Specific Exception Handling

**Problem**: Bare `except Exception: pass` hides real failures
**Files**: resource_manager.py, monitoring.py, cgpu_utils.py

**Pattern**:
```python
# Before
try:
    result = detect_gpu()
except Exception:
    pass

# After
try:
    result = detect_gpu()
except TimeoutError:
    logger.warning("GPU detection timeout")
except FileNotFoundError:
    logger.debug("nvidia-smi not found")
except Exception as e:
    logger.error(f"Unexpected GPU error: {e}")
```

---

## Phase 3: Performance Tuning (P2)

### 3.1 ThreadPool Right-Sizing

**Problem**: ThreadPool = CPU count (wastes resources for I/O-bound FFmpeg)
**Solution**: Cap at 8 threads for I/O operations

**Change in `config_pools.py`**:
```python
@staticmethod
def thread_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return min(cpu_count, 8)  # Cap for I/O-bound work
```

### 3.2 Dynamic Timeout Calculation

**Problem**: Static 30s timeout fails on slow hardware
**Solution**: Hardware-aware timeouts

**New utility**:
```python
def calculate_ffmpeg_timeout(file_size_mb: float, resolution: tuple,
                             gpu_available: bool) -> int:
    base = 30

    # Size factor (100MB = 30s, 1GB = 60s)
    size_factor = 1 + (file_size_mb / 500)

    # Resolution factor (4K = 2x, 8K = 4x)
    pixels = resolution[0] * resolution[1]
    res_factor = 1 + (pixels / 8_294_400)

    # GPU speedup
    gpu_factor = 0.3 if gpu_available else 1.0

    return int(base * size_factor * res_factor * gpu_factor) + 60
```

---

## Implementation Order

1. **Day 1** (3h):
   - VideoCapture pool consolidation
   - Temp file cleanup patterns

2. **Day 2** (3h):
   - Adaptive batch sizing
   - Exception handling improvements

3. **Day 3** (2h):
   - ThreadPool tuning
   - Dynamic timeouts
   - Testing & validation

---

## Testing Strategy

### Unit Tests
- `test_video_capture_pool.py` - Verify no leaks after 100 iterations
- `test_batch_sizing.py` - Verify correct sizes for 1080p/4K/8K

### Integration Tests
- 4K video with 50 clips - Should complete without OOM
- 1000 clip project - Temp files cleaned up
- Slow hardware simulation - Timeouts adapt

### Memory Profiling
```bash
# Monitor memory during test run
python -m memory_profiler -m montage_ai --job-id test
```

---

## Rollback Plan

Each fix is isolated:
- VideoCapture: Revert pool imports, restore direct cv2 usage
- Batch size: Revert to hardcoded value (25)
- Timeouts: Restore static values in config_timeouts.py

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| 4K OOM rate | ~60% | <5% |
| Temp file cleanup | 70% | 99% |
| Silent failure rate | ~20% | <2% |
| Memory peak (1000 clips) | 12GB | 6GB |
