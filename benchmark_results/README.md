# Performance Baselines

This directory contains performance baseline measurements for Montage AI, used to track optimization progress over time.

## Overview

Performance baselines establish reference metrics to:
- Track optimization improvements across releases
- Identify performance regressions
- Validate changes in import time, startup time, and test performance
- Document available optional features in each environment

## Files

- **`baseline.json`**: Current baseline measurements (auto-generated)
- **`phase3_benchmark.json`**: Previous phase 3 results (historical)
- **`micro_benchmark_baseline.json`**: Low-level performance metrics

## Running Benchmarks

### Quick Baseline (< 2 min)

```bash
make benchmark
```

This will:
1. Measure import times for core modules
2. Test configuration loading
3. Check available optional features
4. Run test suite collection
5. Estimate installation sizes
6. Save results to `baseline.json`

### Optional Dependencies Validation

```bash
make validate-deps
```

This will:
1. Test core installation
2. Test each optional group ([ai], [web], [cloud])
3. Test combined installation
4. Estimate venv sizes per configuration
5. Report which packages are working

## Key Metrics

### Import Performance

Lower is better (milliseconds):

| Module | Baseline | Target |
|--------|----------|--------|
| montage_ai | 3045ms | < 2000ms |
| montage_ai.config | 0ms | < 100ms |
| montage_ai.core.montage_builder | 0ms | < 100ms |

*Note: First import slower due to GPU initialization*

### Configuration Loading

| Metric | Current | Target |
|--------|---------|--------|
| Config load time | 0.01ms | < 1ms |

### Optional Features

Tracks which optional packages are available:

✓ Available:
- librosa (audio analysis)
- scipy (signal processing)  
- flask (web UI)
- redis (cache backend)

✗ Not available (install [ai], [cloud] group):
- mediapipe (face detection)
- torch (deep learning)

### Installation Sizes

Estimated venv sizes:

| Configuration | Size |
|---------------|------|
| core | 35.6 MB |
| core + [ai] | ~420 MB |
| core + [web] | ~155 MB |
| core + [cloud] | ~200 MB |
| core + [all] | ~550 MB |

## Test Suite Performance

| Metric | Value |
|--------|-------|
| Test collection time | 15.22s |
| Config tests | 10.65s |
| Total tests discovered | 803 |

## Tracking Improvements

To compare against baseline:

```bash
# Run new benchmark
make benchmark

# Manually compare files
diff benchmark_results/baseline.json benchmark_results/previous.json

# Or use jq for specific metrics
jq '.benchmarks.imports' benchmark_results/baseline.json
```

### Example Optimization Workflow

1. **Identify slow imports**:
   ```bash
   jq '.benchmarks.imports' benchmark_results/baseline.json
   ```

2. **Make optimization changes**

3. **Verify improvement**:
   ```bash
   make benchmark
   # Compare new results
   ```

4. **Document in commit message**

## Integration with CI/CD

Baselines can be integrated into CI/CD pipelines:

```bash
# In CI: Run benchmark and check against threshold
make benchmark

# Fail if import time degrades > 10%
python3 scripts/check_performance_regression.py \
  --baseline baseline.json \
  --threshold 1.1
```

## Historical Tracking

Baselines are committed to repository:

```bash
git log --oneline -- benchmark_results/baseline.json
```

This creates a performance history you can compare against:

```bash
git diff HEAD~5 benchmark_results/baseline.json
```

## Notes

- First import of montage_ai is slower (~3s) due to GPU encoding initialization
- Subsequent imports are much faster
- Optional features shown depend on installed packages
- Installation sizes estimated for Python 3.10 virtual environments
- Measurements taken on reference hardware (varies by system)
