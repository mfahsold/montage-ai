# Codebase Stabilization & Refactoring

## Overview

This document outlines the stabilization efforts undertaken to improve the robustness, maintainability, and configurability of the Montage AI codebase.

## Key Improvements

### 1. Centralized Configuration

**Problem:** The codebase previously relied on scattered `os.environ.get()` calls with hardcoded default values (e.g., `/data/input`, `/tmp`) distributed across many files. This made it difficult to change paths or defaults consistently and led to "magic strings" throughout the code.

**Solution:** All configuration is now centralized in `src/montage_ai/config.py` using the `Settings` class (Pydantic/Dataclasses).

**Usage:**
Instead of:
```python
# BAD
input_dir = os.environ.get("INPUT_DIR", "/data/input")
```

Use:
```python
# GOOD
from .config import get_settings
settings = get_settings()
input_dir = settings.paths.input_dir
```

### 2. Removal of Hardcoded Paths

**Problem:** Hardcoded paths like `/tmp/segments` or `/data/luts` made the application brittle and difficult to run in different environments (e.g., local dev vs. container vs. CI).

**Solution:** All paths are now derived from the `PathConfig` in `Settings`.

**Refactored Modules:**
- `src/montage_ai/ffmpeg_tools.py`: Uses `settings.paths.lut_dir` and `settings.paths.temp_dir`.
- `src/montage_ai/timeline_exporter.py`: Uses `settings.paths.output_dir`.
- `src/montage_ai/video_agent.py`: Uses `settings.paths.temp_dir` for the SQLite database.
- `src/montage_ai/segment_writer.py`: Uses `settings.paths.temp_dir` for intermediate segments.
- `src/montage_ai/cgpu_upscaler_v3.py`: Uses `settings.paths.temp_dir` for script generation.
- `src/montage_ai/cgpu_jobs/analysis.py`: Uses `settings.paths.output_dir`.
- `src/montage_ai/broll_planner.py`: Uses `settings.paths.input_dir`.
- `src/montage_ai/monitoring.py`: Uses `settings.paths.output_dir` for logs.

### 3. Configuration Bug Fixes

- **Export Config:** Fixed a bug where `ExportConfig` was defined but not included in the main `Settings` class, causing `AttributeError` when accessing `settings.export`.

## Testing

The changes have been verified using the test suite:
```bash
make test
```
or
```bash
pytest tests/
```

All tests are passing, ensuring that the refactoring did not introduce regressions.

## Future Guidelines

- **No Magic Strings:** Do not hardcode paths or configuration values in functional code. Add them to `config.py` if they are missing.
- **Use `get_settings()`:** Always retrieve configuration via the singleton `get_settings()` function.
- **Type Safety:** Rely on the typed fields in `Settings` rather than parsing environment variables manually.
