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

### 4. LLM & Creative Director Configuration

**Problem:** `src/montage_ai/creative_director.py` contained global variables initialized from environment variables at module level, making it hard to test and reload configuration.

**Solution:** Refactored `CreativeDirector` to use `settings.llm` for all backend configuration (OpenAI, Google AI, Ollama, cgpu).

**Changes:**
- Removed global constants (`OPENAI_API_KEY`, `CGPU_HOST`, etc.) from `creative_director.py`.
- Updated `__init__` to accept `None` for optional arguments and fall back to `settings.llm`.
- Updated API client initialization to use `settings.llm` properties.

### 5. FFmpeg & Encoding Configuration

**Problem:** `src/montage_ai/ffmpeg_config.py` used a local helper `_env_or_default` to read environment variables, duplicating logic found in `config.py`.

**Solution:**
- Updated `EncodingConfig` in `config.py` to include all necessary FFmpeg settings (added `audio_codec`, `audio_bitrate`).
- Refactored `FFmpegConfig` dataclass in `ffmpeg_config.py` to use `settings.encoding` and `settings.gpu` for default values.
- Removed `_env_or_default` helper.

### 6. CGPU Configuration

**Problem:** `src/montage_ai/cgpu_jobs/analysis.py` used a hardcoded `os.environ.get("CGPU_OUTPUT_DIR")`.

**Solution:**
- Added `cgpu_output_dir` to `LLMConfig` in `config.py`.
- Updated `analysis.py` to use `settings.llm.cgpu_output_dir`.

## Verification

Run the following tests to verify the configuration changes:

```bash
# Verify configuration loading
python3 -m pytest tests/test_config.py

# Verify editor and creative director integration
python3 -m pytest tests/test_editor_basic.py
```

### 7. Command Execution Standardization

**Problem:** `subprocess.run` was used directly in many places with inconsistent error handling, logging, and timeout management.

**Solution:**
- Created `src/montage_ai/core/cmd_runner.py` with `run_command` helper.
- Refactored `audio_analysis.py`, `clip_enhancement.py`, and `ffmpeg_tools.py` to use `run_command`.
- This ensures consistent logging of commands and errors, and centralized timeout handling.

### 8. Style Templates Configuration

**Problem:** `src/montage_ai/style_templates.py` used `os.environ.get` directly for style paths.

**Solution:**
- Added `style_preset_path` and `style_preset_dir` to `PathConfig` in `config.py`.
- Updated `style_templates.py` to use `settings.paths`.
