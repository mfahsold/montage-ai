# Logging Strategy & Gap Analysis

**Status:** Draft
**Date:** 2026-01-02
**Scope:** Global logging infrastructure review

## Executive Summary
The current logging infrastructure is **sufficient for local CLI usage** but **insufficient for Enterprise/Distributed deployment**. While the visual presentation for the user is excellent (emojis, clean console output), the backend logging lacks the structure and consistency required for automated monitoring and debugging at scale.

## Current Architecture
*   **Centralized Setup:** `src/montage_ai/logger.py` correctly manages handlers and formatters.
*   **Dual Output:**
    *   **Console:** `MontageFormatter` (Human-readable, Emoji-enhanced).
    *   **File:** `FileFormatter` (Timestamped, detailed).
*   **Context:** Job-based file logging is supported via `configure_file_logging`.

## Gap Analysis

### 1. Inconsistent Subprocess Execution (Critical)
*   **Issue:** While `src/montage_ai/core/cmd_runner.py` provides a robust wrapper with logging and error handling, core modules like `segment_writer.py` and `cgpu_upscaler_v3.py` still use raw `subprocess.run`.
*   **Risk:** Failures in FFmpeg encoding or upscaling may result in silent failures or missing `stderr` context in the logs.
*   **Recommendation:** Refactor all `subprocess` calls to use `cmd_runner.run_command`.

### 2. Lack of Structured Logging (Strategic)
*   **Issue:** Logs are unstructured text.
*   **Risk:** The Web UI (via SSE) and Kubernetes log aggregators (Fluentd/Datadog) cannot easily parse log levels, timestamps, or metadata.
*   **Recommendation:** Implement a `LOG_FORMAT=json` environment variable to switch the formatter to a JSON structure.

### 3. Third-Party Noise
*   **Issue:** Libraries like `urllib3`, `moviepy`, and `matplotlib` are not explicitly silenced or managed.
*   **Risk:** Debug logs can be flooded with HTTP connection details or font manager debug info, burying actual application logic.
*   **Recommendation:** Configure default log levels for known noisy libraries in `logger.py`.

### 4. Distributed Tracing
*   **Issue:** In a K3s cluster, logs from different pods are disjointed.
*   **Risk:** Hard to trace a single job across multiple pods (e.g., split between analysis and rendering).
*   **Recommendation:** Inject `job_id` and `shard_index` into the `extra` dict of every log record.

## Roadmap

### Phase 1: Stabilization (Immediate)
- [ ] Refactor `segment_writer.py` to use `cmd_runner`.
- [ ] Configure silence for `urllib3` and `matplotlib`.

### Phase 2: Enterprise Readiness
- [ ] Implement JSON formatter.
- [ ] Add `correlation_id` support for distributed tracing.
