# Test Infrastructure & Optimized Test Runner

This document explains the `scripts/run_optimized_tests.py` utility, monitoring behavior, and recovery strategies used by Montage AI for local performance tests.

## Overview

`run_optimized_tests.py` provides several test profiles (smoke, standard, 4k-quick, stress, parallel).
It chooses eligible videos from `data/input/` based on configured `max_input_duration` and runs Montage with a controlled environment.

## Monitoring

- A lightweight monitor runs alongside each test and samples system CPU/memory every 5 seconds.
- If `nvidia-smi` is available, basic GPU utilization is attempted.
- Monitoring JSON files are saved to `data/output/monitoring_<timestamp>_<TestName>.json` on test completion or interruption.

## Robustness

- The runner uses `subprocess.Popen` and `communicate()` to support:
  - Clean timeouts (returns status `timeout` and captures `stderr`)
  - KeyboardInterrupt handling (returns status `interrupted` and kills the child process)
- If a GPU encoder fails during rendering (e.g., VAAPI hwupload errors), the renderer will automatically retry with a software encoder (`libx264`) to avoid hard failures.

## Usage

- Run smoke test:
  ```bash
  PYTHONPATH=src .venv/bin/python scripts/run_optimized_tests.py smoke
  ```
- Run 4K quick test:
  ```bash
  PYTHONPATH=src .venv/bin/python scripts/run_optimized_tests.py 4k-quick
  ```

## Where to look for data
- Render logs: `data/output/render.log`
- Monitoring: `data/output/monitoring_*.json`

## Notes & Troubleshooting
- If monitoring shows `cpu: 100%` and memory > 80%, consider running the test with `LOW_MEMORY_MODE=true` or reducing `BATCH_SIZE`.
- If VAAPI errors persist, confirm `/dev/dri` exists and is mapped into containers or test nodes, or force `FFMPEG_HWACCEL=none` to use software encoder.
