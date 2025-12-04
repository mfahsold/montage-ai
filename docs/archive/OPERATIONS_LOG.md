# Operations Log (Engineering)

Consolidated internal notes that were previously scattered across multiple ad‑hoc markdowns. These are retained for engineering context but kept out of the user-facing doc set.

---

## Stability Improvements (2025-12-02)
- Memory cleanup in `editor.py`: tracks temp clips and deletes them; closes MoviePy clips to avoid RAM bloat (~10GB → ~2GB for 50 clips).
- Cloud GPU pipeline: uploads pipeline script as a file instead of inline string to avoid quote/escape issues; easier debugging.
- CUDA diagnostics: parses stdout/stderr for common CUDA failure signatures (OOM, missing GPU) and prints actionable hints.

## cgpu Cloud GPU Fixes (2025-12-02)
- Increased default `CGPU_TIMEOUT` to 1800s to accommodate large uploads/renders.
- Retry logic no longer retries after a timeout (prevents 20+ minute stalls); still retries session errors.
- Dynamic upload timeout: 1 minute per 10MB, minimum 10 minutes.
- Detailed failure output now includes file name, size, first error line, and self-serve troubleshooting steps.

## cgpu Manual Test Verification (2025-12-02)
- **Status:** ✅ Successful on Tesla T4 (15GB VRAM).
- Verified commands: `cgpu status`, `cgpu copy` (8.3MB sample), Python `subprocess` integration.
- Confirmed new timeout defaults and dynamic upload timeout math.
- Error handling prints concise diagnostics instead of silent failures.
