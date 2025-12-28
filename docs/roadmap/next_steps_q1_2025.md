# Montage AI - Roadmap Q1 2025

## Status Summary

### Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Cleanup (delete open_sora.py, wan_vace.py) | ✅ Done |
| Phase 2 | Smart Editor (transcriber.py, broll_planner.py) | ✅ Done |
| Phase 3 | Cloud GPU Pipeline (cgpu_jobs module) | ✅ Done |

### Current Architecture

```
src/montage_ai/
├── cgpu_jobs/              # NEW: Unified job architecture
│   ├── __init__.py         # Lazy imports
│   ├── base.py             # CGPUJob, JobStatus, JobResult
│   ├── manager.py          # CGPUJobManager (singleton, queue)
│   ├── transcribe.py       # TranscribeJob (Whisper)
│   ├── upscale.py          # UpscaleJob (Real-ESRGAN)
│   └── stabilize.py        # StabilizeJob (FFmpeg vidstab)
├── transcriber.py          # LEGACY: Direct cgpu_utils usage
├── cgpu_upscaler.py        # LEGACY: 760 lines, monolithic
├── broll_planner.py        # Uses video_agent for semantic search
├── editor.py               # Main orchestrator
├── creative_director.py    # LLM interface
└── web_ui/app.py           # Flask UI
```

---

## Phase 4: Integration & Migration (Next)

### 4.1 Migrate transcriber.py → TranscribeJob

**Current State:**
- `transcriber.py` directly uses `cgpu_utils` functions
- Manual session management, no retry logic
- ~110 lines of custom code

**Target:**
- Thin wrapper around `TranscribeJob`
- Preserves backward-compatible API
- Benefits from manager's retry logic

**Implementation:**
```python
# transcriber.py (after migration)
from .cgpu_jobs import TranscribeJob

class Transcriber:
    def __init__(self, model="medium"):
        self.model = model

    def transcribe(self, audio_path, output_format="srt"):
        job = TranscribeJob(
            audio_path=audio_path,
            model=self.model,
            output_format=output_format
        )
        result = job.execute()
        return result.output_path if result.success else None
```

**Effort:** ~30 min

---

### 4.2 Migrate cgpu_upscaler.py → UpscaleJob

**Current State:**
- 760 lines of complex code
- Embedded pipeline scripts
- Session caching logic
- Multiple fallback paths

**Target:**
- Extract core logic to `UpscaleJob`
- Keep `upscale_with_cgpu()` as thin wrapper
- Remove ~500 lines of redundant code

**Effort:** ~2 hours (careful refactoring needed)

---

### 4.3 Web UI Integration

**Current State:**
- Jobs run inline in Flask routes
- No unified job status tracking
- Manual progress reporting

**Target:**
- Use `CGPUJobManager` for all heavy operations
- Add `/api/jobs` endpoint for status
- Real-time progress updates via SSE or polling

**New Endpoints:**
```
GET  /api/jobs              # List all jobs
GET  /api/jobs/<id>         # Get job status
POST /api/jobs/transcribe   # Submit transcription
POST /api/jobs/upscale      # Submit upscaling
POST /api/jobs/stabilize    # Submit stabilization
```

**Effort:** ~1-2 hours

---

### 4.4 editor.py Integration

**Current State:**
- Calls `cgpu_upscaler.upscale_with_cgpu()` directly
- No batch job support
- Sequential processing

**Target:**
- Submit jobs via `CGPUJobManager`
- Batch stabilization + upscaling
- Parallel job submission where possible

**Effort:** ~1 hour

---

### 4.5 Fix Stale Tests

**Failing Tests:**
- `test_allowed_file` - needs `/data` directory
- `test_beat_detection_mock` - numba compatibility
- `test_footage_clip_dataclass` - API changed
- `test_style_template_loading` - 'dynamic' style missing

**Action:**
- Mock `/data` directory in test setup
- Update `FootageClip` usage to match current API
- Add missing 'dynamic' style or update test

**Effort:** ~30 min

---

## Phase 5: Professional Export (Q1 2025)

### 5.1 OTIO Export Maturity

**Goal:** Ensure `timeline_exporter.py` produces files compatible with:
- DaVinci Resolve 18+
- Adobe Premiere Pro
- Final Cut Pro (via XML)

**Tasks:**
- [ ] Test import in DaVinci Resolve 18/19
- [ ] Add FCP XML export option
- [ ] Handle transitions properly
- [ ] Preserve color metadata

---

### 5.2 Proxy Workflow

**Goal:** Generate low-res proxies for fast preview editing.

**Architecture:**
```
Original Footage (4K)
    ↓
Generate Proxies (720p, cgpu)
    ↓
Edit with Proxies (fast)
    ↓
Final Render (replace proxies with originals)
```

**Implementation:**
- Add `ProxyJob` to cgpu_jobs
- Track proxy ↔ original mapping
- editor.py: use proxies for preview, originals for render

---

## Priority Matrix

| Task | Impact | Effort | Priority |
|------|--------|--------|----------|
| 4.1 Migrate transcriber.py | Medium | Low | **P1** |
| 4.3 Web UI Integration | High | Medium | **P1** |
| 4.5 Fix Stale Tests | Medium | Low | **P1** |
| 4.2 Migrate cgpu_upscaler.py | Medium | High | P2 |
| 4.4 editor.py Integration | Medium | Medium | P2 |
| 5.1 OTIO Maturity | High | Medium | P2 |
| 5.2 Proxy Workflow | Medium | High | P3 |

---

## Metrics for Success

| Metric | Current | Target |
|--------|---------|--------|
| Test pass rate | 64/68 | 68/68 |
| cgpu_upscaler.py LOC | 760 | <200 |
| Job retry logic | Per-module | Centralized |
| Web UI job status | Manual | Real-time |

---

## Next Immediate Action

Start with **Phase 4.1** (migrate transcriber.py) as it's:
- Low effort (~30 min)
- Proves the pattern
- No breaking changes
