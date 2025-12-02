# Over-Engineering Review

Assessment of Montage AI codebase following **DRY** and **KISS** principles.

---

## ✅ What's Good (Not Over-Engineered)

### 1. Web UI
- **Vanilla JS** (no React/Vue/build system)
- **Flask only** (no FastAPI, no async complexity)
- **In-memory job queue** (sufficient for single-instance)
- **Simple file uploads** (no S3/cloud complexity)

### 2. Core Editor
- **Direct FFmpeg calls** (no abstraction layers)
- **librosa for audio** (standard, proven)
- **MoviePy for composition** (simple API)
- **Minimal dependencies**

### 3. Deployment
- **Docker Compose** (easy local dev)
- **Kustomize** (no Helm charts)
- **Simple PVCs** (no operators, no CSI drivers)

### 4. Configuration
- **Environment variables** (no config management system)
- **JSON style templates** (no YAML/TOML layers)

---

## ⚠️ Potential Over-Engineering

### 1. Timeline Exporter

**Current:**
```python
class TimelineExporter:
    def export_timeline(...)
    def _generate_proxy(...)
    def _export_otio(...)
    def _export_edl(...)
    def _export_csv(...)
    def _export_metadata(...)
    def _create_project_package(...)
    def _seconds_to_timecode(...)
```

**Analysis:**
- 8 methods for timeline export
- Project package creation (rarely used?)
- Multiple format exports (do users need all?)

**Recommendation:**
- ✅ **Keep**: OTIO + EDL (most universal)
- ⚠️ **Consider removing**: CSV export (can be generated from JSON)
- ⚠️ **Consider removing**: Project package (users can organize files themselves)

**Simplified:**
```python
class TimelineExporter:
    def export_timeline(...) -> dict:
        """Export OTIO + EDL + metadata.json"""
```

---

### 2. Footage Manager Complexity

**Current:**
```python
class FootagePoolManager:
    # 15+ methods
    def add_scene(...)
    def get_unused_clip(...)
    def mark_as_used(...)
    def get_pool_status(...)
    def reset_pool(...)
    # ... more
```

**Analysis:**
- Very sophisticated clip management
- Complex scoring algorithms
- Story phase tracking

**Recommendation:**
- ✅ **Keep as is** - This is **core value**, not over-engineering
- Story arc awareness is a **differentiator**

---

### 3. Monitoring Module

**Current:**
```python
class EditingMonitor:
    # Real-time decision logging
    def log_clip_selection(...)
    def log_beat_alignment(...)
    def log_phase_transition(...)
    # ... 10+ methods
```

**Analysis:**
- Comprehensive logging
- Used only when VERBOSE=true
- Adds complexity

**Recommendation:**
- ⚠️ **Simplify**: Use Python's standard `logging` module
- Current custom monitor is over-engineered for debug logs

**Simplified:**
```python
import logging

logger = logging.getLogger(__name__)
logger.info(f"Selected clip: {clip_path}")
```

---

### 4. Creative Director LLM Integration

**Current:**
- Keyword matching **before** LLM call (good!)
- Ollama **and** cgpu/Gemini support
- Fallback chain

**Analysis:**
- ✅ Good: Tries keywords first (no unnecessary LLM calls)
- ✅ Good: Dual backend (local + cloud)
- ✅ Not over-engineered

**Recommendation:**
- ✅ **Keep as is**

---

### 5. Proxy Generation

**Current:**
```python
def _generate_proxy(self, source_path):
    # Half-res, H.264, 5Mbps
```

**Analysis:**
- Only used if `GENERATE_PROXIES=true`
- Simple FFmpeg call

**Recommendation:**
- ✅ **Keep** - Optional feature, not over-engineered

---

## 🔧 Refactoring Recommendations

### Priority 1: Simplify Monitoring

**Before (84 lines):**
```python
# monitoring.py - Full custom class
class EditingMonitor:
    def log_clip_selection(self, clip_path, score, reason):
        ...
    def log_beat_alignment(self, beat_idx, timestamp):
        ...
```

**After (use standard logging):**
```python
import logging
logger = logging.getLogger(__name__)

# In editor.py:
logger.info(f"Clip selected: {clip_path}, score={score:.2f}, reason={reason}")
logger.debug(f"Beat alignment: beat #{beat_idx} @ {timestamp:.2f}s")
```

**Impact:**
- Remove ~200 lines of code
- Easier to configure (logging.conf)
- Standard Python practice

---

### Priority 2: Remove CSV Export

**Reason:**
- Metadata JSON already exported
- CSV can be generated from JSON with one-liner:
  ```python
  import pandas as pd
  pd.read_json('metadata.json')['clips'].to_csv('timeline.csv')
  ```

**Remove:**
```python
def _export_csv(self, timeline): ...
```

**Impact:**
- Remove ~60 lines
- Users who need CSV can convert JSON

---

### Priority 3: Consolidate Proxy Logic

**Current:** Proxies in `timeline_exporter.py`

**Better:** Move to `editor.py` as optional preprocessing

**Reason:**
- Proxies are part of video processing, not timeline export
- Tighter integration with main workflow

---

## 📊 Complexity Metrics

| Module | Lines | Classes | Methods | Verdict |
|--------|-------|---------|---------|---------|
| `editor.py` | 1700 | 0 | 15+ | ✅ Complex but necessary |
| `footage_manager.py` | 800 | 5 | 20+ | ✅ Core feature |
| `timeline_exporter.py` | 580 | 2 | 8 | ⚠️ Could simplify |
| `monitoring.py` | 200 | 1 | 12 | ❌ Over-engineered (use logging) |
| `creative_director.py` | 510 | 1 | 8 | ✅ Good balance |
| `web_ui/app.py` | 280 | 0 | 10 | ✅ Simple Flask |

---

## 🎯 Action Plan

### Immediate (Do Now)

1. **Replace monitoring.py with standard logging**
   - Remove custom `EditingMonitor` class
   - Use `logging.getLogger(__name__)`
   - Configure via `logging.basicConfig()`

2. **Remove CSV export from timeline_exporter.py**
   - Keep JSON metadata
   - Document how to convert JSON → CSV

### Medium Priority (Next Sprint)

3. **Simplify project package creation**
   - Remove `_create_project_package()` method
   - Users can organize files themselves

4. **Consolidate proxy generation**
   - Move from timeline_exporter to editor
   - Make it a preprocessing step

### Low Priority (Future)

5. **Consider removing EDL export**
   - OTIO is more modern
   - EDL is 1970s tech
   - But: EDL is universally compatible → **keep for now**

---

## 🔍 Dependency Analysis

### Necessary
- ✅ `moviepy` - Core video editing
- ✅ `librosa` - Beat detection
- ✅ `opencv-python-headless` - Frame analysis
- ✅ `Flask` - Web UI
- ✅ `OpenTimelineIO` - NLE export
- ✅ `openai` - cgpu/Gemini API

### Questionable
- ⚠️ `psutil` - System monitoring (monitoring.py)
  - **Keep**: Useful for resource tracking

### Not Over-Engineered
- ✅ All dependencies are justified

---

## 📝 Code Patterns to Maintain

### Good: Direct FFmpeg Calls

```python
subprocess.run([
    "ffmpeg", "-i", input_path,
    "-vf", "scale=960:540",
    output_path
])
```

**Why good:** No abstraction layers, easy to debug.

### Good: Simple Configuration

```python
STABILIZE = os.environ.get("STABILIZE", "false").lower() == "true"
```

**Why good:** Environment variables, no config framework.

### Good: In-Memory State

```python
jobs = {}  # Simple dict for job tracking
```

**Why good:** No Redis/database for single-instance use.

---

## Conclusion

**Overall Assessment:** 🟢 **Not Over-Engineered**

**Strong Points:**
- KISS principles followed in web UI
- Minimal dependencies
- Direct FFmpeg usage
- No unnecessary abstractions

**Improvement Areas:**
- Monitoring module (use standard logging)
- Timeline export (remove CSV, simplify project packaging)

**Final Score:** 8.5/10

Montage AI is **lean and focused**. The complexity in `footage_manager.py` and `editor.py` is **justified** because it's the core value proposition (story arc awareness, beat-sync).

---

## Next Steps

1. Implement Priority 1 changes (monitoring → logging)
2. Test without CSV export
3. Document refactoring decisions in CHANGELOG
4. Re-run tests to ensure nothing breaks
