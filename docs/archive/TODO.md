# TODO - Manual Tasks for User

Tasks that require your decision or manual action.

---

## 🚀 High Priority (Do First)

### 1. Test Web UI

**Action Required:**
```bash
# Start web UI
make web

# Open browser: http://localhost:5000
# Upload test videos and music
# Create a montage
# Verify it works
```

**Decision:** Does the web UI meet your needs? Any missing features?

---

### 2. Create Demo Video

**Why:** Huge UX improvement for new users (see `docs/comparison.md`)

**Action Required:**
1. Record 2-3 minute screencast showing:
   - Upload videos/music
   - Configure style
   - Create montage
   - Download result
2. Upload to YouTube
3. Add thumbnail to README.md

**Example:**
```markdown
## Demo

[![Watch Demo](thumbnail.jpg)](https://youtube.com/watch?v=...)
*50 clips → 2min montage in 30 seconds*
```

---

### 3. Test Timeline Export with DaVinci Resolve

**Action Required:**
```bash
# Create montage with timeline export
./montage-ai.sh run hitchcock --export-timeline

# Try importing the .otio file into:
# - DaVinci Resolve
# - Adobe Premiere Pro (use .edl)
# - Final Cut Pro

# Document results in docs/timeline_export.md
```

**Decision:** Does timeline export work correctly? Any issues?

---

### 4. Review Model Documentation

**Action Required:**

Read `docs/OVER_ENGINEERING_REVIEW.md` and decide:

1. **Remove monitoring.py?** Replace with standard Python logging?
2. **Remove CSV export?** Keep only JSON metadata?
3. **Simplify project package?** Remove auto-packaging feature?

**Decision:** Which simplifications should we implement?

---

## 📊 Medium Priority

### 5. Sample Footage Repository

**Why:** Users need demo content to try the tool (Low Hanging Fruit #1)

**Action Required:**

1. Find 5-10 Creative Commons videos (e.g., from Pexels)
2. Option A: Add to Git LFS
3. Option B: Create download script:

```bash
# demo-data.sh
curl -o data/input/beach.mp4 "https://..."
curl -o data/input/sunset.mp4 "https://..."
curl -o data/music/demo.mp3 "https://..."
```

4. Add `make demo` command to Makefile

---

### 6. Add Comparison Table to README

**Action Required:**

Copy comparison content from `docs/comparison.md` to README.md.

Add section:
```markdown
## How Montage-AI Compares

| Feature | Montage-AI | Frame | Adobe Premiere |
|---------|------------|-------|----------------|
| Beat-Sync | ✅ Auto | ❌ | ⚠️ Manual |
| ... | ... | ... | ... |
```

---

### 7. GitHub Repository Setup

**Action Required:**

1. **Add Topics:**
   - `video-editing`
   - `ai`
   - `llm`
   - `beat-sync`
   - `ffmpeg`
   - `self-hosted`

2. **Social Preview Image:**
   - Create screenshot of web UI or example output
   - Upload in GitHub Settings → Social Preview

3. **Create v0.3.0 Release:**
   ```bash
   git tag -a v0.3.0 -m "Release v0.3.0: Web UI, Timeline Export, cgpu"
   git push origin v0.3.0
   ```

4. **Pre-built Docker Images:**
   ```bash
   make release VERSION=v0.3.0
   ```

---

## 🔍 Low Priority (Optional)

### 8. Create docs/models.md

**Action Required:**

Document model decisions:

```markdown
# AI Models Used

## Creative Director LLM

### Llama 3.1 70B (Default)
- **Why:** Best open-source reasoning
- **Tested:** 100 prompts, 95% JSON accuracy
- **RAM:** 40GB (quantized)
- **Alternative:** Gemini 2.0 Flash via cgpu (98% accuracy, faster)

## Video Upscaling

### Real-ESRGAN-ncnn-vulkan
- **Why:** SOTA quality (CVPR 2021)
- **Model:** realesr-animevideov3
- **Speed:** ~20 FPS on T4 GPU (cgpu)
- **Alternative:** Topaz Video AI ($300, better quality)

## Beat Detection

### librosa 0.10+
- **Why:** Accuracy > Speed
- **Tested:** Better than Madmom for non-jazz
- **Trade-off:** Pure Python (slower)
- **Performance:** 30s audio → 2-3s analysis
```

**Sources:**
- Use research from `docs/comparison.md`
- Add benchmarks if available

---

### 9. Implement Refactorings from OVER_ENGINEERING_REVIEW.md

**If you agree with recommendations:**

1. **Replace monitoring.py:**
   ```bash
   # Remove custom monitor
   rm src/montage_ai/monitoring.py

   # Update editor.py to use standard logging
   import logging
   logger = logging.getLogger(__name__)
   ```

2. **Remove CSV export:**
   ```python
   # In timeline_exporter.py, remove:
   def _export_csv(self, timeline): ...
   ```

3. **Update tests** after refactoring

---

### 10. Add Integration Tests

**Action Required:**

Create `tests/test_integration.py`:

```python
def test_full_montage_workflow():
    """Test complete montage creation."""
    # 1. Place test videos in data/input/
    # 2. Place test music in data/music/
    # 3. Run editor
    # 4. Verify output exists
    # 5. Check video duration
    pass

@pytest.mark.slow
def test_timeline_export():
    """Test OTIO/EDL export."""
    # Create montage with export_timeline=True
    # Verify .otio, .edl, .csv files exist
    # Validate OTIO structure
    pass
```

---

## 🧪 Testing Checklist

Before considering v1.0 stable:

- [ ] Web UI works on localhost
- [ ] Web UI works in Kubernetes
- [ ] Timeline export imports correctly into DaVinci Resolve
- [ ] Timeline export imports correctly into Premiere Pro
- [ ] cgpu integration works
- [ ] All unit tests pass (`make test-unit`)
- [ ] Docker build succeeds (`make build`)
- [ ] Kubernetes deployment succeeds (`make deploy`)

---

## 📝 Documentation Checklist

- [ ] README has comparison table
- [ ] README has demo video link
- [ ] docs/models.md created (model decisions documented)
- [ ] docs/web_ui.md reviewed and accurate
- [ ] docs/timeline_export.md tested with real NLE software
- [ ] All environment variables documented in docs/configuration.md

---

## 🎯 Release Checklist (v0.3.0)

When ready for public release:

- [ ] All High Priority TODOs completed
- [ ] Demo video published
- [ ] Sample footage available
- [ ] Pre-built Docker images on ghcr.io
- [ ] GitHub topics added
- [ ] Social preview image set
- [ ] v0.3.0 git tag created
- [ ] Announcement on Reddit/HN/Twitter

---

## 🔗 Useful Links

- **Comparison Research:** `docs/comparison.md`
- **Over-Engineering Review:** `docs/OVER_ENGINEERING_REVIEW.md`
- **Web UI Docs:** `docs/web_ui.md`
- **Timeline Export:** `docs/timeline_export.md`
- **Features:** `docs/features.md`
- **Architecture:** `docs/architecture.md`

---

## Questions to Answer

1. **Web UI:** Keep simple Flask or add Redis/Celery for production?
2. **Monitoring:** Keep custom class or switch to standard logging?
3. **CSV Export:** Remove it (use JSON only)?
4. **Demo Data:** Git LFS or download script?
5. **Model Docs:** Document all model decisions or keep minimal?

---

**Last Updated:** 2025-12-02
**By:** Claude (AI Assistant)
