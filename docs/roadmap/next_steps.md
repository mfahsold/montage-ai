# Next Steps & Roadmap (Re-planned)

Based on the analysis and current codebase state (Jan 2026).

## 1. Completed Tricky Tasks

- [x] **EDL/XML Export Integration**: `TimelineExporter` is now integrated into the main `MontageBuilder` pipeline. Every run with `--export` enabled will generate OTIO, EDL, and XML files for professional post-production.

## 2. Low Hanging Fruits (Scope-near)

These are high-value, low-effort tasks to solidify the "AI-Rough-Cut" positioning.

### A. Pacing Curves & Energy Matching

*Current State*: `audio_analysis.py` detects "low/medium/high" energy sections.
*Task*:

- [ ] **Refine Music Analysis**: Explicitly label sections as Intro, Build, Drop, Outro based on position and energy.
- [ ] **Energy Matching**: Enforce "High Energy Video" on "Drop" sections and "Low Energy/Scenic" on "Intro".

### B. Smart Clip Selection Signals

*Current State*: `ClipMetadata` tracks energy, action, shot type.
*Task*:

- [ ] **Face Detection Signal**: Add basic face detection (OpenCV) to `scene_analysis.py` and boost score for clips with faces in "Vlog" style.
- [ ] **Visual Novelty**: Penalize clips that look too similar to the previous one (using color histogram comparison).

### C. Preset/Template Expansion

*Current State*: Basic styles exist.
*Task*:

- [ ] **Add "Vlog" Template**: Focus on face-centric clips, moderate pacing.
- [ ] **Add "Sport" Template**: Focus on high-action, fast cuts, high energy music sections.

### D. Web-UI Improvements

*Current State*: Basic Web UI exists.
*Task*:

- [ ] **Preview Mode**: Generate a low-res (360p) preview first for quick feedback before full render.

## 3. Strategic Scope Adjustments

**Confirmed Scope:**

- **Montage AI** = Automatic Beat-Sync-Editing + Smart Selection + Enhancement.
- **Goal**: "AI-Rough-Cut" tool, not a full NLE.

**Deprioritized (as per analysis):**

- Full NLE Timeline Editing (use Export instead).
- Generative Video (focus on polishing existing pixels).
- Cloud-only Rendering (keep local option).
