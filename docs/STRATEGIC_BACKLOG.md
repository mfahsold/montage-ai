# Montage AI – Strategic Backlog (Q1 2026)

**Based on:** [STRATEGY.md](STRATEGY.md) (PwC-Style Update)
**Timeline:** 90 Days (Weeks 0-12)
**Focus:** Polish, Don't Generate.

---

## Phase 1: Foundation & Preview-First (Weeks 0–4)

**Theme:** "Make it usable and fast."
**KPIs:** Time-to-First-Preview < 2-3 min, Preview-Success-Rate > 95%.

### Epic 1: Transcript Editor Productization
*Goal: Turn the prototype into a reliable editing surface.*
- [ ] **Story 1.1: Live Preview Flow**
  - Implement 360p "ultrafast" preview generation triggered by transcript edits.
  - Ensure < 5s latency between "Apply Edits" and playback start.
- [ ] **Story 1.2: Word-Level Cut List**
  - UI for selecting/striking words.
  - Backend logic to translate word indices to time ranges (EDL).
  - "Undo" stack for non-destructive editing.
- [ ] **Story 1.3: Filler Removal & Speaker Tags**
  - Detect filler words (um, uh) via Whisper.
  - UI toggle to "Strike all fillers".
  - Visual speaker change indicators in the text stream.
- [ ] **Story 1.4: Text-Based Export**
  - Export button in Transcript view.
  - Generate OTIO/EDL directly from the text selection state.

### Epic 2: Preview-First Pipeline
*Goal: Immediate feedback loop.*
- [ ] **Story 2.1: Default Preview Generation**
  - Automatically start 360p render immediately after upload/ingest.
  - Show clear ETA/Progress bar in the UI header.
- [ ] **Story 2.2: "Final Render" Separation**
  - Distinct UI flow for "Export Master".
  - Only apply heavy effects (Upscale, Stabilization) in this step.
- [ ] **Story 2.3: Telemetry**
  - Instrument "Time to Preview" and "Edit Latency" metrics.

---

## Phase 2: Shorts Studio 2.0 (Weeks 5–8)

**Theme:** "Vertical Video that feels native."
**KPIs:** Shorts-Creation-Cycle < 10 Min, Reframe-Accuracy > 90%.

### Epic 3: Smart Reframe v2
*Goal: Broadcast-quality vertical crops.*
- [ ] **Story 3.1: Subject Tracking & Smoothing**
  - Upgrade `SmartReframer` to use continuous subject tracking (not just per-frame face detection).
  - Implement motion smoothing (Kalman filter or similar) to prevent jittery camera moves.
- [ ] **Story 3.2: Crop Path Overlay**
  - "Phone Rig" UI: Show the 9:16 window moving over the 16:9 source in real-time.
  - Allow manual keyframe overrides for the crop center.

### Epic 4: Caption Styles & Highlights
*Goal: Social-ready output without external tools.*
- [ ] **Story 4.1: Live Caption Styles**
  - Implement presets: "TikTok" (Classic), "Bold" (Impact), "Karaoke" (Active word highlight).
  - Render CSS-based preview in the web player.
- [ ] **Story 4.2: Highlight Detection MVP**
  - "Magic Moments" sidebar suggesting 15-60s clips based on audio energy/laughter/keywords.
  - Review Cards: "Keep", "Discard", "Edit" for each suggestion.

---

## Phase 3: Pro Polish & Handoff (Weeks 9–12)

**Theme:** "Trust the output."
**KPIs:** Audio-Improvement-Rate > 70%, Export-Success > 95%.

### Epic 5: Pro Handoff Pack
*Goal: Seamless integration with NLEs.*
- [ ] **Story 5.1: Robust OTIO Export**
  - Ensure OTIO files open correctly in DaVinci Resolve and Premiere Pro.
  - Include relative paths to source media.
- [ ] **Story 5.2: Proxy Generation**
  - Option to generate and link ProRes Proxy files.
- [ ] **Story 5.3: Relink README**
  - Auto-generate a `HOW_TO_RELINK.md` in the export folder with instructions.
- [ ] **Story 5.4: Import Smoke Tests**
  - Automated tests to verify generated OTIO files against NLE import specs.

### Epic 6: Audio Polish
*Goal: "Clean Audio" standard.*
- [ ] **Story 6.1: "Clean Audio" Toggle**
  - Single switch that activates Voice Isolation + Denoise.
  - Implement SNR (Signal-to-Noise Ratio) check before/after to verify improvement.
- [ ] **Story 6.2: Fallback Strategy**
  - If artifacts are detected (or confidence low), blend with original audio or reduce effect intensity.

---

## Out of Scope (Explicit)
- Generative Video (Pixels from scratch).
- Full NLE features (Multitrack compositing, complex VFX).
- Social Hosting/Distribution APIs.
- After-Effects-style Motion Graphics.
