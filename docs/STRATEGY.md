# Montage AI – Strategic Product Document

**Version:** 2.7 (Feature Complete)
**Date:** January 6, 2026
**Classification:** Public
**Status:** Active

---

## Executive Summary

**Core Philosophy:** "We do not generate pixels. We polish them."
**Differentiation:** Local-first AI video editing with pro export.
**Current Focus:** Transcript editing, shorts generation, professional handoff.

Montage AI is a post-production assistant, not a generative video tool. It refines existing footage through AI-assisted editing workflows.

### Strategic Priorities (Q1 2026)

1. **Transcript Editor:** Production-ready text-based video editing
2. **Shorts Studio:** Auto-reframe to 9:16 with smart tracking
3. **Pro Handoff:** Reliable OTIO/EDL export to DaVinci, Premiere
4. **Quality Profiles:** Preview-first workflow for fast iteration

### Scope

- **In-Scope:** Text-based editing, beat-sync montages, vertical video, OTIO/EDL export, quality profiles
- **Out-of-Scope:** Generative video, full NLE replacement, social media hosting

### UI Vision

Three focused workflows:
- **Montage Creator:** Beat-sync editing with 7+ styles
- **Shorts Studio:** Vertical video with auto-reframe
- **Transcript Editor:** Text-based editing with live preview

---

## Implementation Status (January 2026)

### ✅ Production-Ready Components

| Component | Evidence | Status |
|-----------|----------|--------|
| **Beat Detection** | `audio_analysis.py`, 419 tests | ✅ Live |
| **Quality Profiles** | `env_mapper.py`, `config.py` | ✅ Live |
| **GPU Auto-Detection** | `ffmpeg_config.py` | ✅ Live |
| **Auto-Reframe** | `auto_reframe.py` + tests | ✅ Live |
| **Style Templates** | `style_templates/` (7 styles) | ✅ Live |
| **OTIO Export** | `timeline_exporter.py`, 17 tests | ✅ Live |
| **SSE Streaming** | Real-time progress | ✅ Live |
| **Docker + K3s** | Multi-arch deployment | ✅ Live |

### ✅ Fully Polished Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Transcript Editor** | ✅ 100% | Word-level edits, live preview, OTIO export |
| **Shorts Studio** | ✅ 100% | Phone preview, safe zones, captions |
| **Audio Polish** | ✅ 100% | SNR measurement, before/after reporting |
| **Caption Burn-In** | ✅ 100% | 5 styles (Karaoke, Bold, Minimal, TikTok, Cinematic) |

### ✅ Recently Implemented (January 6, 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| **Telemetry/Metrics** | ✅ Live | `telemetry.py` - Time-to-preview, success rates |
| **NLE Import Smoke Tests** | ✅ Live | `tests/integration/test_nle_export.py` - OTIO/EDL validation |
| **Subject Tracking v2** | ✅ Live | Kalman filter smoothing in `auto_reframe.py` |
| **Audio Fallback Strategy** | ✅ Live | `audio_fallback.py` - Artifact detection + blending |
| **VideoAnalysisEngine** | ✅ Live | `video_analysis_engine.py` - Single-pass pluggable analyzers |
| **Audio-Aware Shorts** | ✅ Live | Energy-based highlight detection in `shorts_workflow.py` |
| **Enhancement Options (Shorts)** | ✅ Live | Stabilize/Upscale toggles in UI |

---

## UI/UX Status (Post-Phase 2)

| Route | Template | Status |
| :--- | :--- | :--- |
| `/` | `index_strategy.html` | ✅ Primary landing page |
| `/montage` | (redirect to `/`) | ✅ Consolidated |
| `/shorts` | `shorts.html` | ✅ Distinct workflow |
| `/transcript` | `transcript.html` | ✅ Distinct workflow |

**Status:** ✅ Fully consolidated. 3 distinct workflows: Montage (default), Shorts, Transcript.

---

## Market Signals & Benchmarks

1. **Quality = Trust:** Video is mainstream; quality profiles and audio polish are mandatory [1].
2. **Text-Based Editing:** Standard feature for market leaders (Descript, Adobe) [2][3].
3. **Short-Form Strategy:** YouTube Shorts and TikTok require dedicated vertical workflows [5][6].
4. **AI-Reframe:** A key competitive advantage for repurposing tools (Opus Clip) [4].
5. **Clean Audio:** Pro NLEs set the expectation for Voice Isolation [7].
6. **OTIO Standard:** The industry standard for timeline handoff [8].

---

## Focus Features (Q1 Priorities)

### 1. Transcript Editor Productization

- **Live-Preview (360p):** Immediate playback of edits. ✅ Implemented
- **Word-Level-Cut-List:** Apply/Undo stack for precise editing. ✅ Implemented
- **Filler-Removal:** Auto-detect and remove "um", "uh" with Speaker Tags. ✅ Implemented
- **Pro-Export:** OTIO/EDL export directly from text edits. ✅ Implemented

### 2. Shorts Studio 2.0

- **Smart-Reframe v2:** Subject Tracking + Motion Smoothing. ⚠️ Base tracking works, v2 smoothing TBD
- **Caption-Styles:** Real styles (TikTok/Bold/Karaoke) with Live-Preview. ✅ Implemented
- **Highlight-Detection:** MVP with Review-Cards. ⚠️ API exists, UI needs wiring

### 3. Preview-First Pipeline

- **Default-Preview:** Starts immediately after upload. Clear ETA/Progress. ✅ Implemented
- **"Final Render":** A separate, deliberate step. ✅ Implemented
- **Upscale:** Only applied in High/Master profiles via Real-ESRGAN [9]. ✅ Implemented

### 4. Pro Handoff Pack

- **OTIO-Export:** Compatible with DaVinci/Premiere. ✅ Verified (17 tests)
- **Proxies:** Automatic generation. ✅ Implemented
- **Relink-README:** Auto-generated guide for importing. ✅ Implemented
- **Smoke Tests:** Verified imports in target NLEs. 🔴 Not implemented

### 5. Audio-Polish

- **Clean Audio Toggle:** Voice Isolation + Denoise + Fallback. ✅ Implemented (`/api/audio/clean`)
- **SNR-Check:** Quality assurance metric. ⚠️ Basic, needs calibration

---

## Consolidation & Cleanup

- **AI Director:** Bundle LLM toggles under a single "AI Director" flag. Move "Creative Loop" to an Advanced drawer. ⚠️ Partial
- **UI Reduction:** Deprecate Legacy/v2 variants. One "Outcome Hub" + three distinct workflows. ⚠️ In progress
- **Style Presets:** Curate the catalog to core styles. Move the rest to a "Community Pack". ✅ Done
- **Silence Removal:** Treat as a utility baseline (like auto-editor), not a differentiator [10]. ✅ Done
- **Cloud Options:** Single "Cloud Boost" toggle instead of granular flags. ✅ Implemented

---

## UI/UX Vision: "Hip & Innovative"

- **Transcript-First Tri-Pane:** Video + Text + Beat/Story-Timeline with Live-Markers. ⚠️ Partial (2-pane now)
- **Kinetic Beat Timeline:** Energy-Curve, Beat-Ticks, and Story-Arc phases as overlays. ✅ Implemented (Transcript Editor)
- **Shorts-Studio "Phone-Rig":** Crop-Path-Overlay with Keyframe-Handles, Safe-Zone-Presets. ✅ Implemented
- **"Preview vs Final" Ritual:** Clear state distinction, comparison split, fast A/B loops. ⚠️ Partial
- **Typo & Motion:** Strong headlines, subtle motion-reveals, UI-Sounding (Click-to-Cut). ⚠️ Basic

---

## Remaining Q1 Work (Weeks 5-12)

### ✅ Completed (Week 5 - January 4, 2026)

1. ✅ **Wire Shorts Highlights UI to API** — Connected to real endpoint, mock data removed
2. ✅ **Optimize Transcript Preview Latency** — Added zerolatency, multithreading, 30s cap
3. ✅ **Deprecate `/v2` Route** — Redirects to `/`, README updated
4. ✅ **Add Beat Timeline to Transcript** — Collapsible energy/beats pane with click-to-seek

### Should Complete (Next)

5. **Subject Tracking v2** — Kalman filter smoothing for reframe paths
6. **Telemetry Instrumentation** — Time-to-preview, export success metrics
7. **Audio SNR Calibration** — More accurate before/after measurements

### Nice to Have

8. **NLE Import Smoke Tests** — Automated DaVinci/Premiere verification
9. **Audio Fallback Strategy** — Artifact detection and blending

---

## 90-Day Plan (Compressed)

| Phase | Weeks | Focus | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **1** | 0–4 | **Foundation** | Transcript-Editor Preview-Flow, Export stabilization, Telemetry. |
| **2** | 5–8 | **Shorts 2.0** | Reframe v2, Caption-Styles, Highlight-MVP, UI Polish. |
| **3** | 9–12 | **Pro Polish** | Pro-Handoff Pack, Audio-Polish, Performance Targets. |

---

## Core KPIs

*   **Time-to-First-Preview:** < 2–3 Minutes.
*   **Preview-Success-Rate:** > 95%.
*   **Transcript-Editing-Adoption:** > 40% of sessions.
*   **Export-Success:** > 95%.
*   **Shorts-Creation-Cycle:** < 10 Minutes.
*   **Reframe-Accuracy:** > 90%.
*   **Audio-Improvement-Rate:** > 70% (SNR Check).

---

## Risks & Mitigation

*   **Performance/Hardware:** Mitigate via Preview-First + Proxy Path + GPU Fallback.
*   **LLM Reliability:** Mitigate via Guardrails + Deterministic Defaults.
*   **UI Complexity:** Mitigate via Outcome-Flows + Progressive Disclosure.
*   **Cloud Availability:** Mitigate via Hard Fallback Strategy + Clear UI Communication.

---

## References

[1] [Wyzowl Video Marketing Statistics](https://www.wyzowl.com/video-marketing-statistics/)
[2] [Descript Video Editing](https://www.descript.com/video-editing)
[3] [Adobe Text-Based Editing](https://helpx.adobe.com/premiere/desktop/edit-projects/edit-video-using-text-based-editing/transcribe-video.html)
[4] [Opus Clip AI Reframe](https://www.opus.pro/ai-reframe)
[5] [YouTube Shorts Getting Started](https://support.google.com/youtube/answer/10059070?hl=en)
[6] [YouTube Creation Tools](https://support.google.com/youtube/answer/2734796?hl=en)
[7] [DaVinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve)
[8] [OpenTimelineIO](https://opentimelineio.readthedocs.io/en/stable/)
[9] [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
[10] [Auto-Editor](https://github.com/WyattBlue/auto-editor)
[11] [OpenAI Whisper](https://github.com/openai/whisper)
