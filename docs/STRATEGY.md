# Montage AI – Strategic Product Document

**Version:** 2.4 (PwC-Style Strategy Update)
**Date:** January 4, 2026
**Classification:** Public
**Status:** Active

---

## Executive Summary

**Core Philosophy:** "Polish, don't generate."
**Differentiation:** Local-first + Pro-handoff.
**Current Focus:** Tangible speed and control ("Preview-First").

Montage AI is not a generative video tool. It is a post-production assistant that refines existing footage. Our goal for Q1 2026 is to productize the **Transcript Editor**, deliver **Shorts Studio 2.0**, and stabilize **Pro-Handoff** and **Audio-Polish**.

### Strategic Priorities (Q1)
1.  **Transcript Editor Productization:** Move from prototype to a first-class editing surface.
2.  **Shorts Studio 2.0:** Smart-Reframe v2 and native caption styles.
3.  **Preview-First Pipeline:** Immediate feedback loops (360p) before final rendering.
4.  **Pro-Handoff:** Reliable OTIO export to DaVinci Resolve/Premiere.

### Scope
*   **In-Scope:** Rough-Cut, Transcript-Editing, Shorts/Vertical, Captions, Smart-Reframe, Audio-Polish, OTIO/EDL-Handoff.
*   **Out-of-Scope:** Generative Video, Full NLE replacement (multitrack compositing, VFX), Social Hosting.

### UI Vision
Transition from a "Toggle Graveyard" to an **"Outcome Studio"**.
*   Clear workflows (Transcript, Shorts, Montage).
*   Visible Story/Beat logic.
*   Style-defining motion and "Cyber-NLE" aesthetics.

---

## Implementation Snapshot (Codebase Reality)

| Component | Status | Alignment |
| :--- | :--- | :--- |
| **ffmpeg_config.py** | ✅ Ready | Has Preview-Preset (360p, ultrafast) + GPU-Auto-Encoding. Perfect basis for Preview-First UX. |
| **video_metadata.py** | ✅ Ready | Determines Output-Profiles via weighted medians; fits ideal Quality-Profile automation. |
| **transcript.html** | ⚠️ Partial | Delivers Word-Level-Edits, Filler-Removal, Export-Buttons. Preview-Flow is currently a stub. |
| **shorts.html** | ⚠️ Partial | Has Safe-Zones, Caption-Style-Preview, Reframe-Modes. Highlights/Render logic is partly stubbed. |
| **Documentation** | ❌ Inconsistent | `STRATEGY.md` vs. actual UI state creates expectation gaps. |

---

## Market Signals & Benchmarks

1.  **Quality = Trust:** Video is mainstream; quality profiles and audio polish are mandatory [1].
2.  **Text-Based Editing:** Standard feature for market leaders (Descript, Adobe) [2][3].
3.  **Short-Form Strategy:** YouTube Shorts and TikTok require dedicated vertical workflows [5][6].
4.  **AI-Reframe:** A key competitive advantage for repurposing tools (Opus Clip) [4].
5.  **Clean Audio:** Pro NLEs set the expectation for Voice Isolation [7].
6.  **OTIO Standard:** The industry standard for timeline handoff [8].

---

## Focus Features (Q1 Priorities)

### 1. Transcript Editor Productization
*   **Live-Preview (360p):** Immediate playback of edits.
*   **Word-Level-Cut-List:** Apply/Undo stack for precise editing.
*   **Filler-Removal:** Auto-detect and remove "um", "uh" with Speaker Tags.
*   **Pro-Export:** OTIO/EDL export directly from text edits.

### 2. Shorts Studio 2.0
*   **Smart-Reframe v2:** Subject Tracking + Motion Smoothing.
*   **Caption-Styles:** Real styles (TikTok/Bold/Karaoke) with Live-Preview.
*   **Highlight-Detection:** MVP with Review-Cards.

### 3. Preview-First Pipeline
*   **Default-Preview:** Starts immediately after upload. Clear ETA/Progress.
*   **"Final Render":** A separate, deliberate step.
*   **Upscale:** Only applied in High/Master profiles via Real-ESRGAN [9].

### 4. Pro Handoff Pack
*   **OTIO-Export:** Compatible with DaVinci/Premiere.
*   **Proxies:** Automatic generation.
*   **Relink-README:** Auto-generated guide for importing.
*   **Smoke Tests:** Verified imports in target NLEs.

### 5. Audio-Polish
*   **Clean Audio Toggle:** Voice Isolation + Denoise + Fallback.
*   **SNR-Check:** Quality assurance metric.

---

## Consolidation & Cleanup

*   **AI Director:** Bundle LLM toggles under a single "AI Director" flag. Move "Creative Loop" to an Advanced drawer.
*   **UI Reduction:** Deprecate Legacy/v2 variants. One "Outcome Hub" + three distinct workflows.
*   **Style Presets:** Curate the catalog to core styles. Move the rest to a "Community Pack".
*   **Silence Removal:** Treat as a utility baseline (like auto-editor), not a differentiator [10].
*   **Cloud Options:** Single "Cloud Boost" toggle instead of granular flags.

---

## UI/UX Vision: "Hip & Innovative"

*   **Transcript-First Tri-Pane:** Video + Text + Beat/Story-Timeline with Live-Markers.
*   **Kinetic Beat Timeline:** Energy-Curve, Beat-Ticks, and Story-Arc phases as overlays.
*   **Shorts-Studio "Phone-Rig":** Crop-Path-Overlay with Keyframe-Handles, Safe-Zone-Presets.
*   **"Preview vs Final" Ritual:** Clear state distinction, comparison split, fast A/B loops.
*   **Typo & Motion:** Strong headlines, subtle motion-reveals, UI-Sounding (Click-to-Cut).

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
