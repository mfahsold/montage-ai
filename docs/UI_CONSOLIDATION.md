# UI Consolidation & Vision

**Date:** January 2026
**Status:** Strategic Update (PwC-Style)
**Related:** [STRATEGY.md](STRATEGY.md)

---

## Vision: "Outcome Studio"

We are moving from a "Toggle Graveyard" (feature-centric) to an **"Outcome Studio"** (result-centric). The UI must guide the user through clear workflows with visible story logic and style-defining motion.

### Core Principles
1.  **Outcome-First:** Users choose *what* they want to make (Short, Montage, Clean Transcript), not *how* (settings).
2.  **Preview-First:** Always show a rough draft immediately. Final render is a deliberate "Export" action.
3.  **Visible Logic:** Show *why* a cut happened (Beat markers, Energy curve).
4.  **Cyber-NLE Aesthetic:** High-contrast, kinetic, "pro-tool" feel, but accessible.

---

## The "Outcome Hub" (Landing Page)

**Goal:** Replace the dashboard of checkboxes with three distinct doors.

```
┌─────────────────────────────────────────────────────────────┐
│  Montage AI                                                 │
│  [Status: Ready] [Cloud: On]                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  What are we making today?                                  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 📝           │  │ 📱           │  │ 🎬           │       │
│  │ Transcript   │  │ Shorts       │  │ Montage      │       │
│  │ Editor       │  │ Studio       │  │ Creator      │       │
│  │              │  │              │  │              │       │
│  │ "Edit text,  │  │ "Viral clips │  │ "Music-sync  │       │
│  │ get video."  │  │ in minutes."  │  │ storytelling"│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                             │
│  Recent Projects:                                           │
│  • Interview_01 (Draft)                                     │
│  • Vlog_Final (Exported)                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## UI/UX Recommendations

### 1. Transcript-First Tri-Pane
For the **Transcript Editor** workflow:
*   **Left:** Video Player (360p Preview).
*   **Center:** Interactive Transcript (Word-level selection, Speaker tags).
*   **Right:** Beat/Story Timeline (Vertical or Horizontal).
    *   **Live Markers:** Show *why* a cut is suggested (e.g., "Long Pause", "Filler Word").

### 2. Kinetic Beat Timeline
For the **Montage Creator** workflow:
*   **Energy Curve:** Visual overlay of audio energy.
*   **Beat Ticks:** Rhythmic markers on the timeline.
*   **Story Arc:** Color-coded phases (Intro, Build-up, Climax, Outro).
*   **Motion:** Subtle animations when the playhead crosses a beat.

### 3. Shorts Studio "Phone Rig"
For the **Shorts Studio** workflow:
*   **Crop Path Overlay:** A semi-transparent 9:16 rectangle moving over the 16:9 source.
*   **Keyframe Handles:** Allow users to drag the crop center to correct the AI.
*   **Safe Zones:** Toggleable overlays for TikTok, Reels, Shorts UI elements.
*   **Caption Composer:** Live preview of caption styles (Bold, Karaoke) directly on the video.

### 4. "Preview vs Final" Ritual
*   **Default:** All edits happen in "Preview Mode" (low res, instant).
*   **Comparison:** A "Compare" button (slider) to see Raw vs. Graded/Stabilized.
*   **Export:** A distinct modal for "Final Render" where heavy compute (Upscale, Clean Audio) is applied.

### 5. Typo & Motion Guidelines
*   **Typography:** Monospaced fonts for data/timecodes (JetBrains Mono), strong Sans-Serif for headlines (Inter/Roboto).
*   **Motion:** "Click-to-Cut" sound effects (subtle mechanical clicks). Fast transitions (0.2s) for UI elements.
*   **Color:** Dark mode default. Neon accents for "AI Magic" (Cyan/Magenta).

---

## Consolidation Steps (Cleanup)

1.  **Toggle Friedhof:** Remove individual flags for `stabilize`, `upscale`, `cgpu`. Bundle them into **Quality Profiles** (Preview, Standard, High, Master).
2.  **AI Director:** Group all LLM-related settings (Prompt, Creativity, Style) into a single "AI Director" panel.
3.  **Cloud Boost:** A single toggle "Enable Cloud Acceleration" that manages all remote offloading logic.

---

## Implementation Plan

*   **Phase 1:** Implement the "Outcome Hub" landing page.
*   **Phase 2:** Build the "Phone Rig" overlay for Shorts.
*   **Phase 3:** Refine the Transcript Tri-Pane.
