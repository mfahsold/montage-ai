# Montage AI – UI Design Brief & Style Guide

**Version:** 1.0
**Date:** January 2026
**Theme:** "Cyber-NLE" (Professional, Kinetic, Dark, Precision)

---

## 1. Core Philosophy
**"Outcome-Studio"**
We are not a settings panel; we are a creative cockpit. The UI should feel like a high-end instrument—responsive, precise, and slightly futuristic, but grounded in professional utility.

*   **Dark Mode Default:** Video content pops against deep grays.
*   **Kinetic Feedback:** Every action has a reaction (micro-interactions).
*   **Data Density:** High. Use monospace for data, sans-serif for UI.
*   **Visual Hierarchy:** Content > Timeline > Controls.

---

## 2. Color Palette

### Surface & Background
*   **Void Black:** `#0a0a0a` (Main background)
*   **Panel Gray:** `#161616` (Card/Panel background)
*   **Border Gray:** `#2a2a2a` (Separators, Borders)
*   **Input Gray:** `#1f1f1f` (Form fields)

### Accents (Functional)
*   **Cyber Yellow (Primary/Action):** `#F4D03F` (Primary Buttons, Active States, Playhead)
    *   *Usage:* "Do this", "Active Selection", "The current moment".
*   **Signal Blue (Info/Selection):** `#3498DB` (Selection rectangles, Info badges)
*   **Record Red (Destructive/Recording):** `#E74C3C` (Delete, Remove Filler, Recording)
*   **Success Green:** `#2ECC71` (Job Complete, Safe Zone)

### Text
*   **Primary Text:** `#FFFFFF` (Headlines, Primary Labels)
*   **Secondary Text:** `#888888` (Metadata, Descriptions)
*   **Disabled:** `#444444`

---

## 3. Typography

### Headlines & UI Labels
**Font:** `Inter` (Weights: 700, 900)
*   Tight tracking (-0.02em).
*   Uppercase for section headers (e.g., `// MEDIA_INPUTS`).

### Data & Timecode
**Font:** `Space Mono` or `JetBrains Mono` (Weights: 400, 700)
*   Used for: Timecodes, File paths, Transcripts, Stats.
*   *Why:* Monospace implies precision and "code-level" control.

### Scale
*   **H1 (Page Title):** 24px / 900 / Uppercase
*   **H2 (Section):** 14px / 700 / Uppercase / Spaced
*   **Body:** 14px / 400
*   **Mono/Data:** 12px / 400

---

## 4. Component Library (Cyber-NLE)

### The "Voxel" Card
A distinct container style for panels.
```css
.voxel-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 4px; /* Sharp corners, slight round */
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
```

### Buttons
*   **Primary (Yellow):** Solid `#F4D03F`, Black Text, Uppercase, Bold. Hover: Glow.
*   **Secondary (Outline):** Transparent, White Border, White Text.
*   **Ghost:** Text only, hover underline.

### The "Kinetic" Timeline
*   **Beat Ticks:** Vertical lines on the timeline representing musical beats.
*   **Energy Curve:** Subtle area graph overlay showing audio energy.
*   **Playhead:** Bright Yellow line with a "scanner" glow effect.

### Transcript Editor
*   **Word Spans:** `inline-block`, padding `2px 4px`.
*   **Hover:** Light gray background `#2a2a2a`.
*   **Active (Playing):** Yellow background `#F4D03F`, Black text.
*   **Deleted (Strikethrough):** Red strikethrough, opacity 0.5.

---

## 5. Motion Principles

*   **Instant Response:** Buttons click immediately (active state).
*   **Smooth Transport:** The playhead moves linearly, no easing.
*   **Reveal:** Panels slide in from the right (Drawers).
*   **Micro-glitch:** (Optional) Subtle glitch effect on major state changes (e.g., "Render Complete") to reinforce the AI/Tech vibe.

---

## 6. Layout Patterns

### Transcript-First Tri-Pane
1.  **Left (Source):** Video Player (16:9).
2.  **Center (Edit):** Transcript Text (Scrollable).
3.  **Bottom (Timeline):** Beat/Story Timeline (Full width).

### Shorts Studio ("Phone Rig")
*   **Center:** 9:16 Canvas (Phone aspect ratio).
*   **Overlay:** Safe Zones (TikTok/Reels UI mockups) toggled on/off.
*   **Right:** Controls (Caption Styles, Reframe settings).

---

## 7. Implementation Notes (CSS Variables)

```css
:root {
    --bg-color: #0a0a0a;
    --surface-color: #161616;
    --border-color: #2a2a2a;
    --primary-color: #F4D03F;
    --primary-text-on-color: #000000;
    --text-primary: #ffffff;
    --text-secondary: #888888;
    --danger-color: #E74C3C;
    --font-ui: 'Inter', sans-serif;
    --font-mono: 'Space Mono', monospace;
}
```
