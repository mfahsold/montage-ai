# Montage AI – Strategic Backlog (2026)

Based on the "PwC-like" Strategy Update (Jan 2026).

## Phase 1: Foundation (0-4 Weeks)

### Epic 1: Transcript-Based Editing UI (Priority: High)
**Goal:** Bring the existing backend capability (`text_editor.py`) to the Web UI.
- [ ] **Story 1.1: Transcript Panel Component**
  - Create a split-view layout in `index.html` / `app.js`.
  - Display transcription text with word-level timestamps.
  - Highlight words as the video plays.
- [ ] **Story 1.2: Click-to-Cut Interaction**
  - Allow users to click words to "strike them out".
  - Update the EDL/Cut-list in real-time.
  - Send updated cut-list to backend for preview generation.
- [ ] **Story 1.3: Filler Word Removal UI**
  - Add a "Remove Fillers" button in the UI.
  - Visualize removed filler words (e.g., red strikethrough).
- [ ] **Story 1.4: Preview Generation**
  - "Generate Preview" button that renders the text-based edits.
  - Use the existing CLI logic via a new API endpoint.

### Epic 2: Preview-First Pipeline (Priority: High)
**Goal:** Shift from "render everything" to "fast preview, then final render".
- [x] **Story 2.1: Low-Res Preview Job**
  - Create a fast render profile (360p, low bitrate).
  - Update `montage-ai.sh` / `editor.py` to support `--preview` flag efficiently.
- [ ] **Story 2.2: Web UI Preview Player**
  - Default the main player to show the preview stream.
  - Add a prominent "Final Render" button for high-quality export.

### Epic 3: UI Consolidation (Priority: Medium)
**Goal:** Clean up the "Toggle Graveyard".
- [ ] **Story 3.1: Quality Profiles**
  - Replace individual toggles (Upscale, Stabilize, Enhance) with a single "Quality Profile" dropdown (Draft, Social, Pro).
  - Map profiles to backend flags.
- [ ] **Story 3.2: Advanced Drawer**
  - Move granular controls into a collapsible "Advanced" section.

## Phase 2: Growth (5-12 Weeks)

### Epic 4: Shorts Studio (Priority: High)
**Goal:** Dedicated workflow for vertical video.
- [ ] **Story 4.1: Vertical Player Layout**
  - Add a "Shorts Mode" to the UI with 9:16 player container.
  - Show safe-area overlays (TikTok/Reels UI zones).
- [ ] **Story 4.2: Smart Reframe Visualization**
  - Visualize the crop path (where the camera is panning) over the original footage.
  - Allow manual adjustment of the crop center.
- [ ] **Story 4.3: Caption Styles**
  - Add style picker for captions (Karaoke, Minimal, Bold).
  - Live preview of caption rendering (using CSS/Canvas overlay before burning).

### Epic 5: Highlight Detection MVP (Priority: Medium)
**Goal:** Automated suggestion of "viral" clips.
- [ ] **Story 5.1: Backend Analysis**
  - Implement `highlight_detector.py` using audio energy and speech patterns.
  - Return a list of candidate time-ranges.
- [ ] **Story 5.2: Highlight Review UI**
  - Display candidate clips as cards in the UI.
  - Allow user to "Accept" or "Reject" highlights.

## Phase 3: Pro (3-6 Months)

### Epic 6: Audio Polish (Priority: Medium)
- [ ] **Story 6.1: Voice Isolation Integration**
  - Expose the cgpu `voice_isolation.py` job in the UI.
  - Add "Clean Audio" toggle.
- [ ] **Story 6.2: Noise Reduction**
  - Implement basic noise gate/reduction in `editor.py` (ffmpeg filters).

### Epic 7: Pro Export Pack (Priority: Low)
- [x] **Story 7.1: OTIO Export UI**
  - Add "Export to Premiere/Resolve" button.
  - Generate `.otio` file alongside `.mp4`.

## 90-Day Plan (Compressed)

| Week | Focus | Deliverable |
|------|-------|-------------|
| **0-4** | **Foundation** | Transcript Editor Preview-Flow, Doku-Sync, Telemetrie. |
| **5-8** | **Growth** | Shorts Studio 2.0 (Reframe v2 + Caption Styles + Highlight MVP). |
| **9-12** | **Pro** | Pro Handoff Pack + Audio Polish + Performance Targets. |

## Kern-KPIs

1.  **Time-to-First-Preview:** < 2-3 Min (Success Rate > 95%)
2.  **Transcript-Editing-Adoption:** > 40% der Sessions
3.  **Shorts-Creation-Cycle:** < 10 Min
4.  **Audio-Improvement-Rate:** > 70% (SNR-Check)
