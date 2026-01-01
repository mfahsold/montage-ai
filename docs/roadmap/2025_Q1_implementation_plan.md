# Q1 2025 Implementation Plan: "The Polish Update"

## 1. Roadmap (Revised)

Based on the "AI Rough Cut" strategy: Focus on speed, musicality, and pro-workflow integration.

| Feature | Priority | Status | Target Date |
|---------|----------|--------|-------------|
| **Distributed Build System** | P0 | ✅ Done | Week 1 |
| **Pacing Curves (Audio Analysis)** | P0 | ✅ Done | Week 1 |
| **Smart Clip Selection (Energy Aware)** | P0 | ✅ Done | Week 1 |
| **EDL/XML Export (Pro Workflow)** | P0 | ✅ Done | Week 1 |
| **Enhancement Toggles (Fast/Quality)** | P1 | 📅 Next | Week 2 |
| **Preset/Template Pack** | P1 | 📅 Next | Week 2 |
| **Speech-to-Text (Whisper)** | P2 | 📅 Planned | Week 3 |
| **Scene Detection (PySceneDetect)** | P2 | 📅 Planned | Week 3 |
| **Web UI: Batch & Preview** | P2 | 📅 Planned | Week 4 |

## 2. Gap Matrix

| Feature Area | Current State | Desired State | Gap |
|--------------|---------------|---------------|-----|
| **Workflow** | Output is final MP4 | Output is XML + Proxies for Resolve/Premiere | **Closed** (XML Export) |
| **Pacing** | Random/Fixed beats | Dynamic (Intro/Verse/Chorus) | **Closed** (MusicSection) |
| **Selection** | Random usage | Energy-matched (Action=High Energy) | **Closed** (Smart Selector) |
| **Enhancement**| Hardcoded settings | User toggles (Fast vs. Quality) | Open (Need Config Update) |
| **Styles** | Hardcoded logic | JSON Templates (Vlog, Travel, Sport) | Open (Need Template System) |
| **Speech** | None | Auto-captions & Text-based cuts | Open (Need Whisper) |

## 3. User Stories (Top 3)

### Story 1: The "Pro" Handover
**As a** professional editor,
**I want** to export an XML file from Montage AI,
**So that** I can fine-tune the cuts and color grade in DaVinci Resolve without re-cutting.

*   **Acceptance Criteria:**
    *   System generates `.xml` (FCP7 format) alongside `.mp4`.
    *   XML imports into Resolve/Premiere with correct timing.
    *   Source files link correctly.

### Story 2: The "Vlog" Preset
**As a** YouTuber,
**I want** to select a "Travel Vlog" style,
**So that** the montage uses upbeat pacing and bright color grading defaults.

*   **Acceptance Criteria:**
    *   `presets/travel_vlog.json` defines pacing (fast), color (vibrant), and music (upbeat).
    *   CLI accepts `--style travel_vlog`.

### Story 3: The "Quick" Preview
**As a** user,
**I want** to generate a low-res preview in <1 minute,
**So that** I can check the sync before committing to a full 4K render.

*   **Acceptance Criteria:**
    *   `--preview` flag disables upscaling and stabilization.
    *   Renders at 480p/720p.
    *   Uses "Fast" enhancement profile.

## 4. Technical Next Steps

1.  **Enhancement Toggles:** Update `config.py` and `cgpu_upscaler.py` to respect a `quality_profile` (fast/balanced/best).
2.  **Template System:** Extract hardcoded style logic from `montage_builder.py` into `src/montage_ai/styles/*.json`.
3.  **Whisper Integration:** Create a `Transcriber` class wrapping `openai-whisper` (running on GPU node).
