# Backlog (2026) — Pro Video Editing Trends

This backlog reflects current industry and research signals for professional video editing. It is structured as **Now / Next / Later**, and scoped to Montage AI's **polish-first, local-first** mandate (no generative video creation).

## Trend Signals (2024–2026)

- **Text-based editing + transcript workflows** (Premiere Pro, Descript, creator tools)
- **AI-driven object masking/roto and smart tracking** (faster cleanup, reframing)
- **HDR/Color management automation (ACES/auto log transforms)**
- **Semantic scene detection + highlight extraction** (shorts/marketing)
- **B-roll retrieval from local libraries using embeddings**
- **Audio-first polish (dialogue cleanup, loudness, speaker-aware ducking)**
- **Production-grade stabilization + motion smoothing**
- **Shot-boundary detection improvements (transformer-based)**
- **Workflow automation & QC: consistency, legal/music clearance, technical checks**

## Now (0–2 months)

1. **Text-Based Editing v2 (Transcript-first)**
   - **Why:** Pro editors increasingly expect transcript-driven cut assembly.
   - **Scope:** Word-level cut handles, filler-word removal presets, “keep-voice” guardrails, timeline sync.
   - **Files:** `transcriber.py`, `web_ui/` transcript editor, `segment_writer.py`.

2. **AI Stabilization Controls + Presets (UX hardening)**
   - **Why:** Stabilization is now a must-have for professional delivery.
   - **Scope:** Preset labels in UI, per-clip overrides, preview vs master behavior.
   - **Files:** `web_ui/`, `clip_enhancement.py`, `core/montage_workflow.py`.

3. **Semantic B-Roll Matching (Local-Only)**
   - **Why:** Pro edits benefit from quick, relevant cutaways.
   - **Scope:** Embed local footage tags, fast vector search, confidence gating.
   - **Files:** `broll_planner.py`, `footage_manager.py`.

4. **HDR/Log Auto Color Management (Safe Defaults)**
   - **Why:** Log footage is standard in pro pipelines.
   - **Scope:** Camera log detection → auto transform to Rec.709; optional ACES-like curve.
   - **Files:** `color_grading.py`, `ffmpeg_config.py`.

## Next (2–6 months)

- **Smart Masking/Tracking Hooks**
   - **Why:** Pro workflows rely on clean subject isolation and targeted fixes.
   - **Scope:** Optional object/face mask plugin hooks; non-blocking in preview.
   - **Files:** `auto_reframe.py`, `clip_enhancement.py`.

- **Shot Boundary Detection Upgrade**
   - **Why:** Cleaner scene cuts improve pacing and reduce jump cuts.
   - **Scope:** Evaluate TransNetV2-style detector for scene detection; fall back to current method.
   - **Files:** `scene_analysis.py`.

- **Audio Consistency Pack**
   - **Why:** Loudness, noise and music ducking are critical for delivery.
   - **Scope:** Multi-speaker VAD, target LUFS per output profile, multi-band ducking.
   - **Files:** `audio_enhancer.py`.

- **QC & Compliance Checks**
   - **Why:** Professional delivery requires predictable output.
   - **Scope:** Detect missing audio, clipped peaks, extreme brightness, black frames.
   - **Files:** `segment_writer.py`, `audio_analysis.py`.

## Later (6–12 months)

- **Multi-Cam Auto-Sync + Angle Selection**
   - **Why:** Common in interviews and event edits.
   - **Scope:** Audio-based sync, “best angle” heuristics, storyboard-level swaps.
   - **Files:** `core/montage_builder.py`.

- **Collaborative Review (Local-first)**
   - **Why:** Pro teams need review and approval loops.
   - **Scope:** Local review packages, comment export, OTIO annotations.
   - **Files:** `export/`, `web_ui/`.

- **Advanced Story Engine (Narrative Templates)**
   - **Why:** Story pacing is the differentiator for pro edits.
   - **Scope:** Story arcs per format (docu, marketing, trailer), dynamic cut-length rules.
   - **Files:** `core/montage_builder.py`, `styles/`.

## Research Watchlist (No immediate scope)

- **Text-based talking-head video editing** (diffusion-based edits) — monitor for ethical, local-safe use.
- **Agentic long-form editing assistants** — evaluate for planning only, not generation.
- **Video-to-music alignment models** — consider for automatic music selection, not generation.

## Explicit Non-Goals

- Prompt-to-video generation or fully synthetic video creation.
- Cloud-only proprietary pipelines without a local-first path.

## Sources (Signals)

- Adobe Premiere Pro Text-Based Editing (Help Center, 2026)
- Adobe blog: AI object masking & faster tracking (2026)
- ArXiv 2024–2026: shot boundary detection surveys, transcript-based editing research
- Industry tools: Descript, Runway, Pika (trend indicators only)
