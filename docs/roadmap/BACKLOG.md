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

## SWOT Update (Q1 2026, Critical)

### Strengths

- **Clear positioning:** Local-first, polish-first is still a meaningful differentiator for privacy-sensitive teams and regulated environments.
- **Practical pipeline DNA:** Montage AI already focuses on the part professionals actually ship: cut decisions, stabilization, audio cleanup, reframing, and render reliability.
- **FFmpeg-centered architecture:** Strong leverage on proven tooling keeps runtime costs lower than full generative stacks and makes deterministic output easier.
- **Config and deployment maturity:** Recent hardening work reduced fragility in config/runtime behavior and improved CI confidence.

### Weaknesses (Brutal Honesty)

- **Product gap vs incumbent NLEs:** Premiere, Resolve, and Final Cut now expose AI features directly inside the editing timeline (transcript search/edit, beat intelligence, masking, speaker-aware tooling). We are not yet the default daily surface for editors.
- **No dominant UX wedge yet:** Our technical modules are solid, but the end-to-end editing interaction is less sticky than in-IDE NLE workflows.
- **Risk of "feature parity chase":** A roadmap built around matching checkboxes can trap us in permanent catch-up against larger product teams.
- **Research-to-product latency:** New academic capabilities (controllable insertion/removal, long-video consistency, efficient editing) are moving faster than our integration loop.
- **Brand signal problem:** In the AI creator space, distribution and ecosystem gravity often beat technical merit.

### Opportunities

- **Own the "trusted finishing" layer:** Position Montage AI as the reliable post stack for footage enhancement, QC, compliance, and delivery packaging instead of generic AI creation.
- **Exploit local/privacy constraints:** Offer a strong alternative where cloud-only assistants are blocked by policy, client contracts, or data locality requirements.
- **Workflow bridge strategy:** Tight round-trip support with editor ecosystems (Premiere/Resolve/FCP timeline interchange, review artifacts, local approvals) can beat direct UI competition.
- **Verticalized presets:** Packaging repeatable outcomes (podcast cleanup, interview multicam prep, social cutdown polish) can create faster time-to-value than broad "general AI editing" claims.
- **Leverage fast open research:** The current paper wave on efficient and controllable video editing can be harvested selectively for polish tasks without abandoning non-generative positioning.

### Threats

- **Incumbent acceleration:** Adobe/Blackmagic/Apple are shipping AI-assisted editing features on short release cycles with native workflow lock-in.
- **Open-source model velocity:** Community stacks (for example ComfyUI and research repos around instruction/reference-guided editing) are iterating weekly, compressing differentiation windows.
- **Expectation inflation:** Users increasingly expect "one-click magic" while still demanding broadcast-safe reliability; failing either side loses trust.
- **Compute arms race:** Even "editing" capabilities are drifting toward foundation-model-level resource demands that can challenge local-first economics.
- **Commoditization risk:** If transcript editing, masking, beat sync, and cleanup become baseline everywhere, undifferentiated features collapse into table stakes.

### Strategic Implications (Next 12 Months)

1. **Do not compete as a general AI editor.** Compete as the most reliable local finishing and delivery copilot.
2. **Prioritize workflow integration over novelty.** Winning import/export, review, and QA loops can outlast flashy model demos.
3. **Ship fewer, sharper bets.** Focus on transcript-first assembly quality, audio polish depth, and deterministic QC.
4. **Treat research as option value, not roadmap vanity.** Pull in only what materially improves speed, consistency, or operator control.
5. **Instrument proof of value.** Track time saved per edit, defect reduction, and revision cycles, not just feature count.

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

- **MoE Editing Control Plane (Foundation)**
   - **Why:** Coordinating multiple specialized AI experts is needed to optimize many editing levers without losing reliability.
   - **Scope:** Shared timeline state, delta proposal schema, conflict-aware composer, fallback path to deterministic pipeline.
   - **Files:** `core/montage_builder.py`, `segment_writer.py`, new `src/montage_ai/moe/*`, `web_ui/` explainability panel.
   - **Concept:** `docs/MOE_EDITING_CONCEPT.md`

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

- Apple Final Cut Pro release notes (Jan 2026): transcript search, visual search, beat detection
- Adobe Premiere Pro updates (Jan 2026): AI object masking and tracking workflow updates
- Blackmagic DaVinci Resolve 20 "What's New": AI IntelliScript, AI Multicam, AI Audio Assistant, AI Beat Detector
- Runway product/changelog updates (2025–2026 cadence)
- CapCut AI editor positioning (consumer automation benchmark)
- ArXiv 2025–2026 representative papers:
   - EditCtrl (real-time generative video editing efficiency)
   - MLV-Edit (minute-level long-video consistency)
   - RFDM (causal/efficient variable-length editing)
   - PropFly, NOVA, FREE-Edit, Kiwi-Edit (instruction/reference-guided controllable editing)
   - TalkLess, EditBoard (text/transcript editing UX + evaluation rigor)
   - UniVBench (unified evaluation of understanding/generation/editing)
- Open-source community signals: ComfyUI release cadence/ecosystem scale, Auto-Editor adoption, emerging research repos (for example Kiwi-Edit)
