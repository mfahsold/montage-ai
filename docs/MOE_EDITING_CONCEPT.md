# MoE Editing Control Plane (Core Concept)

This document defines a robust baseline concept for using multiple specialized AI experts as coordinated "tentacles" that control many editing levers at once.

Scope: AI-assisted post-production for existing footage (no synthetic video generation).

---

## 1. Goal

Build a Mixture-of-Experts (MoE) editing architecture that can:

- Optimize many levers in parallel (pacing, rhythm, framing, audio, color, story continuity).
- Stay deterministic and debuggable in production.
- Respect local-first constraints and safe fallbacks.
- Improve quality without sacrificing reliability or cost control.

---

## 2. Design Principles

1. **Control plane first, model plane second**
   - Orchestration quality decides product quality more than a single model upgrade.
2. **Expert proposals, not direct writes**
   - Experts propose deltas. A composer validates and applies them.
3. **Hard constraints over soft preferences**
   - No expert may violate delivery constraints, legal constraints, or user locks.
4. **Always degradable**
   - If experts fail, pipeline falls back to deterministic heuristic editing.
5. **Trace every decision**
   - Every accepted or rejected delta is logged with reason and confidence.

---

## 3. Reference Architecture

```text
User Intent + Style + Media
            |
            v
   Intent/Policy Layer
            |
            v
     MoE Control Plane
  (planner -> experts -> composer)
      |        |        |
      |        |        +--> conflict resolver
      |        +-----------> confidence calibrator
      +--------------------> budget/latency governor
            |
            v
   Timeline State + Delta Log
            |
            v
  Validation + Render + QC Loop
            |
            v
        Final Output
```

---

## 4. Expert Set (Tentacles)

Each expert controls a subset of editing levers and emits a bounded proposal.

1. **Rhythm Expert**
   - Levers: cut points, cut length, beat alignment.
2. **Scene/Pacing Expert**
   - Levers: scene transitions, tempo curve, jump-cut suppression.
3. **Narrative Expert**
   - Levers: intro-build-climax-outro flow, callback placement, b-roll narrative relevance.
4. **Audio Polish Expert**
   - Levers: ducking, loudness targets, speech clarity prioritization.
5. **Framing Expert**
   - Levers: reframe path, subject lock, vertical crop smoothness.
6. **Color/Look Expert**
   - Levers: style LUT choice, contrast/saturation constraints, shot-to-shot consistency.
7. **Compliance/QC Expert**
   - Levers: black frame removal, clipping detection, loudness and brightness guards.
8. **Cost/Latency Expert**
   - Levers: preview vs master compute allocation, optional cloud offload decisions.

---

## 5. Contract: Shared State and Delta API

### 5.1 Shared Timeline State

- Immutable input metadata:
  - scene boundaries, beat map, transcript spans, detected subjects, motion vectors
- Mutable edit state:
  - timeline segments, transition graph, audio automation lanes, render profile
- Policy state:
  - user locks, style constraints, non-goals, runtime budget

### 5.2 Delta Proposal Schema

```json
{
  "expert": "rhythm",
  "target": "timeline.segment[12:19]",
  "change": {
    "cut_points": [12.42, 13.01, 13.76],
    "transition": "hard_cut"
  },
  "confidence": 0.84,
  "impact": {
    "quality_gain": 0.18,
    "risk": 0.09,
    "compute_ms": 120
  },
  "constraints_checked": ["no_user_lock_violation", "max_cut_rate"]
}
```

Rules:
- Experts cannot mutate state directly.
- Every delta must include confidence and estimated impact.
- Every delta must be reversible.

---

## 6. Composition and Conflict Resolution

The **Composer** merges expert deltas in ordered phases:

1. **Feasibility filter**
   - Reject invalid ranges, lock violations, impossible transitions.
2. **Constraint filter**
   - Enforce hard bounds (delivery profile, legal, safety, compute).
3. **Conflict graph build**
   - Detect overlapping edits on same target scope.
4. **Priority + confidence arbitration**
   - Weighted policy: hard constraints > narrative coherence > rhythm > cosmetic polish.
5. **Local simulation pass**
   - Evaluate merged candidates in short windows before global apply.
6. **Commit with audit log**
   - Store accepted/rejected deltas with reason codes.

Conflict strategies:
- Non-overlapping deltas: merge automatically.
- Overlapping but compatible: compose sequentially.
- Overlapping and incompatible: choose winner by policy score, keep loser as fallback variant.

---

## 7. Multi-Pass Execution Loop

Pass 0: Analyze media and build baseline timeline.

Pass 1: Experts propose independent deltas.

Pass 2: Composer builds candidate timelines (A/B/N variants).

Pass 3: Fast render + automatic QC scoring.

Pass 4: Repair pass for only failing dimensions (for example loudness or cut jitter).

Pass 5: Final render under selected quality profile.

This keeps the system modular and allows targeted retries instead of full reruns.

---

## 8. Reliability and Fallbacks

1. **Circuit breaker per expert**
   - Disable unstable expert after repeated failures.
2. **Graceful degrade**
   - Fall back to deterministic logic from existing pipeline modules.
3. **Time budget enforcement**
   - Stop proposal rounds when preview SLA is at risk.
4. **Safe defaults**
   - Prefer no-op over risky edits when confidence is low.

---

## 9. Observability and Evaluation

Track quality as a vector, not a single score:

- Rhythm alignment score
- Narrative coherence score
- Audio intelligibility score
- Visual consistency score
- Technical QC score
- Time-to-preview and total compute cost

Log requirements:
- Delta decision logs (accepted/rejected + reason)
- Per-expert reliability metrics
- Versioned policy and model fingerprints

---

## 10. Security and Policy Guardrails

- Local-first by default for sensitive content.
- Explicit policy gate for any cloud offload path.
- No synthetic video generation in this architecture.
- Full provenance of edits for auditability.

---

## 11. Incremental Implementation Plan

### Phase A (2-4 weeks): Control Plane Skeleton

- Add shared state model and delta schema.
- Wrap current deterministic modules as pseudo-experts.
- Implement composer with simple priority rules.

### Phase B (4-8 weeks): Parallel Expert Runtime

- Add asynchronous proposal execution.
- Add conflict graph + local simulation.
- Add per-expert circuit breakers and metrics.

### Phase C (8-12 weeks): Adaptive Policy

- Learn policy weights from accepted edits and QC outcomes.
- Add profile-specific strategies (preview vs master).
- Add optional cloud expert workers behind strict policy gates.

---

## 12. Mapping to Current Montage AI Modules

- `core/montage_builder.py`: host control-plane lifecycle and multi-pass loop
- `audio_analysis.py` + `audio_enhancer.py`: rhythm/audio experts
- `scene_analysis.py` + `footage_manager.py`: pacing/narrative experts
- `auto_reframe.py` + `clip_enhancement.py`: framing and polish experts
- `segment_writer.py`: candidate render and variant outputs
- `web_ui/`: expose lock controls, expert trace, and explainability panel

Suggested new package:

```text
src/montage_ai/moe/
  contracts.py
  state.py
  experts/
  composer.py
  policy.py
  metrics.py
```

---

## 13. Definition of Done (for a "robust" v1)

The concept is considered production-ready when:

1. Preview SLA is stable under mixed workloads.
2. At least 80% of edits are explainable via delta logs.
3. Expert failure does not break pipeline completion.
4. QC regressions are lower than baseline deterministic pipeline.
5. Operators can lock regions and trust they remain untouched.
