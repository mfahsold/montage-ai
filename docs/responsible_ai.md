# Responsible AI & Transparency

Montage AI is built as a **local-first, OSS-leaning** editing assistant. We focus on explainability, user control, and practical privacy. This document describes how the system behaves and which safeguards are in place.

---

## Principles

- **Local by default:** Footage stays on your machine unless you explicitly enable cloud GPU or remote LLM backends.
- **No training on your data:** User footage and prompts are never used to train models.
- **User control:** You choose features and can export editable timelines (OTIO/EDL) for full manual control.
- **Transparency:** We expose what models are available, what the system decided, and how to audit results.
- **OSS-first:** Prefer open-source libraries and open research; closed backends are optional.

---

## Data Handling

- **Local processing:** Video, audio, and intermediate files are stored locally in `data/` by default.
- **Optional remote compute:** When `CGPU_ENABLED=true`, heavy analysis or LLM calls can be offloaded. This is opt-in.
- **No training or retention:** We do not train on user footage or prompts. The system does not upload content unless you enable cgpu or other remote backends.

---

## Model Usage & Guardrails

- **Creative Director LLM** translates prompts into structured JSON.
- **Schema enforcement:** Responses are validated and normalized before use.
- **Fallback chain:** Keyword match → LLM (if enabled) → default instructions.
- **Guardrails:** Prompts are wrapped with system rules to keep edits safe, scoped, and deterministic.

---

## Explainability & Auditability

- **Decision logs:** When `EXPORT_DECISIONS=true`, jobs emit `decisions_<job_id>.json`.
- **API access:** `GET /api/jobs/<id>/decisions` surfaces those logs for inspection.
- **Metrics:** The Story Engine exposes measurable error (MSE) between target tension curve and output.

---

## OSS & Open Research Stack

Montage AI relies heavily on open-source libraries:

- **FFmpeg** — encoding/decoding
- **OpenCV** — visual analysis
- **librosa** — audio analysis
- **OpenTimelineIO** — NLE export
- **Whisper** — transcription
- **Demucs** — voice isolation
- **Real-ESRGAN** — upscaling

See `docs/models.md` for details and citations.

---

## Scope Boundaries

**In scope:**

- AI-assisted rough cuts from existing footage
- Beat sync and story arc pacing
- Professional handoff via OTIO/EDL/XML

**Out of scope:**

- Generative text-to-video
- Full non-linear editing replacement
- Social hosting platform

---

## UI Transparency

The Web UI exposes a **Responsible AI** card that reflects:

- Data handling policy
- Explainability options
- Available LLM backends
- OSS stack
- Scope boundaries

See `/api/transparency` for the source payload.

