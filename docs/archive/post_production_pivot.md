# Montage AI: Pivot to AI Post-Production Assistant

## 1. Vision & Strategy

**Goal:** Transform Montage AI from a "video generator" into a **professional AI Post-Production Assistant**.
**Philosophy:** We do not generate pixels from scratch. We enhance, organize, cut, and polish existing footage to tell a story.

**Core Pillars:**
1.  **AI Editing & Storytelling:** Intelligent cutting based on beat, mood, and narrative arc.
2.  **Enhancement & Restoration:** Upscaling, stabilization, and noise reduction using Cloud GPU.
3.  **Professional Workflow:** Proxy workflows, B-roll planning, and export to industry-standard NLEs (DaVinci Resolve, Premiere).
4.  **Cloud GPU Integration:** Offload heavy compute tasks (Upscaling, Transcoding, Subtitling) to `cgpu` (Google Colab/Kaggle).

---

## 2. Architecture Overview

The system is divided into three main stages: **Ingest & Analysis**, **Creative Direction & Edit**, and **Finishing & Export**.

### Stage 1: Ingest & Analysis (Local + Cloud Hybrid)
*   **Footage Manager (`footage_manager.py`):** Scans input directories.
*   **Video Agent (`video_agent.py`):**
    *   **Action:** Deep semantic analysis of clips.
    *   **Tech:** CLIP/SigLIP embeddings, Object Tracking.
    *   **Reuse:** Existing `video_agent.py` is perfect. Enhance with `cgpu` for faster embedding generation if local CPU is weak.
*   **Subtitles & Transcription (New):**
    *   **Action:** Transcribe dialogue for "paper edit" and subtitle generation.
    *   **Tech:** OpenAI Whisper.
    *   **Implementation:** Run `whisper` via `cgpu` to avoid local resource drain.

### Stage 2: Creative Direction & Edit (Local LLM)
*   **Creative Director (`creative_director.py`):**
    *   **Action:** Takes user prompt ("Make a high-energy gym video") and analysis data.
    *   **Output:** JSON Edit Decision List (EDL) with specific clip selections, timing, and transition types.
    *   **Tech:** Ollama (Llama 3.2) running locally or on `cgpu` (Gemini/OpenAI).
*   **B-Roll Planner (New Module):**
    *   **Action:** Identifies gaps in the narrative and selects B-roll from `video_agent` memory to fill them.
    *   **Logic:** "Narrator mentions 'protein shake' -> Query Video Agent for 'shaker bottle' -> Insert clip."

### Stage 3: Finishing & Export (Cloud GPU Heavy)
*   **Upscaling (`cgpu_upscaler.py`):**
    *   **Action:** 4x upscale for low-res footage.
    *   **Tech:** Real-ESRGAN on `cgpu`.
    *   **Reuse:** Existing module is solid.
*   **Stabilization:**
    *   **Action:** Fix shaky handheld footage.
    *   **Tech:** Gyroflow (if metadata exists) or ffmpeg `vidstab`.
    *   **Plan:** Move `vidstab` processing to `cgpu` if possible, or keep local as fallback.
*   **Color Grading (Expanded):**
    *   **Action:** Apply LUTs or match color between clips.
    *   **Tech:** FFmpeg LUTs (current) -> AI Color Match (future).
*   **Export (`timeline_exporter.py`):**
    *   **Action:** Generate `.otio` or `.xml` for NLEs.
    *   **Reuse:** Existing module needs to be promoted from "experimental" to core.

---

## 3. Component Reuse & Migration Plan

| Component | Status | Action | Notes |
| :--- | :--- | :--- | :--- |
| `open_sora.py` | **REMOVE** | Delete | Pure generation is out of scope. |
| `wan_vace.py` | **REMOVE** | Delete | Pure generation is out of scope. |
| `video_agent.py` | **KEEP** | Refactor | Core for "B-roll planning" and semantic search. |
| `cgpu_upscaler.py` | **KEEP** | Expand | Add support for other heavy tasks (e.g., stabilization). |
| `editor.py` | **KEEP** | Refactor | Strip out generation calls; focus on assembly. |
| `timeline_exporter.py` | **PROMOTE** | Polish | Essential for "Professional Export" feature. |
| `creative_director.py` | **KEEP** | Update | Tune prompts for "Editing" vs "Creating". |

---

## 4. Detailed Implementation Roadmap

### Phase 1: Cleanup & Focus (Immediate)
1.  **Delete Generation Code:** Remove `open_sora.py`, `wan_vace.py`, and related dependencies.
2.  **Refactor `editor.py`:** Remove any logic that attempts to "generate" missing clips. Instead, log a "Missing Footage" warning or use a placeholder.

### Phase 2: The "Smart Editor" (Short Term)
1.  **Integrate Subtitles:**
    *   Create `src/montage_ai/transcriber.py`.
    *   Implement `cgpu run openai-whisper ...` to process audio tracks.
    *   Save subtitles as `.srt` and `.vtt`.
2.  **Enhanced B-Roll Planning:**
    *   Update `creative_director.py` to accept a "Script" or "Voiceover" input.
    *   Use `video_agent.py` to find clips matching script keywords.

### Phase 3: Cloud GPU Pipeline (Medium Term)
1.  **Generalize `cgpu` Wrapper:**
    *   Currently `cgpu_upscaler.py` is specific.
    *   Create `src/montage_ai/cgpu_jobs.py` to handle generic jobs: `Upscale`, `Stabilize`, `Transcribe`.
2.  **Remote Rendering (Optional):**
    *   Allow the final FFmpeg render to happen on `cgpu` if the timeline is complex.

### Phase 4: Professional Export (Long Term)
1.  **OTIO Maturity:** Ensure `timeline_exporter.py` produces files compatible with DaVinci Resolve 18+.
2.  **Proxy Workflow:** Generate low-res proxies locally for fast preview, use high-res originals for final export.

---

## 5. Technical Architecture: Cloud GPU Integration

We will use `cgpu` as a "Sidecar Compute Unit".

```mermaid
graph TD
    User[User Laptop] -->|1. Config & Media| Docker[Montage AI Container]
    Docker -->|2. Heavy Tasks (Upscale/Transcribe)| CGPU[Google Colab / Kaggle]
    CGPU -->|3. Processed Assets| Docker
    Docker -->|4. Assembly| FinalVideo[Final MP4]
    Docker -->|5. XML/OTIO| NLE[DaVinci Resolve]
```

**Data Flow:**
1.  **Local:** User organizes footage.
2.  **Hybrid:** `video_agent` scans footage (Local CPU or Cloud GPU).
3.  **Cloud:** Specific clips flagged for Upscaling/Stabilization are uploaded to `cgpu`, processed, and downloaded.
4.  **Local:** `editor.py` assembles the timeline using the *processed* clips.
5.  **Export:** Project file generated for NLE.

## 6. Next Steps

1.  **Approve this plan.**
2.  **Execute Phase 1:** Delete generation files.
3.  **Execute Phase 2:** Implement `transcriber.py` with Whisper via `cgpu`.
