# cgpu Offloading Analysis & Recommendations

## Executive Summary
We have successfully integrated `cgpu` for **AI Upscaling (Real-ESRGAN)** and **Generative Video (Wan2.1)**. This document analyzes other CPU/GPU intensive tasks in the Montage AI pipeline to identify further offloading opportunities to free up local resources and leverage free cloud GPUs.

## Current Offloading Status
| Task | Status | Implementation | Benefit |
|------|--------|----------------|---------|
| **AI Upscaling** | ✅ Implemented | `cgpu_upscaler.py` | Massive speedup (T4 GPU vs CPU), frees local RAM |
| **Transcription** | ✅ Implemented | `transcriber.py` | Offloads Whisper (VRAM heavy) to Cloud GPU |
| **LLM Director** | ⚠️ Partial | `creative_director.py` | Can use `cgpu serve` for Gemini, but currently defaults to local Ollama |

## Candidates for Future Offloading

### 1. Video Stabilization (`vidstab`)
**Current:** Runs locally via FFmpeg `vidstabdetect` + `vidstabtransform` filters (CPU-bound).
**Impact:** High CPU usage, slow for 4K footage. Blocks other operations.
**Feasibility:** **High**.
- **Workflow:** Upload clip → Run FFmpeg 2-pass stabilization on Colab → Download result.
- **Pros:** Offloads heavy CPU crunching. Colab CPUs are often faster than local dev environment.
- **Cons:** Upload/Download overhead. `vidstab` is not GPU-accelerated by default, but running it on a remote powerful CPU still helps.
- **Recommendation:** Implement `cgpu_stabilizer.py` following the `cgpu_upscaler.py` pattern.

### 2. Deep Semantic Analysis (Feature Extraction)
**Current:** `footage_analyzer.py` uses heuristic OpenCV methods (edge density, color temp) which are fast but limited.
**Potential:** Upgrade to Deep Learning models (CLIP, ResNet) for semantic understanding (e.g., "find clips with happy people").
**Feasibility:** **Very High**.
- **Workflow:** Upload clip (or keyframes) → Run CLIP/BLIP analysis on Colab GPU → Return JSON metadata.
- **Pros:** Enables "Google Photos" style search and much smarter editing decisions. Impossible to run efficiently on local CPU.
- **Cons:** Requires new logic in `footage_analyzer.py`.
- **Recommendation:** Create `cgpu_analyzer.py` to run CLIP on Colab and return semantic tags.

### 3. Transcoding / Proxy Generation
**Current:** Local FFmpeg.
**Feasibility:** **Low to Medium**.
- **Analysis:** Transcoding is fast enough locally for 1080p. The upload/download time for raw footage often exceeds the transcoding time savings unless the connection is very fast (fiber).
- **Recommendation:** Keep local for now.

### 4. Final Rendering
**Current:** Local `moviepy` + FFmpeg.
**Feasibility:** **Low**.
- **Analysis:** Requires uploading ALL source clips, assets, music, and fonts to Colab. Complex synchronization.
- **Recommendation:** Keep local.

## Implementation Roadmap

### Phase 1: Stabilization Offloading
Create `StabilizationService` that transparently uses cgpu if available.
```python
def stabilize_clip(input_path):
    if is_cgpu_available():
        return stabilize_with_cgpu(input_path) # Remote FFmpeg
    else:
        return stabilize_locally(input_path)   # Local FFmpeg
```

### Phase 2: Semantic Analysis Agent
Enhance `footage_analyzer.py` to query a remote CLIP agent.
- **Input:** Video file
- **Process:** Extract 1 frame/sec → Upload to Colab → CLIP Encode → Average Embeddings
- **Output:** List of semantic tags ("beach", "sunset", "running") + Mood scores.

## Technical Requirements
- **Colab Environment:** Need to ensure `libvidstab` is available or compile FFmpeg with it on Colab (standard Colab FFmpeg usually has it, but need to verify).
- **Concurrency:** The new UUID-based job directory structure (implemented in v2.2.1) supports parallel jobs, allowing us to stabilize multiple clips at once if we implement a queue system.

## Conclusion
Immediate priority should be given to **Stabilization** as it is a direct drop-in replacement for existing slow local code. **Semantic Analysis** is a feature enabler that should be explored next to improve the "AI" capabilities of the editor.
