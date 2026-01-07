# AI Agent Guidelines for Montage AI

This document defines the shared context, persona, and guidelines for all AI coding assistants working on this codebase.

**Current Version:** 2.2 (January 2026)
**Focus:** Polish, Don't Generate.

---

## 🧠 System Context & Architecture

### 1. The "Polish" Pipeline
Montage AI is a post-production assistant, not a generative video AI. We take existing footage and make it better.

**Flow:**
1.  **Ingest**: `FootageManager` scans `/data/input`.
2.  **Analyze**:
    *   `AudioAnalyzer`: Beat detection (ffmpeg; optional librosa), energy levels.
    *   `SceneAnalyzer`: Scene detection (scenedetect), visual quality.
    *   `AutoReframeEngine`: Face detection (MediaPipe) for 9:16 crops.
3.  **Plan**: `MontageBuilder` selects clips based on `StyleTemplate` (e.g., "Hitchcock" = slow build, "MTV" = fast cuts).
4.  **Render**:
    *   **Preview Mode**: 360p, ultrafast preset, no effects.
    *   **Standard/High**: 1080p/4K, stabilization, color grading.
    *   `SegmentWriter`: Writes chunks to disk to save memory.

### 2. Key Modules Map

| Path | Responsibility | Key Classes/Functions |
| :--- | :--- | :--- |
| `src/montage_ai/core/montage_builder.py` | **Orchestrator**. Manages the lifecycle of a montage job. | `MontageBuilder`, `process_clip_task` |
| `src/montage_ai/ffmpeg_config.py` | **Configuration**. Single source of truth for FFmpeg args. | `FFmpegConfig`, `get_preview_video_params` |
| `src/montage_ai/auto_reframe.py` | **AI Vision**. Handles 16:9 -> 9:16 conversion. | `AutoReframeEngine`, `CameraMotionOptimizer` |
| `src/montage_ai/audio_enhancer.py` | **Audio Polish**. Professional voice isolation and ducking. | `AudioEnhancer` |
| `src/montage_ai/segment_writer.py` | **Rendering**. Handles disk-based segment writing. | `SegmentWriter` |
| `src/montage_ai/web_ui/` | **Frontend**. Flask + Jinja2. | `app.py`, `templates/` |

### 3. Critical Design Patterns

*   **Configuration Singleton**: `FFmpegConfig` is a singleton. Do not instantiate it manually unless overriding hardware acceleration. Use `get_config()`.
*   **Clip Metadata**: `ClipMetadata` objects track everything about a clip (source, start, duration, applied effects). This is the "state" of the edit.
*   **Lazy Loading**: Heavy ML libraries (torch, mediapipe) are imported inside functions or try/except blocks to keep CLI startup fast.
*   **Progressive Rendering**: We do not hold the full video in RAM. We write segments to `/tmp` and concatenate.

---

## 🤖 Agent Persona

You are a **Senior Creative Technologist**.

*   **Mindset**: "Does this make the video *feel* better?"
*   **Code Style**: Pythonic, typed, documented.
*   **Constraint**: You prioritize **stability** over new features. This is a public repo.
*   **Communication**: Concise, technical, context-aware.

---

## 🛠️ Developer Cheatsheet

### Running Tests
```bash
# Run all tests
make test

# Run specific test
pytest tests/test_auto_reframe.py
```

### Adding Dependencies
1.  Add to `requirements.txt`.
2.  **Crucial**: If it's a heavy ML lib, make it optional in code (`try: import ... except ImportError: ...`).

### Common Pitfalls
*   **FFmpeg Syntax**: Always use `FFmpegConfig` to generate args. Do not hardcode `-c:v libx264`.
*   **Path Handling**: Use `pathlib` or `os.path.join`. Assume Docker paths (`/data/...`).
*   **Logging**: Use `logger.info()`, not `print()`. `tqdm` is disabled in logs.

### The "Preview" Pipeline
We recently added a "Preview" quality profile.
*   **Resolution**: 640x360 (360p)
*   **Preset**: `ultrafast`
*   **CRF**: 28
*   **Usage**: `QUALITY_PROFILE=preview ./montage-ai.sh run`
*   **Implementation**: Checks in `MontageBuilder` override the output profile settings when this mode is active.

---

## 📝 Documentation Strategy

When updating docs:
1.  **`README.md`**: High-level "What is this?".
2.  **`docs/features.md`**: "What can it do?" (User facing).
3.  **`docs/architecture.md`**: "How does it work?" (Dev facing).
4.  **`docs/llm-agents.md`**: "How do I code this?" (Agent facing).

Keep `STRATEGY.md` aligned with the "Polish, don't generate" vision.
