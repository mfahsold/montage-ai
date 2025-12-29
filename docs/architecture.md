# Architecture

System design and component overview.

---

## High-Level Flow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Interface                                 │
│                                                                          │
│   ./montage-ai.sh run hitchcock --cgpu --upscale                        │
│                              │                                           │
└──────────────────────────────┼───────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Docker Container                                  │
│                                                                          │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │ Creative        │     │ Style           │     │ Footage         │   │
│  │ Director        │────▶│ Templates       │────▶│ Manager         │   │
│  │ (LLM)           │     │ (JSON)          │     │ (Story Arc)     │   │
│  └────────┬────────┘     └─────────────────┘     └────────┬────────┘   │
│           │                                               │             │
│           └───────────────────┬───────────────────────────┘             │
│                               ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         Editor                                    │   │
│  │                                                                   │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │   │ Beat     │  │ Scene    │  │ Clip     │  │ Video    │        │   │
│  │   │ Detection│─▶│ Detection│─▶│ Assembly │─▶│ Rendering│        │   │
│  │   │ (librosa)│  │(scenedet)│  │          │  │ (FFmpeg) │        │   │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘        │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Enhancement Pipeline                           │   │
│  │                                                                   │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │   │ Stabilize│  │ Upscale  │  │ Color    │  │ Sharpen  │        │   │
│  │   │ (FFmpeg) │  │(ESRGAN)  │  │ Grade    │  │          │        │   │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘        │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                               │                                          │
└───────────────────────────────┼──────────────────────────────────────────┘
                               ▼
                        /data/output/
                        montage_001.mp4
```

---

## Hybrid Architecture (Cloud Offloading)

For systems with limited resources (e.g., laptops), Montage AI supports a hybrid mode where heavy compute tasks are offloaded to the cloud via `cgpu`.

See [Hybrid Workflow](hybrid-workflow.md) for details.

- **LLM Inference:** Offloaded to Google Gemini via `cgpu serve`.
- **Upscaling:** Offloaded to Google Colab GPUs via `cgpu run`.
- **Local:** Orchestration, cutting, and basic rendering.

## Module Responsibilities

### Core Pipeline (src/montage_ai/core/)

The editing engine has been refactored into a modular pipeline.

**MontageBuilder (`montage_builder.py`)**
The central orchestrator that executes the editing pipeline in phases:
1. **Setup:** Initialize workspace and logging.
2. **Analyze:** Process audio (beats/energy) and video (scenes/content).
3. **Plan:** Select clips and map them to the timeline based on the story arc.
4. **Enhance:** Apply stabilization, upscaling, and color grading.
5. **Render:** Generate the final video file.

**Components:**

| Module | Purpose |
|--------|---------|
| `audio_analysis.py` | Beat detection, tempo extraction, energy profiling (librosa + FFmpeg fallback) |
| `scene_analysis.py` | Scene detection, content analysis, visual similarity with LRU cache |
| `video_metadata.py` | Technical metadata extraction (ffprobe wrapper) |
| `clip_enhancement.py` | Stabilization, upscaling, color matching (Local/Cloud hybrid) |
| `ffmpeg_config.py` | GPU encoder detection (NVENC/VAAPI/QSV), encoding parameters |

### Performance Optimizations

| Optimization | Implementation | Impact |
|--------------|----------------|--------|
| **LRU Histogram Cache** | `@lru_cache` for frame extraction | 91% cache hit rate, 2-3x faster clip selection |
| **Parallel Scene Detection** | `ThreadPoolExecutor(max_workers=4)` | 3-4x speedup on multi-core |
| **FFmpeg Beat Detection** | `silencedetect` + `ebur128` filters | Works without librosa (Python 3.12 compat) |
| **Auto GPU Encoding** | NVENC > VAAPI > QSV > CPU | 2-6x encoding speedup |

### editor.py (CLI Entry Point)

A thin wrapper that initializes the `MontageBuilder` and handles CLI arguments.

---

### creative_director.py (LLM Interface)

Translates natural language to editing parameters.

**Responsibilities:**

- Parse user prompts
- Query LLM (Ollama or Gemini)
- Validate JSON responses
- Map to style parameters

**Backends:**

| Backend     | Protocol          | Model            |
| ----------- | ----------------- | ---------------- |
| Ollama      | REST API          | llama3.1:70b     |
| cgpu/Gemini | OpenAI-compatible | gemini-2.0-flash |

**Flow:**

```text
User Prompt
    │
    ▼
┌─────────────────┐
│ System Prompt   │
│ + Style Options │
│ + Examples      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM (Ollama or  │
│ Gemini)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ JSON Validation │
│ + Schema Check  │
└────────┬────────┘
         │
         ▼
Editing Instructions
```

---

### footage_manager.py (Clip Selection)

Professional-grade clip management with story arc awareness.

**Key concepts:**

| Concept                | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| **UsageStatus**        | Track if clip is UNUSED, USED, or RESERVED                     |
| **SceneType**          | Classify clips: ESTABLISHING, ACTION, DETAIL, PORTRAIT, SCENIC |
| **StoryPhase**         | Timeline position: INTRO, BUILD, CLIMAX, SUSTAIN, OUTRO        |
| **FootageClip**        | Data class with all clip metadata                              |
| **FootagePoolManager** | Manages available clip pool                                    |
| **StoryArcController** | Maps timeline position to requirements                         |

**Selection algorithm:**

```text
Current Position → Story Phase → Required Energy + Scene Type
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │ Score Available Clips │
                               │ - Energy match        │
                               │ - Scene type match    │
                               │ - Visual interest     │
                               │ - Variety bonus       │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               Select Highest Score
                               Mark as USED
```

---

### style_templates.py (Style Loader)

Loads and validates JSON style presets.

**Responsibilities:**

- Discover preset files
- Parse and validate JSON
- Merge defaults with overrides
- Cache loaded templates

**File discovery order:**

1. Built-in: `src/montage_ai/styles/*.json`
2. Env override: `STYLE_PRESET_DIR/*.json`
3. Single file: `STYLE_PRESET_PATH`

---

### cgpu_upscaler.py (Cloud GPU)

Offloads AI upscaling to free cloud GPUs.

**Flow:**

```text
Video Frames (local)
        │
        ▼
┌─────────────────┐
│ cgpu connect    │
│ (Google Colab)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Real-ESRGAN     │
│ (T4/A100 GPU)   │
└────────┬────────┘
         │
         ▼
Upscaled Frames (local)
```

---

### monitoring.py (Logging)

Real-time decision logging for debugging.

**Logged events:**

- Clip selection decisions
- Beat alignment choices
- Energy level changes
- Phase transitions
- Performance metrics

---

## Data Flow

### Input Processing

```text
/data/input/*.mp4
        │
        ├──▶ Scene Detection ──▶ Scene List
        │
        ├──▶ Energy Analysis ──▶ Clip Scores
        │
        └──▶ Metadata Extraction ──▶ Clip Database
```

### Audio Processing

```text
/data/music/*.mp3
        │
        ├──▶ Beat Detection ──▶ Beat Timestamps
        │
        ├──▶ Tempo Analysis ──▶ BPM
        │
        └──▶ Energy Curve ──▶ Energy Timeline
```

### Assembly

```text
Beat Timeline + Clip Database + Style Parameters
                      │
                      ▼
            ┌─────────────────┐
            │ For each beat:  │
            │ - Get story     │
            │   phase         │
            │ - Score clips   │
            │ - Select best   │
            │ - Mark used     │
            └────────┬────────┘
                     │
                     ▼
              Clip Sequence
```

### Rendering

```text
Clip Sequence
      │
      ├──▶ Crop/Scale to STANDARD_WIDTH x STANDARD_HEIGHT (1080x1920)
      │
      ├──▶ Optional: Stabilize (vidstab 2-pass)
      │
      ├──▶ Optional: Upscale (Real-ESRGAN)
      │
      ├──▶ Color Grade (20+ presets)
      │
      └──▶ Progressive Renderer
            │
            ├──▶ Batch clips (default 25)
            ├──▶ Write segments to disk
            ├──▶ FFmpeg concat (-c copy)
            ├──▶ Optional: xfade transitions
            └──▶ Audio mix + Logo overlay
                    │
                    ▼
            /data/output/montage.mp4
```

---

## Memory-Efficient Progressive Rendering

The system uses `ProgressiveRenderer` (in `segment_writer.py`) to prevent OOM crashes.

```text
┌─────────────────────────────────────────────────────────────┐
│                    Progressive Renderer                      │
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Clip 1  │   │ Clip 2  │   │ ...     │   │ Clip 25 │     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
│       │             │             │             │            │
│       └─────────────┴─────────────┴─────────────┘            │
│                         │                                    │
│                         ▼                                    │
│              ┌──────────────────┐                           │
│              │ flush_batch()    │                           │
│              │ - Normalize      │                           │
│              │ - Write segment  │                           │
│              │ - GC + cleanup   │                           │
│              └────────┬─────────┘                           │
│                       │                                      │
│                       ▼                                      │
│              segment_0001.mp4 (disk)                        │
│                                                              │
│  ... repeat for all batches ...                             │
│                                                              │
│              ┌──────────────────┐                           │
│              │ finalize()       │                           │
│              │ - FFmpeg concat  │                           │
│              │ - Audio mix      │                           │
│              │ - Logo overlay   │                           │
│              └────────┬─────────┘                           │
│                       │                                      │
│                       ▼                                      │
│              output.mp4                                      │
└─────────────────────────────────────────────────────────────┘
```

**Key Constants (Dynamically Determined):**

These constants are automatically determined from input footage using `determine_output_profile()`:

| Constant           | Default | Determination Method                                           |
| ------------------ | ------- | -------------------------------------------------------------- |
| `STANDARD_WIDTH`   | 1080    | Weighted median of input widths, snapped to standard presets   |
| `STANDARD_HEIGHT`  | 1920    | Weighted median of input heights, snapped to standard presets  |
| `STANDARD_FPS`     | 30      | Weighted median of input frame rates                           |
| `STANDARD_PIX_FMT` | yuv420p | Dominant pixel format from inputs (by duration)                |
| `TARGET_CODEC`     | libx264 | Dominant codec from inputs, or env `OUTPUT_CODEC`              |
| `TARGET_PROFILE`   | high    | Auto-selected based on resolution (4.1 for HD, 5.1 for 4K)     |
| `TARGET_BITRATE`   | auto    | Weighted median of input bitrates, or calculated from pixels   |

**Output Profile Heuristics:**

- Orientation (horizontal/vertical/square) determined by weighted aspect ratios
- Resolution snapped to common presets (1080p, 720p, 4K) if within 12% variance
- Avoids upscaling beyond maximum input resolution
- Honors environment overrides: `OUTPUT_CODEC`, `OUTPUT_PIX_FMT`, `OUTPUT_PROFILE`, `OUTPUT_LEVEL`

---

## External Dependencies

| Dependency  | Purpose                   | Version |
| ----------- | ------------------------- | ------- |
| FFmpeg      | Video encoding/processing | Latest  |
| librosa     | Audio analysis            | 0.10+   |
| MoviePy     | Video manipulation        | 1.0+    |
| OpenCV      | Frame processing          | 4.0+    |
| scenedetect | Scene detection           | 0.6+    |
| Real-ESRGAN | AI upscaling              | Latest  |
| OpenAI SDK  | cgpu/Gemini client        | 1.0+    |

---

## Scaling Considerations

### Parallel Processing

- Clip enhancement runs in parallel threads
- Frame upscaling can use cloud GPU
- FFmpeg uses multi-threading

### Memory Management

- Clips loaded on-demand, not all at once
- Temporary files cleaned after processing
- Large videos processed in chunks

### GPU Utilization

- Auto-detection of available GPU encoders
- Fallback chain: Vulkan → V4L2 → CPU
- Cloud GPU option for heavy workloads
