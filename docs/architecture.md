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

## Module Responsibilities

### editor.py (Main Orchestrator)

The central module that coordinates all processing.

**Responsibilities:**

- Load and validate input clips
- Perform beat detection and energy analysis
- Coordinate scene detection
- Execute clip selection via Footage Manager
- Apply visual effects
- Render final output via FFmpeg

**Key functions:**

| Function              | Purpose                                 |
| --------------------- | --------------------------------------- |
| `detect_beats()`      | Extract beats, tempo, energy from audio |
| `detect_scenes()`     | Find scene changes in clips             |
| `assemble_timeline()` | Build clip sequence                     |
| `render_output()`     | Export final video                      |
| `upscale_clip()`      | Apply AI upscaling                      |
| `stabilize_clip()`    | Apply stabilization                     |

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

**Key Constants (DRY):**
| Constant           | Value   | Purpose                      |
| ------------------ | ------- | ---------------------------- |
| `STANDARD_WIDTH`   | 1080    | Target width (9:16 vertical) |
| `STANDARD_HEIGHT`  | 1920    | Target height                |
| `STANDARD_FPS`     | 30      | Frame rate                   |
| `STANDARD_PIX_FMT` | yuv420p | Pixel format                 |
| `STANDARD_PROFILE` | high    | H.264 profile                |

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
