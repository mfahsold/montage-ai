# Features

Montage AI combines AI-powered editing decisions with professional video processing. This document explains each feature in detail.

---

## 🎬 Beat-Synchronized Editing

**What it does:** Automatically detects music beats and aligns video cuts to the rhythm.

**How it works:**
1. Audio analysis via [librosa](https://librosa.org/) extracts beat positions, tempo (BPM), and energy levels
2. Each beat becomes a potential cut point
3. The system selects which beats to cut on based on the chosen style
4. Higher energy sections get faster cuts; calmer sections get longer takes

**Configuration:**
```bash
# Tempo detection is automatic, but you can influence cut behavior:
CUT_STYLE=dynamic     # Adapts to music energy (default)
CUT_STYLE=fast        # Cuts on every beat
CUT_STYLE=slow        # Cuts every 4-8 beats
```

---

## 🎭 Cinematic Style Templates

**What it does:** Applies professional editing patterns from famous directors and genres.

**Available styles:**

| Style          | Inspiration      | Characteristics                                     |
| -------------- | ---------------- | --------------------------------------------------- |
| `hitchcock`    | Alfred Hitchcock | Slow tension build, explosive climax, high contrast |
| `mtv`          | Music videos     | Rapid 1-2 beat cuts, maximum energy, hard cuts only |
| `action`       | Michael Bay      | Fast cuts, motion preference, dynamic transitions   |
| `documentary`  | Ken Burns        | Natural pacing, longer takes, establishing shots    |
| `minimalist`   | Art house        | Contemplative long takes, calm mood                 |
| `wes_anderson` | Wes Anderson     | Symmetric framing, warm colors, playful pacing      |

**How styles work:**

Each style is a JSON preset with parameters for:

- **Pacing:** Cut speed, variation, intro/climax behavior
- **Cinematography:** Shot preferences, match cuts, invisible cuts
- **Transitions:** Hard cuts, crossfades, energy-aware mixing
- **Effects:** Color grading, stabilization, sharpening

**Custom styles:**

Create your own style presets as JSON files. See [Style Guide](styles.md) for details.

```bash
STYLE_PRESET_DIR=/path/to/my/styles ./montage-ai.sh run my_custom_style
```

---

## 🤖 Creative Director (Natural Language Control)

**What it does:** Describe your vision in plain English and the AI translates it to editing parameters.

**How it works:**

1. Your prompt goes to an LLM (Ollama local or Gemini cloud)
2. The LLM generates structured JSON editing instructions
3. Instructions are validated and applied to the editing engine

**Example prompts:**

```bash
# Via environment variable
CREATIVE_PROMPT="Edit this like a tense thriller with slow builds and sudden cuts"

# The LLM interprets this and generates:
{
  "style": {"name": "custom", "mood": "suspenseful"},
  "pacing": {"speed": "dynamic", "variation": "high"},
  "transitions": {"type": "hard_cuts"}
}
```

**Supported backends:**

| Backend                 | Setup                      | Pros                  | Cons                            |
| ----------------------- | -------------------------- | --------------------- | ------------------------------- |
| **Ollama** (local)      | Install Ollama, pull model | Private, no API costs | Requires 70B+ model for quality |
| **cgpu/Gemini** (cloud) | `npm i -g cgpu`            | Free, high quality    | Requires internet               |

**Configuration:**

```bash
# Local (Ollama)
OLLAMA_HOST=http://localhost:11434
DIRECTOR_MODEL=llama3.1:70b

# Cloud (Gemini via cgpu)
CGPU_ENABLED=true
CGPU_MODEL=gemini-2.0-flash
```

---

## 📖 Story Arc Awareness

**What it does:** Structures your montage like a narrative with intro, build-up, climax, and resolution.

**How it works:**

The **Footage Manager** divides the timeline into story phases:

| Phase       | Position | Energy  | Cut Rate | Clip Types           |
| ----------- | -------- | ------- | -------- | -------------------- |
| **Intro**   | 0-15%    | Low     | Slow     | Establishing, scenic |
| **Build**   | 15-40%   | Rising  | Medium   | Action, detail       |
| **Climax**  | 40-70%   | Peak    | Fast     | High action          |
| **Sustain** | 70-90%   | High    | Medium   | Mixed variety        |
| **Outro**   | 90-100%  | Falling | Slow     | Resolution, scenic   |

**Clip selection logic:**

- Each clip is "consumed" once (no repetition)
- Clips are scored for energy, visual interest, and scene type
- The system matches clips to the current story phase
- Variety scoring prevents repetitive sequences

**Scene types detected:**

- `ESTABLISHING` - Wide shots, environment
- `ACTION` - Movement, energy
- `DETAIL` - Close-ups, inserts
- `PORTRAIT` - People-focused
- `SCENIC` - Nature, atmosphere

---

## 🎨 Visual Enhancement Pipeline

### Stabilization

**What it does:** Removes camera shake from handheld footage.

**How it works:**

- Uses FFmpeg's `vidstabdetect` and `vidstabtransform` filters
- Two-pass process: analyze motion → apply correction
- Preserves intentional camera movement

**Usage:**

```bash
./montage-ai.sh run --stabilize
# Or via environment:
STABILIZE=true ./montage-ai.sh run
```

### AI Upscaling (Real-ESRGAN)

**What it does:** Increases resolution using neural networks (2x or 4x).

**How it works:**

1. Frames extracted from video
2. Each frame processed by Real-ESRGAN model
3. Upscaled frames reassembled into video

**Execution options:**

| Method               | Speed  | Setup             |
| -------------------- | ------ | ----------------- |
| **cgpu Cloud GPU**   | Fast   | `--cgpu-gpu` flag |
| **Local Vulkan GPU** | Medium | Auto-detected     |
| **CPU fallback**     | Slow   | Always available  |

**Usage:**

```bash
./montage-ai.sh run --upscale              # Local processing
./montage-ai.sh run --cgpu-gpu --upscale   # Cloud GPU (faster)
```

### Color Grading

**What it does:** Applies cinematic color looks automatically.

**Available grades:**

- `neutral` - No color change
- `warm` - Orange/yellow shift (golden hour look)
- `cool` - Blue shift (thriller/sci-fi look)
- `high_contrast` - Deep blacks, bright highlights
- `vibrant` - Saturated colors (music video look)
- `desaturated` - Muted, documentary feel

**Applied via style presets:**

```json
{"effects": {"color_grading": "high_contrast"}}
```

---

## ⚡ Hardware Acceleration

**What it does:** Uses GPU or specialized hardware for faster encoding.

**Auto-detected capabilities:**

| Hardware       | Encoder        | Use Case                          |
| -------------- | -------------- | --------------------------------- |
| **Vulkan GPU** | `h264_vulkan`  | Desktop GPUs (NVIDIA, AMD, Intel) |
| **V4L2**       | `h264_v4l2m2m` | Raspberry Pi, ARM SoCs            |
| **CPU**        | `libx264`      | Fallback (NEON optimized on ARM)  |

**Parallel processing:**

- Clip enhancement runs in parallel (`MAX_PARALLEL_JOBS`)
- FFmpeg threading auto-configured
- Leaves 2 CPU cores free for system stability

**Configuration:**

```bash
USE_GPU=auto          # Auto-detect (default)
USE_GPU=vulkan        # Force Vulkan
USE_GPU=none          # Force CPU only
MAX_PARALLEL_JOBS=4   # Parallel workers
FFMPEG_PRESET=fast    # Speed vs quality tradeoff
```

---

## 📤 Output Variants

**What it does:** Generates multiple versions of your montage with slight variations.

**How it works:**

- Same clips and music, different random seed
- Cut timing varies within beat-aligned windows
- Useful for A/B testing or client options

**Usage:**

```bash
./montage-ai.sh run --variants 3
# Creates: montage_001.mp4, montage_002.mp4, montage_003.mp4
```

---

## 🔍 Real-Time Monitoring

**What it does:** Logs every editing decision for debugging and analysis.

**Logged information:**

- Clip selection reasons and scores
- Beat alignment decisions
- Energy level at each cut
- Story phase transitions
- Performance metrics

**Usage:**

```bash
VERBOSE=true ./montage-ai.sh run
```

**Log output example:**

```text
[CLIP] Selected "beach_sunset.mp4" @ 2.3s
       → Phase: INTRO, Energy: 0.3, Score: 0.85
       → Reason: ESTABLISHING shot, high visual interest
[BEAT] Cut at 4.521s (beat #8, strength: 0.92)
[PHASE] Transitioning INTRO → BUILD at 15% position
```

---

## 🚧 Experimental Features

### Timeline Export (WIP)

Export to professional NLE formats:

- **OTIO** (OpenTimelineIO) - Universal format
- **EDL** - Edit Decision List
- **Proxy generation** - Low-res files for offline editing

```bash
EXPORT_TIMELINE=true GENERATE_PROXIES=true ./montage-ai.sh run
```

### Deep Footage Analysis (WIP)

AI-powered clip analysis:

- Object detection
- Face recognition
- Scene classification
- Motion intensity scoring

```bash
DEEP_ANALYSIS=true ./montage-ai.sh run
```

---

## Feature Matrix

| Feature           | Status   | Requires       |
| ----------------- | -------- | -------------- |
| Beat sync         | ✅ Stable | librosa        |
| Style templates   | ✅ Stable | -              |
| Creative Director | ✅ Stable | Ollama or cgpu |
| Story arc         | ✅ Stable | -              |
| Stabilization     | ✅ Stable | FFmpeg         |
| AI upscaling      | ✅ Stable | Real-ESRGAN    |
| Cloud GPU         | ✅ Stable | cgpu           |
| Hardware accel    | ✅ Stable | GPU drivers    |
| Timeline export   | 🚧 WIP    | opentimelineio |
| Deep analysis     | 🚧 WIP    | -              |
