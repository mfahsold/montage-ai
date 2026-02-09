# Features

> **Philosophy:** We do not generate pixels. We polish them.

Complete guide to Montage AI capabilities.

---

## 🆕 2026 Releases

### v1.1 Enhancement Suite (January 2026)

Professional video enhancement with full NLE compatibility.

#### 🔧 High-Resolution & Edge Case Support (Phase 4)

**Resolution Support:**
- ✅ **1080p**: Fully optimized (default batch_size=5)
- ✅ **4K**: Fully supported (adaptive batch_size=2)
- ⚠️ **6K** (6144x3160): Supported with automatic memory optimization (batch_size=1)
- ⚠️ **8K** (7680x4320): Requires proxy workflow (automatic error with instructions)

**RAW Codec Detection:**
Montage AI automatically detects professional RAW formats and provides guidance:
- **ProRes RAW** (requires FFmpeg with `--enable-libprores_raw`)
- **Blackmagic RAW** (requires Blackmagic RAW SDK)
- **RED RAW** (requires REDline SDK or FFmpeg plugin)
- **CinemaDNG** (requires FFmpeg with `--enable-libraw`)
- **ARRIRAW** (requires ARRI SDK)

**Automatic Optimizations:**
| Resolution | Memory/Frame | Batch Size | H.264 Level | HEVC Level |
|------------|--------------|------------|-------------|------------|
| 1080p | 6.2 MB | 5 | 4.1 | 4.1 |
| 4K | 24.8 MB | 2 | 5.0/5.1 | 5.0/5.1 |
| 6K | 58.2 MB | 1 | ❌ | 5.2 |
| 8K | 99.5 MB | ❌ Proxy | ❌ | 6.2 |

**Usage Examples:**

```bash
# 6K material (automatic handling)
export INPUT_DIR=/data/input_6k
./montage-ai.sh run
# → Batch size automatically reduced to 1
# → HEVC Level 5.2 selected automatically

# ProRes RAW (with warning)
export INPUT_DIR=/data/prores_raw
./montage-ai.sh run
# → Warning: "Consider generating H.264/H.265 proxies"

# 8K material (requires proxy workflow)
export INPUT_DIR=/data/8k
./montage-ai.sh run
# → Error with proxy generation instructions

# Generate proxies for 6K+ material (via CLI wrapper)
./montage-ai.sh generate-proxies \
    --input /data/8k/*.mp4 \
    --output /data/proxies \
    --format h264 \
    --scale 1920:-1 \
    --preset fast

# Or directly via Python module
python -m montage_ai.proxy_generator \
    --input /data/8k/*.mp4 \
    --output /data/proxies \
    --format h264 \
    --scale 1920:-1 \
    --preset fast
```

**Recommendations:**
- **6K workflows**: Native support works, but proxy workflow recommended for faster iteration
- **8K workflows**: Proxy workflow mandatory (1080p → conform to 8K in DaVinci Resolve/FCPX)
- **RAW codecs**: Generate H.264/H.265 proxies first for best compatibility

---

#### AI Denoising

Reduce noise while preserving film grain texture.

**Methods:**
- **hqdn3d:** Fast temporal/spatial denoising (default)
- **nlmeans:** High-quality non-local means (slower, better)

**Parameters:**
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `temporal_strength` | 0.0-1.0 | 0.5 | Cross-frame reduction |
| `spatial_strength` | 0.0-1.0 | 0.3 | In-frame reduction |
| `chroma_strength` | 0.0-1.0 | 0.5 | Color noise reduction |
| `preserve_grain` | 0.0-1.0 | 0.2 | Keep film texture |

**Usage:**
```bash
# CLI flag (simplest)
./montage-ai.sh run hitchcock --denoise

# Custom strength via environment
DENOISE_SPATIAL=0.4 DENOISE_TEMPORAL=0.6 ./montage-ai.sh run

# Docker Compose
docker compose run --rm -e DENOISE=true -e DENOISE_SPATIAL=0.4 montage-ai /app/montage-ai.sh run
```

See [Configuration: Denoising](configuration.md#denoising) for all variables.

#### Film Grain Simulation

Add authentic film grain for cinematic look.

**Presets:**
| Type | Character | Use Case |
|------|-----------|----------|
| `fine` | Subtle texture | Modern digital |
| `medium` | Visible grain | Music videos |
| `35mm` | Classic cinema | Feature films |
| `16mm` | Documentary | Indie, documentary |
| `8mm` | Vintage home movie | Retro aesthetics |

**Usage:**
```bash
# CLI flag
./montage-ai.sh run --film-grain 35mm

# Environment
FILM_GRAIN=16mm ./montage-ai.sh run documentary
```

See [Configuration: Film Grain](configuration.md#film-grain) for all variables.

#### Dialogue Ducking

Auto-detect speech and lower background music.

**How it works:**
1. VAD (Voice Activity Detection) finds speech segments
2. Generates volume keyframes with attack/release curves
3. Exports NLE-compatible automation

**Detection Methods:**
| Method | Quality | Speed | Requirements |
|--------|---------|-------|--------------|
| Silero | Excellent | Medium | torch |
| WebRTC | Good | Fast | webrtcvad |
| FFmpeg | Basic | Fast | Built-in |

**Parameters:**
| Setting | Default | Description |
|---------|---------|-------------|
| `duck_level_db` | -12 | Volume reduction |
| `attack_time` | 0.15s | Ramp down time |
| `release_time` | 0.30s | Ramp up time |

**Export:** CSV, DaVinci Resolve text, Premiere XML

#### Enhancement Tracking & NLE Export

Every AI decision is tracked and exportable.

**What's tracked:**
- Stabilization (method, smoothing, crop mode)
- Upscaling (model, scale factor)
- Denoising (spatial/temporal strength)
- Sharpening (amount, radius)
- Color grading (preset, intensity, LUT)
- Color matching (reference clip, method)
- Film grain (type, intensity)
- Dialogue ducking (keyframes, segments)

**Export formats:**
- **OTIO metadata:** Full parameters in clip metadata
- **EDL comments:** `* MONTAGE_AI DENOISE: spatial=0.3`
- **Recipe cards:** Human-readable Markdown instructions

### v1.2 Shorts Studio 2.0 (February 2026)

End-to-end vertical video automation with smart reframing and highlight detection.

#### Smart Reframing v2

Intelligent 16:9 to 9:16 conversion keeping the subject in frame.

**New capabilities:**
- **Subject Tracking**: Kalman Filter smoothing prevents jerky camera movements
- **Motion Optimization**: `CameraMotionOptimizer` balances subject centering vs. stability
- **Fallback Modes**: Auto-switches to Action Tracking or Center Crop if face detection fails

#### Styles & Captions

Apply trending visual styles instantly.

**Caption Presets:**
| Style | Description |
|-------|-------------|
| `TikTok` | Classic white text with black outline |
| `Bold` | Heavy font, neon colors, high contrast |
| `Minimal` | Clean sans-serif, lower third placement |
| `Cinematic` | Letterboxed, serif font, yellow subtitles |

#### Highlight Detection

AI-powered identification of "viral moments" in long-form content.

**Signals:**
- **Audio Energy**: Loudness, laughter, excitement spikes
- **Visual Action**: High motion content (Optical Flow)
- **Face Presence**: Close-ups and reaction shots

### v1.3 Pro Polish (March 2026)

Professional finishing tools for audio and workflow integration.

#### Audio Polish Suite

Studio-quality audio enhancement pipeline.

- **Voice Isolation:** EQ + Compression + Limiting + Noise Gate to clean up dialogue.
- **Auto-Ducking:** Automatically lowers background music volume during speech segments (Sidechain Compression).
- **SNR Check:** (Beta) Analyzes Signal-to-Noise Ratio to flag poor audio.

#### Professional Export

Workflow tools for NLE integration.

- **OTIO Export:** Generate OpenTimelineIO files for DaVinci Resolve / Premiere Pro.
- **Proxy Generation:** Auto-creates lightweight H.264/ProRes proxies linked to the timeline.
- **Timeline Integrity:** Relink guides ensure proxies and source files match perfectly.

#### Performance Tuning

- **Fast Preview:** "Time-to-First-Preview" optimized to < 3 minutes via `ultrafast` preset and parallel processing.
- **Visual Action**: High motion scenes
- **Face Presence**: Clear, expressive faces
- **Score Fusion**: `(Audio * 0.4) + (Action * 0.3) + (Face * 0.3)`

**Review UI (planned):**
- **Interactive Cards**: Click to jump to highlight
- **Score Breakdown**: See why a moment was picked (Action/Face/Audio)
- **Visual Cues**: Color-coded confidence scores

**Usage:**
Use the CLI/API for now (Web UI integration in progress):
```bash
./montage-ai.sh run --workflow shorts --audio-aware
```

**Recipe Card Example:**
```markdown
## Clip: DJI_0042.MP4
### DaVinci Resolve Recreation:
1. **Stabilizer** (Color Page > Tracker)
   - Mode: Perspective
   - Smoothing: 0.30
2. **Noise Reduction** (Color Page > Spatial NR)
   - Luma Threshold: 3.0
3. **Color Wheels** (Color Page > Primary)
   - Saturation: 115%
```

---

### Story Engine (Narrative Arc)

AI-driven 5-phase narrative structure for cinematic storytelling.

**Story Arc Phases:**
| Phase | Position | Energy | Purpose |
|-------|----------|--------|---------|
| Intro | 0-15% | Low-Medium | Establish context |
| Build | 15-40% | Rising | Develop tension |
| Climax | 40-70% | High | Peak intensity |
| Sustain | 70-90% | High-Medium | Maintain engagement |
| Outro | 90-100% | Descending | Resolution |

**Arc Presets:**
- **hero_journey:** Classic narrative arc
- **mtv_energy:** Peak early, sustain high
- **documentary:** Gradual reveal, observational
- **thriller:** Slow build, explosive release
- **flat:** Consistent energy throughout

**How it works:**
1. Analyzes all clips for visual tension (motion, faces, objects)
2. Maps clips to story phases based on energy fit
3. Uses CSP solver for optimal clip placement
4. Ensures narrative coherence across the timeline

**Usage:**
```bash
./montage-ai.sh run hitchcock --story-engine --story-arc thriller
```

---

### Transcript Editor

Edit video by removing text. AI handles the cuts.

**Workflow:**
1. Upload video → Auto-transcribe (Whisper)
2. View transcript with word-level timestamps
3. Delete words to remove segments
4. Rearrange to reorder scenes
5. Export video or OTIO timeline

**Capabilities:**
- **Live Preview:** 360p preview updates 2 seconds after edits
- **Word-Level Sync:** Click any word to seek
- **Filler Detection:** Highlights "um", "uh", "like" for removal
- **Silence Removal:** Auto-gap detection
- **Export:** MP4, EDL (Premiere), OTIO (Resolve)

**Access:** Web UI coming soon (use CLI/API until integrated)

---

### Pro Handoff (Timeline Export)

Move from Montage AI to professional NLEs (DaVinci, Premiere, Resolve).

**Formats:**
- **OTIO (.otio):** Native for Resolve, Premiere, Nuke
- **FCP XML (.xml):** Universal standard
- **EDL (.edl):** Legacy fallback

**Features:**
- Source relinking to original high-res files
- **Smart Proxies:** H.264 (SOTA optimized for scrubbing), ProRes, DNxHR
- Conform guide with step-by-step instructions
- Seamless roundtrip via OTIO

---

### Shorts Studio

Auto-reframe to 9:16 for TikTok, Instagram, YouTube Shorts.

> UI preview elements are planned; CLI/API workflows are available today.

**Preview:**
- Live 9:16 phone frame
- Safe zone overlays (title, action, platform UI)
- Platform guides (TikTok, Instagram, YouTube)

**Tracking Modes:**
- **Auto:** AI detects and follows subject
- **Face:** Face detection for talking heads
- **Center:** Simple center crop
- **Custom:** Manual keyframes

**Smart Features:**
- **Cinema Path:** Convex optimization for fluid camera motion
- **Subject Safety:** Keeps subjects in golden zone
- **Voice Isolation:** Demucs for clean dialogue (denoising)
- **Captions:** TikTok, Minimal, Bold, Karaoke styles
- **Highlights:** Auto-detect best moments by energy/motion/faces

**Access:** Web UI coming soon (use CLI/API until integrated)

---

### Quality Profiles

One selection replaces multiple toggles. Choose based on your goal, not technical details.

| Profile | Resolution | Enhancements | Use Case |
|---------|------------|--------------|----------|
| 🚀 **Preview** | 360p | None | Fast iteration (Ultrafast preset) |
| 📺 **Standard** | 1080p | Color grading | Social media, general use |
| ✨ **High** | 1080p | Grading + stabilization | Professional delivery |
| 🎬 **Master** | 4K | All + AI upscaling | Broadcast, cinema, archival |

**What each profile enables:**

```
Preview:   enhance=false, stabilize=false, upscale=false, resolution=360p, preset=ultrafast
Standard:  enhance=true,  stabilize=false, upscale=false, resolution=1080p
High:      enhance=true,  stabilize=true,  upscale=false, resolution=1080p
Master:    enhance=true,  stabilize=true,  upscale=true,  resolution=4k
```

**Usage:**
```bash
# CLI
./montage-ai.sh preview hitchcock   # Fast 360p render
./montage-ai.sh finalize hitchcock  # Upgrade to High Quality
./montage-ai.sh run hitchcock --quality high

# Environment variable
QUALITY_PROFILE=master ./montage-ai.sh run

# Web UI: Select from Quality Profile cards
```

### Preview-First Workflow

Iterate faster by separating creative decisions from rendering time.

1.  **Auto-Preview:** Upload clips and get a 360p rough cut in seconds.
2.  **Review:** Check pacing, music sync, and story arc immediately.
3.  **Finalize:** Click "Finalize (1080p)" to render the master copy with full stabilization and enhancement.

---

### Cloud Acceleration

Single toggle for all cloud GPU features with graceful local fallback.

**What it enables:**
- AI upscaling via cloud GPU (Real-ESRGAN on H100/A100)
- Fast transcription (Whisper large model)
- LLM creative direction (Gemini Pro)

**Fallback behavior:**
```
Cloud available?  → Use cloud GPU
Cloud unavailable → Fall back to local processing
Local GPU?        → Use Vulkan acceleration  
CPU only?         → Use optimized CPU path
```

**Privacy guarantee:** Only enabled features use cloud. Raw footage stays local unless upscaling is enabled.

**Usage:**
```bash
# CLI
CLOUD_ACCELERATION=true ./montage-ai.sh run --upscale

# Web UI: Toggle "Cloud Acceleration" switch
```

---

## Timeline Export (Pro Handoff) {#timeline-export}

Export your AI rough cut to professional NLEs for finishing.

**Supported formats:**
- **OTIO** — OpenTimelineIO, preferred for modern NLEs
- **EDL** — Edit Decision List, legacy support
- **CSV** — Spreadsheet review, logging
- **JSON** — Metadata, automation

**Usage:**
```bash
./montage-ai.sh run hitchcock --export-timeline --generate-proxies
```

**Outputs in `data/output/`:**
- `montage.otio` — Timeline file
- `montage.edl` — Legacy EDL
- `montage.csv` — Cut log
- `montage_metadata.json` — Full metadata
- `proxies/` — Optional low-res clips for offline editing

**NLE Import:**
| NLE | Recommended Format | Notes |
|-----|-------------------|-------|
| DaVinci Resolve | OTIO | File → Import → Timeline |
| Premiere Pro | OTIO or EDL | May need media relink |
| Final Cut Pro | OTIO | Via third-party plugin |
| Avid Media Composer | EDL | Relink originals |

---

## Feature Matrix

| Feature | CLI Flag | Env Variable | Status |
|---------|----------|-------------|--------|
| Beat-synced editing | *(default)* | `CUT_STYLE` | Stable |
| Scene detection | *(default)* | — | Stable |
| Style templates | `run [STYLE]` | `CUT_STYLE` | Stable |
| Video stabilization | `--stabilize` | `STABILIZE=true` | Stable |
| AI upscaling | `--upscale` | `UPSCALE=true` | Stable |
| AI denoising | `--denoise` | `DENOISE=true` | Stable |
| Film grain | `--film-grain [TYPE]` | `FILM_GRAIN=35mm` | Stable |
| Color grading | `--color-grade [PRESET]` | `COLOR_GRADING=teal_orange` | Stable |
| Dialogue ducking | `--dialogue-duck` | — | Stable |
| Audio normalize | `--audio-normalize` | — | Stable |
| Voice isolation | `--isolate-voice` | `CGPU_ENABLED=true` | Requires cgpu |
| Caption burning | `--captions [STYLE]` | — | Stable |
| Smart reframing (9:16) | `shorts [STYLE]` | — | Stable |
| Story engine | `--story-engine` | — | Stable |
| Timeline export | `--export` | `EXPORT_TIMELINE=true` | Stable |
| Proxy generation | `generate-proxies` | `GENERATE_PROXIES=true` | Stable |
| Creative loop (LLM) | — | `CREATIVE_LOOP=true` | Requires LLM |
| Cloud acceleration | `--cgpu --cgpu-gpu` | `CGPU_ENABLED=true` | Optional |
| Quality profiles | `preview` / `hq` | `QUALITY_PROFILE=high` | Stable |

---

## CLI Examples

```bash
# Basic montage with default style
./montage-ai.sh run

# Hitchcock style with stabilization
./montage-ai.sh run hitchcock --stabilize

# Auto-reframe to 9:16 with captions
./montage-ai.sh shorts viral --captions tiktok

# Denoise + film grain + color grade
./montage-ai.sh run documentary --denoise --film-grain 16mm --color-grade warm

# Dialogue ducking + audio normalize
./montage-ai.sh run --dialogue-duck --audio-normalize

# Quick preview (360p, fast)
./montage-ai.sh preview mtv

# High quality with timeline export
./montage-ai.sh hq hitchcock --export

# Cloud GPU upscaling
./montage-ai.sh run --cgpu --cgpu-gpu --upscale

# Story engine with narrative arc
./montage-ai.sh run hitchcock --story-engine --story-arc thriller

# Generate proxies for 8K footage
./montage-ai.sh generate-proxies --input /data/8k/*.mp4 --output /data/proxies
```

---

## Core Editing

- Beat-synced cuts using FFmpeg (`astats`/tempo) — librosa optional
- Style-aware pacing, transitions, and color looks
- Story arc shaping (intro → build → climax → outro)
- LLM-powered "Creative Director" (Ollama local or Gemini via cgpu)
- **Agentic Creative Loop** for iterative quality refinement

## Style Templates (Built-in)

| Style          | Best for                 | Traits                                      |
| -------------- | ------------------------ | ------------------------------------------- |
| `dynamic`      | General purpose          | Adapts to music energy                      |
| `hitchcock`    | Thrillers, reveals       | Slow build, explosive climax, high contrast |
| `mtv`          | Music videos, dance      | 1-2 beat cuts, vibrant, hard cuts only      |
| `action`       | Sports, adventure        | Fast pacing, motion preference              |
| `documentary`  | Travel, interviews       | Natural pacing, mixed transitions           |
| `minimalist`   | Art house, meditation    | Very slow, desaturated, long takes          |
| `wes_anderson` | Quirky, aesthetic pieces | Symmetry bias, warm pastel look             |

### Custom Styles (JSON)

Place JSON in `src/montage_ai/styles/` or point to it:

```bash
STYLE_PRESET_PATH=/path/to/my_style.json ./montage-ai.sh run my_style
# or whole directory
STYLE_PRESET_DIR=/path/to/styles ./montage-ai.sh run my_style
```

Minimal schema:

```json
{
  "id": "my_style",
  "description": "Energetic vlog",
  "params": {
    "pacing": {"speed": "fast", "variation": "moderate"},
    "transitions": {"type": "hard_cuts"},
    "effects": {"color_grading": "vibrant", "stabilization": false}
  }
}
```

## Web UI (Fastest path)

```bash
docker compose up
# open <MONTAGE_WEB_URL>
```


Flow: upload videos + music → pick style or prompt → toggle enhance/stabilize/upscale/cloud GPU → Create Montage → download MP4 (and timeline if enabled).

Useful endpoints (for automation):

- `GET /api/status` – health
- `GET /api/files` – list uploads
- `POST /api/upload` (multipart, fields: `file`, `type=video|music`)
- `POST /api/jobs` – create job with JSON body (`style`, `prompt`, `stabilize`, `upscale`, `cgpu`, `export_timeline`, ...)
- `GET /api/jobs/{id}` – job status
- `GET /api/download/{filename}` – download outputs

## Responsible AI & Transparency

- **Local-first processing** with opt-in cloud GPU/LLM
- **No training on user footage**
- **Decision logs** available via `EXPORT_DECISIONS=true`
- **Transparency payload** at `GET /api/transparency`

See [responsible_ai.md](responsible_ai.md) for the full policy.

## Cloud LLM & GPU (cgpu)

- Install: `npm i -g cgpu` (plus gemini-cli; run `cgpu connect` once)
- Enable Gemini LLM: `CGPU_ENABLED=true ./montage-ai.sh run --cgpu`
- Enable Colab GPU upscaling: `CGPU_GPU_ENABLED=true ./montage-ai.sh run --upscale --cgpu-gpu`

Fallback order for upscaling: cgpu T4/A100 → local Vulkan GPU → FFmpeg Lanczos (CPU).

## Creative Loop (Agentic Refinement)

When enabled, the LLM evaluates each cut and suggests improvements:

```bash
CREATIVE_LOOP=true ./montage-ai.sh run hitchcock
```

**How it works:**
1. First cut is built with initial editing instructions
2. LLM evaluates pacing, variety, energy, transitions
3. If satisfaction score < 80%, adjustments are applied
4. Process repeats until approved or max iterations (default: 3)

**Evaluation criteria:**
- **Pacing:** Does cut rhythm match the style and music energy?
- **Variety:** Enough shot variation? No jump cuts or repetition?
- **Energy:** Fast cuts on high-energy sections, breathing room on calm ones?
- **Story Arc:** Does the edit follow intro → build → climax → outro?

See [configuration.md](configuration.md#creative-loop-agentic-refinement) for all options.

## Shorts Workflow (Vertical Video) {#shorts-workflow}

> **Note:** Shorts Studio UI is in progress; this section covers CLI usage.

- **Smart Reframing**: Automatically crops horizontal footage to 9:16 vertical aspect ratio using face detection and segmented tracking.
- **Segmented Tracking**: Stabilizes camera movement by keeping the crop window static until the subject moves significantly, preventing jitter.
- **Auto-Captions**: Generates and burns in subtitles (requires `whisper`).  
  **Styles:** `tiktok` (Bold/Shadowed), `minimal` (Clean), `cinematic` (Serif/Boxed), `bold` (Impact).
- **Web UI Integration**: In progress (use CLI/API in the meantime).

**CLI usage:**
```bash
# Basic vertical output
./montage-ai.sh run viral --aspect 9:16

# With captions
./montage-ai.sh run viral --aspect 9:16 --captions

# High quality shorts
./montage-ai.sh run viral --aspect 9:16 --quality high --captions
```

---

## API Reference

The Web UI exposes a REST API for automation:

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Health check |
| `/api/files` | GET | List uploaded files |
| `/api/upload` | POST | Upload video/music (multipart) |
| `/api/jobs` | POST | Create montage job |
| `/api/jobs/{id}` | GET | Job status |
| `/api/download/{file}` | GET | Download output |

### Transcript Editor

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/transcript/upload` | POST | Upload video for editing |
| `/api/transcript/transcribe` | POST | Generate transcript |
| `/api/transcript/export` | POST | Export edited video/EDL/OTIO |

**Export formats:** `video`, `edl`, `otio`

### Shorts Studio

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/shorts/upload` | POST | Upload video for shorts |
| `/api/shorts/analyze` | POST | Analyze for smart reframing |
| `/api/shorts/highlights` | POST | Detect highlight moments |
| `/api/shorts/render` | POST | Render vertical video |
| `/api/shorts/create` | POST | Alias for render |

**Highlight types:** Energy, Drop, Speech, Beat

### Audio Polish

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/audio/clean` | POST | One-click voice isolation + noise reduction |
| `/api/audio/analyze` | POST | Analyze audio quality, get recommendations |

### Quality Profiles

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/quality-profiles` | GET | Get available profiles |
| `/api/cloud/status` | GET | Check cloud acceleration availability |

**Example: Create a job via API**
```bash
MONTAGE_API_BASE="http://<MONTAGE_API_HOST>"
curl -X POST "${MONTAGE_API_BASE}/api/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "style": "hitchcock",
    "quality_profile": "high",
    "cloud_acceleration": false,
    "export_timeline": true
  }'
```

**Example: Clean audio**
```bash
curl -X POST "${MONTAGE_API_BASE}/api/audio/clean" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/data/output/my_video.mp4",
    "isolate_voice": true,
    "reduce_noise": true
  }'
```

**Example: Detect highlights**
```bash
curl -X POST "${MONTAGE_API_BASE}/api/shorts/highlights" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/data/output/my_video.mp4",
    "max_clips": 5,
    "min_duration": 5,
    "include_speech": true
  }'
```

---

## Troubleshooting

Having issues? Check [troubleshooting.md](troubleshooting.md) for common fixes.

---

## See Also

- [Configuration](configuration.md) — All settings explained
- [Architecture](architecture.md) — How it works under the hood
