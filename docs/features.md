# Features

> **Philosophy:** We do not generate pixels. We polish them.

Complete guide to Montage AI capabilities.

---

## 🆕 2026 Releases

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

**Access:** Web UI at `/transcript`

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

**Access:** Web UI at `/shorts`

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

## Core Editing

- Beat-synced cuts using `librosa` beat detection
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
make web              # or: docker compose -f docker-compose.web.yml up
# open http://localhost:5001
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

## Timeline Export (OTIO/EDL)

Enable during run:

```bash
./montage-ai.sh run hitchcock --export-timeline --generate-proxies
```

Outputs in `data/output/`:
- `*.otio` (preferred), `*.edl`, `*.csv`, metadata JSON, optional proxies folder.


Import tips:
- **DaVinci Resolve:** File → Import → Timeline → select `.otio`; relink media if paths differ.
- **Premiere Pro / Avid:** use `.edl` and relink originals.

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

> **Note:** The full Shorts Studio UI is available at `/shorts`. This section covers CLI usage.

- **Smart Reframing**: Automatically crops horizontal footage to 9:16 vertical aspect ratio using face detection and segmented tracking.
- **Segmented Tracking**: Stabilizes camera movement by keeping the crop window static until the subject moves significantly, preventing jitter.
- **Auto-Captions**: Generates and burns in subtitles (requires `whisper`).
- **Web UI Integration**: Toggle "Shorts Mode" in the Web UI for easy creation.

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
curl -X POST http://localhost:8080/api/jobs \
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
curl -X POST http://localhost:8080/api/audio/clean \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/data/output/my_video.mp4",
    "isolate_voice": true,
    "reduce_noise": true
  }'
```

**Example: Detect highlights**
```bash
curl -X POST http://localhost:8080/api/shorts/highlights \
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
- [Strategy](STRATEGY.md) — Product vision and roadmap
- [Backlog](BACKLOG.md) — Upcoming features
