# Montage AI - CLI Reference Guide

**Production-Ready Command Interface**

Full control over 27+ features: styles, enhancement, cloud integration, export, and more.

**✨ NEW (Jan 2026):** Audio normalization (-14 LUFS), variants generation, color intensity control fully integrated.

---

## Quick Start

```bash
# Web UI (recommended for first time)
docker compose up

# Preview (fast, 360p)
docker compose run --rm montage-ai ./montage-ai.sh preview dynamic

# Standard montage
docker compose run --rm montage-ai ./montage-ai.sh run hitchcock --stabilize --captions tiktok

# High quality with all effects
docker compose run --rm montage-ai ./montage-ai.sh hq documentary --upscale --isolate-voice --film-grain 35mm

# Generate 3 variants in parallel
docker compose run --rm montage-ai ./montage-ai.sh run dynamic --variants 3
```

Local Docker note: examples below use `./montage-ai.sh` for brevity. When running locally,
prefix with `docker compose run --rm montage-ai` (or exec into the container).

---

## Commands

### Core Commands

| Command | Purpose | Speed |
|---------|---------|-------|
| `run [STYLE]` | Create montage (default: dynamic) | Medium |
| `preview [STYLE]` | Quick preview (360p, ultrafast) | Fast ⚡ |
| `finalize [STYLE]` | High-quality render (1080p, stabilized) | Slow |
| `hq [STYLE]` | High quality with upscaling (1080p/4K) | Slow |
| `shorts [STYLE]` | Vertical 9:16 with smart reframing | Medium |
| `download JOB_ID` | Download job artifacts (video, timeline, logs) | - |
| `web` | Launch Web UI (prefer `docker compose up`) | - |
| `list` | Show available styles | - |

### Utility Commands

| Command | Purpose |
| :--- | :--- |
| `text-edit` | Transcript-based editing (remove filler words) |
| `build` | Build Docker image locally |
| `status` | Check running job status and logs |
| `jobs` | Submit/manage jobs via the Web API (`/api/jobs`) |
| `cgpu-start` | Start cgpu Gemini server (port 8090) |
| `cgpu-stop` | Stop cgpu server |
| `cgpu-status` | Check cgpu installation & status |
| `cgpu-test` | Test cgpu connection |
| `export-to-nle` | Export results to OTIO/Premiere/EDL |
| `help` | Show full help message |

Note: for local runs, `docker compose up` uses `WEB_PORT` (default 8080).

---

## API Jobs (CLI → Web API)

Use the same `/api/jobs` endpoints as the Web UI, but from the CLI.

```bash
# Submit a job
MONTAGE_API_BASE="http://<MONTAGE_API_HOST>" docker compose run --rm montage-ai ./montage-ai.sh jobs submit --style dynamic --prompt "fast teaser"

# Submit with extra options
docker compose run --rm montage-ai ./montage-ai.sh jobs submit --style hitchcock --option stabilize=true --option quality_profile=preview

# Fetch status
docker compose run --rm montage-ai ./montage-ai.sh jobs status <JOB_ID>

# Cancel a job
docker compose run --rm montage-ai ./montage-ai.sh jobs cancel <JOB_ID>

# Submit and auto-download artifacts after completion
docker compose run --rm montage-ai ./montage-ai.sh jobs submit --style dynamic --prompt "fast teaser" --download --download-dir ~/Downloads

# Download and zip artifacts (video + timeline + logs)
docker compose run --rm montage-ai ./montage-ai.sh jobs submit --style documentary --download --download-zip
```

Notes:
- `MONTAGE_API_BASE` is the server base URL (no `/api` suffix required).
- Use `--option options.<key>=...` to set nested `options` fields in the payload.
- `--download` waits for job completion and saves artifacts to the local machine.

---

## Styles

Apply different editing patterns to your footage:

```bash
./montage-ai.sh run [STYLE]
```

| Style | Character | Best For |
|-------|-----------|----------|
| `dynamic` | Position-aware pacing (intro→build→climax→outro) | General purpose, natural flow |
| `hitchcock` | Slow build-up, explosive climax | Suspense, trailers, dramatic reveals |
| `mtv` | Fast 1-2 beat cuts, high energy | Music videos, shorts, high-energy content |
| `action` | Michael Bay rapid cuts with dynamic FX | Action sequences, hype reels |
| `documentary` | Natural pacing, observational | Interviews, nature docs, slow-burn |
| `minimalist` | Long contemplative takes, silence | Art films, meditation, minimalism |
| `wes_anderson` | Symmetrical, whimsical, centered | Comedy, indie content, quirky vibes |
| `vlog` | Personal, face-centric storytelling | Vlogging, personal narratives |
| `sport` | High-energy action sequences | Sports highlights, extreme sports |

---

## Video Enhancement Options

### Stabilization & Sharpness

```bash
./montage-ai.sh run dynamic --stabilize
```

- `--stabilize` → Enable video stabilization (smooths camera shake)
- `--sharpen` → Sharpening via unsharp mask (enhances detail)
- `--denoise` → AI denoising (for low-light footage)

### Upscaling & Resolution

```bash
./montage-ai.sh hq dynamic --upscale
```

- `--upscale` → AI 4K upscaling (Real-ESRGAN) for 1080p→4K conversion

### Color Grading

```bash
./montage-ai.sh run dynamic --color-grade cinematic
```

Available presets (via `COLOR_GRADING` env var):
- `teal_orange` (default) – Industry-standard contrast
- `cinematic` – Film-like, desaturated shadows
- `warm` – Golden hour vibes
- `cool` – Cold/moody blues
- `vintage` – 70s/80s film stock

### Film Grain

```bash
./montage-ai.sh run dynamic --film-grain 35mm
```

Grain types:
- `fine` – Subtle, 16mm feel
- `medium` – Balanced
- `35mm` – Kodak-esque
- `16mm` – Vintage
- `8mm` – Super8 home video

---

## Audio Polish

Professional audio post-production in one command.

### Voice Isolation

```bash
./montage-ai.sh hq dynamic --isolate-voice
```

- Removes background noise from voice tracks
- Uses PyTorch ML (htdemucs model)
- Best for interviews, podcasts, voiceovers

### Auto Dialogue Ducking

```bash
./montage-ai.sh run dynamic --dialogue-duck
```

- Auto-detects speech
- Automatically lowers music when dialogue plays
- Adjustable via `DIALOGUE_DUCK_LEVEL` (default: -12dB)

### Audio Normalization

```bash
./montage-ai.sh run dynamic --audio-normalize
```

- Normalize all audio to -14 LUFS (broadcast standard)
- Consistent loudness across clips
- Prevents clipping, improves headroom

### Captions (Burn-in)

```bash
./montage-ai.sh run dynamic --captions tiktok
```

Caption styles:
- `tiktok` – Bold, white, bottom-center
- `youtube` – Clean serif, mid-video
- `minimal` – Small, subtle
- `bold` – Large, high contrast
- `karaoke` – Sync'd to words (requires transcription)

**Note:** Captions automatically enabled with `--shorts` mode.

---

## Story & Output

### Story Engine (Narrative Arc Optimization)

```bash
./montage-ai.sh run dynamic --story-engine
```

AI-optimized pacing, tension arcs, and clip selection based on narrative shape.

#### Story Arcs

```bash
./montage-ai.sh run dynamic --story-engine --story-arc hero_journey
```

Presets:
- `hero_journey` – Classic 3-act structure (setup→conflict→resolution)
- `mtv_energy` – Constant high-energy peaks
- `documentary` – Natural buildup, observational
- `thriller` – Slow tension → explosive climax

### Export Options

#### Timeline Export (OTIO/EDL/XML)

```bash
./montage-ai.sh run dynamic --export
```

Exports to `/data/output/timeline.otio` for use in:
- DaVinci Resolve
- Adobe Premiere Pro
- Final Cut Pro
- Avid Media Composer

#### Recipe Card (Enhancement Summary)

```bash
./montage-ai.sh run dynamic --export-recipe
```

Generates Markdown file documenting all applied effects, settings, and decisions (useful for CI/CD logs or client handoff).

### Vertical Video (Shorts)

```bash
./montage-ai.sh shorts dynamic
```

Shorthand for:
```bash
./montage-ai.sh run dynamic --shorts --captions tiktok
```

**Features:**
- Auto-reframes 16:9 footage to 9:16 (vertical)
- Smart face detection (keeps faces centered)
- Captions styled for TikTok/Instagram
- Optimizes cuts for mobile viewing

---

## Cloud Integration (cgpu)

Offload heavy lifting to cloud GPU via Gemini API.

### Enable Cloud LLM (Creative Director)

```bash
export GEMINI_API_KEY="your_api_key"
./montage-ai.sh run dynamic --cgpu
```

Uses Gemini 2.0 Flash for intelligent style suggestions, creative prompts, and content analysis.

### Enable Cloud GPU (Upscaling)

```bash
./montage-ai.sh hq dynamic --cgpu-gpu
```

Offload upscaling to cloud GPU (faster for 4K on weak machines).

### Hybrid Workflow

```bash
./montage-ai.sh run dynamic --cgpu --cgpu-gpu
```

- Local beat detection + scene analysis
- Cloud LLM for Creative Director
- Cloud GPU for upscaling
- Local final render

### Force Cloud-Only

```bash
./montage-ai.sh run dynamic --cloud-only
```

Push ALL heavy processing to cloud (useful for CI/CD or weak machines).

---

## Advanced Options

### Variants (Parallel Generation)

```bash
./montage-ai.sh run dynamic --variants 3
```

Generate 3 different montages in parallel:
- Different cut patterns
- Different effect intensities
- All output to `/data/output/`

Use for A/B testing, iterative refinement, or client presentations.

### Disable Enhancements

```bash
./montage-ai.sh run dynamic --no-enhance
```

Skip color enhancement for raw, ungraded output.

### Environment Variables

Set defaults via `.env`:

```bash
# .env file
CREATIVE_PROMPT=hitchcock
STABILIZE=true
CAPTIONS=true
CAPTIONS_STYLE=tiktok
DENOISE=true
AUDIO_NORMALIZE=true
COLOR_GRADING=cinematic
COLOR_INTENSITY=0.8
UPSCALE=true
SHORTS_MODE=true
EXPORT_RECIPE=true
CGPU_ENABLED=true
CGPU_PORT=8090
QUALITY_PROFILE=high
```

Then:
```bash
./montage-ai.sh run  # Uses defaults from .env
```

---

## Quality Profiles

### Preview (⚡ Fast)
```bash
./montage-ai.sh preview
```
- Resolution: 640×360
- Codec: H.264, CRF 28
- Preset: ultrafast
- Time: ~30-60s per minute of footage
- Effects: Disabled (except basic color)

### Standard (Medium)
```bash
./montage-ai.sh run
```
- Resolution: 1280×720
- Codec: H.264, CRF 23
- Preset: medium
- Effects: All enabled
- Time: ~2-5 min per minute of footage

### High Quality (Slow)
```bash
./montage-ai.sh finalize
./montage-ai.sh hq
```
- Resolution: 1920×1080 or 3840×2160 (4K)
- Codec: H.265/HEVC, CRF 18
- Preset: slow
- Effects: All enabled + stabilization + upscaling
- Time: ~5-15 min per minute of footage

---

## Real-World Examples

### TikTok/Reels Shorts

```bash
./montage-ai.sh shorts mtv --isolate-voice --captions tiktok --denoise
```

- 9:16 vertical aspect
- MTV style (fast cuts)
- Clean voice (noise removal)
- Auto-burned captions

### Documentary Interview

```bash
./montage-ai.sh run documentary --isolate-voice --audio-normalize --dialogue-duck
```

- Natural pacing
- Crisp voice
- Balanced levels
- Music ducks behind dialogue

### High-Res Trailer

```bash
./montage-ai.sh hq hitchcock --upscale --stabilize --color-grade cinematic --film-grain 35mm
```

- 4K upscaled output
- Smooth motion
- Cinematic look
- Kodak film texture

### Fast Social Media Loop

```bash
./montage-ai.sh preview dynamic --captions tiktok
```

- Ultra-fast render
- Mobile-optimized
- Captions included
- Ready to post immediately

### Archive/Professional Handoff

```bash
./montage-ai.sh finalize dynamic --export --export-recipe --upscale
```

- High-quality render
- OTIO timeline for further editing
- Recipe card documenting decisions
- Professional archive-ready format

---

## Troubleshooting

### "ffmpeg not found"
```bash
apt install ffmpeg  # Linux
brew install ffmpeg  # macOS
```

### "cgpu serve failed"
```bash
# Check GEMINI_API_KEY
echo $GEMINI_API_KEY

# Ensure port 8090 is not in use
netstat -an | grep 8090

# Start manually with verbose logging
export GEMINI_API_KEY="..." && cgpu serve --host 0.0.0.0 --port 8090
```

### "Insufficient memory" in HQ mode
```bash
# Use preview mode for iteration
./montage-ai.sh preview

# Or offload upscaling to cloud
./montage-ai.sh run dynamic --cgpu-gpu
```

### Output file is huge
```bash
# Use tighter CRF (higher number = smaller file, lower quality)
export CRF=28
./montage-ai.sh run

# Or use H.265 (smaller than H.264)
export CODEC=libx265
./montage-ai.sh run
```

---

## Architecture

```
User Input (CLI flags)
    ↓
montage-ai.sh (argument parsing)
    ↓
Environment Variables → config.py (Settings)
    ↓
MontageBuilder (core/montage_builder.py)
    ├─ analyze_assets() → Beat/scene detection
    ├─ plan_edits() → Clip selection & pacing
    ├─ enhance_clips() → Effects (stabilize, upscale, color)
    ├─ render_output() → SegmentWriter (disk-based composition)
    └─ cleanup() → Export timeline & recipes
    ↓
FFmpeg (video composition)
    ↓
/data/output/montage.mp4 + metadata
```

---

## Performance Notes

- **Variants:** Scales linearly (2× footage = 2× time)
- **Stabilization:** ~3-4× slower than basic render
- **Upscaling:** ~5-10× slower (or instant with `--cgpu-gpu`)
- **Voice Isolation:** ~2-3× slower (or cloud GPU)
- **Shorts Mode:** Similar speed to standard, but auto-reframes first

---

## Next Steps

- **Tutorials:** See `docs/getting-started.md`
- **Web UI:** Launch with `docker compose up`
- **Kubernetes:** See `deploy/k3s/` for cluster setup
- **Contributing:** See `CONTRIBUTING.md`

---

**Questions?** Open an issue on [GitHub](https://github.com/mfahsold/montage-ai).
