# Features

Everything Montage AI can do for you.

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

## Shorts Workflow (Vertical Video)

- **Smart Reframing**: Automatically crops horizontal footage to 9:16 vertical aspect ratio using face detection and segmented tracking.
- **Segmented Tracking**: Stabilizes camera movement by keeping the crop window static until the subject moves significantly, preventing jitter.
- **Auto-Captions**: Generates and burns in subtitles (requires `whisper`).
- **Web UI Integration**: Toggle "Shorts Mode" in the Web UI for easy creation.

## Troubleshooting

Having issues? Check [troubleshooting.md](troubleshooting.md) for common fixes.
