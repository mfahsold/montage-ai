# Features

Everything Montage AI can do for you.

---

## Core Editing

- Beat-synced cuts using `librosa` beat detection
- Style-aware pacing, transitions, and color looks
- Story arc shaping (intro → build → climax → outro)
- LLM-powered "Creative Director" (Ollama local or Gemini via cgpu)

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

![Web UI Features](images/web-ui-features.png)
<!-- TODO: Add screenshot highlighting Web UI features like style selection and toggles -->

Flow: upload videos + music → pick style or prompt → toggle enhance/stabilize/upscale/cloud GPU → Create Montage → download MP4 (and timeline if enabled).

Useful endpoints (for automation):

- `GET /api/status` – health
- `GET /api/files` – list uploads
- `POST /api/upload` (multipart, fields: `file`, `type=video|music`)
- `POST /api/jobs` – create job with JSON body (`style`, `prompt`, `stabilize`, `upscale`, `cgpu`, `export_timeline`, ...)
- `GET /api/jobs/{id}` – job status
- `GET /api/download/{filename}` – download outputs

## Timeline Export (OTIO/EDL)

Enable during run:

```bash
./montage-ai.sh run hitchcock --export-timeline --generate-proxies
```

Outputs in `data/output/`:
- `*.otio` (preferred), `*.edl`, `*.csv`, metadata JSON, optional proxies folder.

![Timeline Export in DaVinci Resolve](images/timeline-export-davinci.png)
<!-- TODO: Add screenshot of the exported timeline imported into DaVinci Resolve -->

Import tips:
- **DaVinci Resolve:** File → Import → Timeline → select `.otio`; relink media if paths differ.
- **Premiere Pro / Avid:** use `.edl` and relink originals.

## Cloud LLM & GPU (cgpu)

- Install: `npm i -g cgpu` (plus gemini-cli; run `cgpu connect` once)
- Enable Gemini LLM: `CGPU_ENABLED=true ./montage-ai.sh run --cgpu`
- Enable Colab GPU upscaling: `CGPU_GPU_ENABLED=true ./montage-ai.sh run --upscale --cgpu-gpu`

Fallback order for upscaling: cgpu T4/A100 → local Vulkan GPU → FFmpeg Lanczos (CPU).

## Troubleshooting

Having issues? Check [troubleshooting.md](troubleshooting.md) for common fixes.
