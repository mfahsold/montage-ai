# Montage AI 🎬

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm--NC-purple.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> Turn raw footage into beat-synchronized montages — automatically.

![Montage AI Web UI](docs/images/web-ui-hero.png)
<!-- TODO: Add a hero screenshot of the Web UI or a montage example here -->

Drop your clips and a music track, pick a style, and let the AI do the editing. Cuts land on the beat, pacing matches the energy, and you get a polished video in minutes.

---

## Get Started

### Option A: Web UI (easiest)

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
make web
```

Open **http://localhost:5001** → upload videos + music → pick a style → hit Create → done.

![Web UI Dashboard](docs/images/web-ui-dashboard.png)
<!-- TODO: Add screenshot of the Web UI dashboard with upload and style selection -->

### Option B: Command Line

```bash
# Add your media
cp your_videos/*.mp4 data/input/
cp your_music.mp3 data/music/

# Create a montage
./montage run hitchcock
```


Your video lands in `data/output/montage.mp4`.

---

## Styles

| Style          | Vibe                                 |
| -------------- | ------------------------------------ |
| `dynamic`      | Adapts to music energy (default)     |
| `hitchcock`    | Slow build, dramatic payoff          |
| `mtv`          | Fast cuts, music video feel          |
| `action`       | Intense, Michael Bay energy          |
| `documentary`  | Natural, observational               |
| `minimalist`   | Calm, long takes                     |
| `wes_anderson` | Symmetry, warm colors, quirky        |
| `viral`        | Ultra-fast TikTok/Reels, max energy  |

Or just describe what you want:

```bash
CREATIVE_PROMPT="make it feel like a 90s skateboard video" ./montage-ai.sh run
```

---

## What it Does

- **Beat sync** — Cuts align to the music rhythm
- **Smart clip selection** — AI picks the best moments
- **Story arc** — Intro → build → climax → outro pacing
- **Enhancement** — Color grading, stabilization, AI upscaling
- **Timeline export** — OTIO/EDL for DaVinci Resolve, Premiere

---

## Commercial Use & Cloud

Montage AI is **Source Available** and free for personal, non-commercial use.

**Pro Features (Coming Soon):**
*   **Montage Cloud**: Offload heavy upscaling and rendering to our H100 clusters.
*   **Commercial License**: For studios, agencies, and monetized creators.
*   **Hosted Web UI**: No Docker required.

[Contact us](mailto:sales@montage.ai) for early access.

---

## Documentation

Everything else lives in [`docs/`](docs/):

| Doc                                        | What's inside               |
| ------------------------------------------ | --------------------------- |
| [Getting Started](docs/getting-started.md) | All the ways to run it      |
| [Configuration](docs/configuration.md)     | Every setting explained     |
| [Features](docs/features.md)               | Deep dive on capabilities   |
| [Architecture](docs/architecture.md)       | How it works under the hood |
| [Contributing](CONTRIBUTING.md)            | Want to help? Start here    |

---

## Quick Troubleshooting

**Out of memory?**
```bash
MEMORY_LIMIT_GB=12 PARALLEL_ENHANCE=false ./montage-ai.sh run
```

**Want better quality?**
```bash
./montage-ai.sh hq hitchcock  # Adds stabilization + upscaling
```

**Using cloud GPU?**
```bash
CGPU_GPU_ENABLED=true ./montage-ai.sh run --upscale
```

More help → [docs/troubleshooting.md](docs/troubleshooting.md)

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal use, not for commercial products.
