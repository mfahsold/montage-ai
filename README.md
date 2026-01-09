# Montage AI — AI Video Editor. Polish, Don't Generate.

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm--NC-purple.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![K3s](https://img.shields.io/badge/k3s-ready-green.svg)](https://k3s.io/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](docs/ci.md)
[![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen.svg)]()

> **We do not generate pixels. We polish them.**

Free, open-source AI video editor: beat-sync montages, transcript editing, OTIO/EDL export. Local-first, privacy-first alternative to Descript. Professional post-production workflows for content creators, video editors, and AI enthusiasts.

Looking for the website? Visit [mfahsold.github.io/montage-ai](https://mfahsold.github.io/montage-ai) for the live landing page and docs search.

## Quick Start

**Three ways. Pick your workflow. [→ Full Setup Guide](docs/getting-started.md)**

### Web UI (Easiest)

```bash
./montage-ai.sh web
# Open http://localhost:8080 in your browser (or set WEB_PORT)
```

### Command Line (Fastest)

```bash
cp your_videos/*.mp4 data/input/
./montage-ai.sh run dynamic --quality high
```

### Docker (Reproducible)

```bash
docker-compose up -d
# Same web UI, containerized
```

## Installation

### Minimal (Core Only)

```bash
pip install montage-ai
```

Includes video processing, FFmpeg integration, timeline export. No AI features, web UI, or cloud support.

### With AI Enhancements

```bash
pip install montage-ai[ai]
```

Adds smart reframing, face detection, color matching, advanced audio analysis. ~420 MB.

### With Web UI

```bash
pip install montage-ai[web]
```

Full web interface + REST API + background job queue. Requires Redis.

### Everything (Development)

```bash
pip install montage-ai[all]
```

All features included. Same as `pip install -r requirements.txt`.

**→ [See Optional Dependencies Guide](docs/OPTIONAL_DEPENDENCIES.md) for all installation options.**

## Key Features

| Feature | What It Does |
|---------|------------|
| **Beat-Sync** | Cuts aligned to music rhythm via FFmpeg (astats/tempo) |
| **Transcript Editor** | Text-based video editing |
| **Shorts Studio** | Auto-reframe to 9:16 vertical |
| **Pro Export** | OTIO/EDL for DaVinci, Premiere, FCP |
| **Quality Profiles** | Preview → Standard → High → Master |

### New in v1.3 (Pro Polish)

| Feature | Description |
|---------|-------------|
| **Audio Polish** | Voice Isolation (Filter Chain) + Auto-Ducking (Sidechain) |
| **Pro Export** | OTIO + Proxy Generation (H.264/ProRes) for NLEs |
| **Performance** | Optimized Preview Pipeline (<3 min render) |

### New in v1.2 (Shorts Studio 2.0)

| Feature | Description |
|---------|-------------|
| **Smart Reframing v2** | Kalman filter tracking for smoother subject centering |
| **Highlight Detection** | Multi-modal scoring (Audio + Motion + Face) |
| **Review Cards UI** | Interactive review of detected highlight moments |
| **Viral Captions** | 3 new preset styles for vertical video captions |

### New in v1.1

| Feature | Description |
|---------|-------------|
| **AI Denoising** | hqdn3d/nlmeans noise reduction with grain preservation |
| **Color Matching** | Shot-to-shot color consistency across clips |
| **Film Grain** | Authentic 35mm/16mm/8mm grain simulation |
| **Dialogue Ducking** | Auto-detect speech, lower music, export keyframes |
| **NLE Recipe Cards** | Human-readable enhancement instructions |
| **Enhancement Tracking** | Every AI decision exportable to NLEs |

## Editing Styles

`dynamic` · `hitchcock` · `mtv` · `action` · `documentary` · `minimalist` · `wes_anderson` · `viral` · `wedding` · `travel` · `gaming`

Or natural language: `CREATIVE_PROMPT="90s skateboard" ./montage-ai.sh run`

## Processing Options

```bash
# Full enhancement pipeline
./montage-ai.sh run hitchcock \
  --stabilize \
  --upscale \
  --color-grade cinematic \
  --denoise \
  --film-grain 35mm
```

## How It Compares

| | Montage AI | Descript | Frame | AutoClip | Adobe |
|---|---|---|---|---|---|
| **Cost** | Free | $12/mo | Free | Free | $55+/mo |
| **Local** | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Beat-Sync** | ✅ (FFmpeg) | Limited | ❌ | Energy only | No |
| **Story Arc** | ✅ | ❌ | ❌ | ❌ | No |
| **Auto-Reframe** | ✅ | ❌ | ✅ | ❌ | Limited |
| **OTIO/EDL** | ✅ | No | No | No | Limited |
| **Enhancement Tracking** | ✅ | No | No | No | No |

## Architecture

```
Input Footage → Analysis → Creative Direction → Editing → Enhancement → Export
                  │              │                 │           │
                  └──────────────┴─────────────────┴───────────┴──→ NLE (OTIO/EDL)
```

- **Three-Stage Pipeline**: Ingest → Edit → Finish
- **Local-First**: All processing on your hardware
- **Cloud GPU Optional**: cgpu for upscaling/transcription
- **K3s Distributed**: Scale across ARM + AMD nodes

## Documentation

- **[Getting Started](docs/getting-started.md)** — Installation & first project
- **[Features](docs/features.md)** — Complete capabilities guide
- **[Configuration](docs/configuration.md)** — All settings & environment variables
- **[Architecture](docs/architecture.md)** — How it works (for developers)
- **[Parameter Reference](docs/PARAMETER_REFERENCE.md)** — All controllable parameters
- **[Troubleshooting](docs/troubleshooting.md)** — Common issues & fixes

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We welcome:
- Bug fixes and feature PRs
- Style template contributions
- Documentation improvements
- Translation help

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — Free for personal use.

---

<p align="center">
  <a href="https://mfahsold.github.io/montage-ai">Website</a> ·
  <a href="docs/COMPETITIVE_ANALYSIS.md">Why Us?</a> ·
  <a href="docs/features.md">Features</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>
