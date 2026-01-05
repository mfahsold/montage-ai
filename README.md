# Montage AI — AI Video Editor. Polish, Don't Generate.

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm--NC-purple.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> **We do not generate pixels. We polish them.**

Free, open-source AI video editor: beat-sync montages, transcript editing, OTIO/EDL export. Local-first, privacy-first alternative to Descript.

## Quick Start

**Three ways. Pick your workflow. [→ Full Setup Guide](docs/getting-started.md)**

### Web UI (Easiest)
```bash
./montage-ai.sh web
# Open http://localhost:5001 in your browser
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

## Key Features

| Feature | What It Does |
|---------|------------|
| **Beat-Sync** | Cuts aligned to music rhythm |
| **Transcript Editor** | Text-based video editing |
| **Shorts Studio** | Auto-reframe to 9:16 vertical |
| **Pro Export** | OTIO/EDL for DaVinci, Premiere |
| **Quality Profiles** | Preview → Standard → High |

## Editing Styles

`dynamic` · `hitchcock` · `mtv` · `action` · `documentary` · `minimalist` · `wes_anderson`

Or natural language: `CREATIVE_PROMPT="90s skateboard" ./montage-ai.sh run`

## How It Compares

| | Montage AI | Descript | Frame | Adobe |
|---|---|---|---|---|
| **Cost** | Free | $12/mo | Free | $55+/mo |
| **Local** | ✅ | ❌ | ✅ | ❌ |
| **Beat-Sync** | ✅ | Limited | ❌ | No |
| **Auto-Reframe** | ✅ | ❌ | ✅ | No |
| **Export OTIO/EDL** | ✅ | No | No | Limited |

## Documentation

- **[Getting Started](docs/getting-started.md)** — Installation & first project
- **[Features](docs/features.md)** — Complete capabilities guide
- **[Configuration](docs/configuration.md)** — All settings & environment variables
- **[Architecture](docs/architecture.md)** — How it works (for developers)
- **[Troubleshooting](docs/troubleshooting.md)** — Common issues & fixes

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — Free for personal use.

---

<p align="center">
  <a href="https://mfahsold.github.io/montage-ai">Website</a> ·
  <a href="docs/COMPETITIVE_ANALYSIS.md">Why Us?</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>
