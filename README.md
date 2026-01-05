# Montage AI

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm--NC-purple.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> **We polish pixels, we don't generate them.**

AI-powered rough cuts with beat-sync, story arcs, and NLE export. Local-first, privacy-first.

## Quick Start

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
./montage-ai.sh web        # Web UI at http://localhost:5001
```

Or via CLI:

```bash
cp your_videos/*.mp4 data/input/
cp your_music.mp3 data/music/
./montage-ai.sh run hitchcock --quality high
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Beat-Sync** | Cuts aligned to music rhythm (librosa) |
| **Story Arc** | 5-phase narrative structure |
| **Pro Handoff** | OTIO/EDL export to DaVinci/Premiere |
| **Shorts Studio** | Vertical video with auto-reframe |
| **Quality Profiles** | Preview → Standard → High → Master |

## Styles

`dynamic` · `hitchcock` · `mtv` · `action` · `documentary` · `minimalist` · `wes_anderson` · `viral`

Or use natural language: `CREATIVE_PROMPT="90s skateboard vibe" ./montage-ai.sh run`

## Why Montage AI?

| | Montage AI | Descript | Frame | AutoClip |
|---|---|---|---|---|
| **Cost** | Free | $12-30/mo | Free | Free |
| **Local** | ✅ | ❌ Cloud | ✅ | ✅ |
| **Beat-Sync** | ✅ librosa | ⚠️ | ❌ | ⚠️ |
| **Story Arc** | ✅ 5-phase | ❌ | ❌ | ❌ |
| **NLE Export** | ✅ OTIO | ⚠️ MP4 | ❌ | ❌ |

## Documentation

- [Getting Started](docs/getting-started.md) — Installation & first montage
- [Features](docs/features.md) — All capabilities explained
- [Configuration](docs/configuration.md) — Environment variables
- [Competitive Analysis](docs/COMPETITIVE_ANALYSIS.md) — Market positioning

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — Free for personal use.

---

<p align="center">
  <a href="https://mfahsold.github.io/montage-ai">Website</a> ·
  <a href="docs/COMPETITIVE_ANALYSIS.md">Why Us?</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>
