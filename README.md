# Montage AI — Free AI Video Editor for Rough Cuts (Offline Descript Alternative)

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm--NC-purple.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> **We polish pixels, we don't generate them.**

Free, open-source AI video editor for rough cuts: beat-sync, story arcs, OTIO/EDL export. Offline Descript alternative that is local-first and privacy-first.

## Quick Start

**Two workflows. No choices. [→ Read DX Guide](docs/DX.md)**

### Local Development (5 sec feedback)
```bash
make dev       # Build once
make dev-test  # Code → test (instant)
```

### Cluster Deployment (2-15 min)
```bash
make cluster   # Build + push + deploy
```

### Or use Web UI
```bash
./montage-ai.sh web  # http://localhost:5001
```

### Or traditional CLI
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

**Start here:** [DX Guide](docs/DX.md) — Golden Path (4 commands, 90% of work)

- [Getting Started](docs/getting-started.md) — Installation & first montage
- [Features](docs/features.md) — All capabilities
- [Configuration](docs/configuration.md) — Environment variables
- [Cluster Deployment](deploy/CLUSTER_WORKFLOW.md) — Multi-arch builds
- [Competitive Analysis](docs/COMPETITIVE_ANALYSIS.md) — Market positioning

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — Free for personal use.

---

<p align="center">
  <a href="https://mfahsold.github.io/montage-ai">Website</a> ·
  <a href="docs/COMPETITIVE_ANALYSIS.md">Why Us?</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>
