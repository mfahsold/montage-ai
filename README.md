# Montage AI 🎬

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm--NC-purple.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![OSS Compliant](https://img.shields.io/badge/OSS-Compliant-green.svg)](THIRD_PARTY_LICENSES.md)

> **AI rough cut + social-ready output + pro handoff** — all in one pipeline.

![Montage AI Web UI](docs/images/web-ui-hero.png)

**We polish pixels, we don't generate them.** Drop your clips, pick a quality profile, and get a beat-synchronized edit ready for social media or NLE handoff. Local-first, privacy-first, no footage leaves your machine unless you opt into cloud acceleration.

### ✨ What's New in 2026

| Feature | Description |
|---------|-------------|
| 🎙️ **[Transcript Editor](/docs/features.md#transcript-editor)** | Descript-style text editing — delete words to cut video |
| 📱 **[Shorts Studio](/docs/features.md#shorts-studio)** | Vertical reframing, safe zones, caption styles for TikTok/Reels/Shorts |
| 🎚️ **[Quality Profiles](/docs/features.md#quality-profiles)** | Preview → Standard → High → Master — one click for all enhancement settings |
| ☁️ **[Cloud Acceleration](/docs/features.md#cloud-acceleration)** | Single toggle for GPU offloading with graceful local fallback |
| 📤 **[Pro Handoff](/docs/features.md#timeline-export)** | OTIO/EDL export for DaVinci Resolve, Premiere, Final Cut |

---

## 🎯 Who It's For

| Audience | Use Case |
|----------|----------|
| **Creator/Marketing Teams** | Fast turnaround social clips, consistent brand style |
| **Professional Editors** | AI rough cut → OTIO export → finish in your NLE |
| **Indie Filmmakers** | Festival trailers, behind-the-scenes, sizzle reels |

---

## Get Started

### Option A: Web UI (Recommended)

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
./montage-ai.sh web
```

Open **http://localhost:8080** — upload videos + music → select Quality Profile → hit Create → done.

**Quality Profiles:**
- 🚀 **Preview** — 360p, fast draft, no enhancements
- 📺 **Standard** — 1080p, light color grading (default)
- ✨ **High** — 1080p + stabilization + enhancement
- 🎬 **Master** — 4K + all enhancements + AI upscaling

**New UIs:**
- `/v2` — Outcome-based interface with Quality Profiles
- `/shorts` — Shorts Studio for vertical video
- `/transcript` — Text-based video editing

![Web UI Dashboard](docs/images/web-ui-dashboard.png)

### Option B: Command Line

```bash
# Add your media
cp your_videos/*.mp4 data/input/
cp your_music.mp3 data/music/

# Create a montage with a quality profile
./montage-ai.sh run hitchcock --quality high

# Or use environment variables
QUALITY_PROFILE=master ./montage-ai.sh run
```

The new CLI is powered by Python `click` and `rich` for a modern, interactive experience.

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

## 🚀 Key Features

### Transcript Editor
Edit video by editing text. Delete a word, the corresponding video segment is removed. Like Descript, but local-first.

### Shorts Studio  
Create vertical content with:
- 📱 Phone-frame preview with safe zones
- 🎯 Smart speaker tracking and auto-reframe
- 💬 Caption styles (TikTok, Minimal, Bold, Karaoke)
- ✂️ AI highlight detection for best moments

### Quality Profiles
One selection replaces 5+ separate toggles:
- **Preview**: Fast iteration, low resource usage
- **Standard**: Production-ready 1080p
- **High**: Stabilization + enhancement
- **Master**: Maximum quality, 4K output

### Cloud Acceleration
Single toggle enables GPU offloading for:
- AI upscaling (Real-ESRGAN)
- Transcription (Whisper)
- LLM creative direction (Gemini)

Graceful fallback to local processing if cloud unavailable.

### Timeline Export (Pro Handoff)
Export to your NLE of choice:
- **OTIO** — DaVinci Resolve, Premiere Pro
- **EDL** — Legacy NLE support
- **CSV** — Spreadsheet review
- **Proxies** — Optional low-res for offline editing

---

## 🔒 Privacy & Licensing

### Local-First Philosophy
- All processing happens on your machine by default
- No footage uploaded without explicit opt-in
- No training on user content — ever
- Decision logs available via `EXPORT_DECISIONS=true`

### Commercial Use & Cloud

Montage AI is **Source Available** under [PolyForm Noncommercial 1.0.0](LICENSE) — free for personal, non-commercial use.

**Pro Features (Coming Soon):**
- **Montage Cloud**: Offload heavy rendering to H100 clusters
- **Commercial License**: For studios, agencies, monetized creators
- **Team Features**: Shared styles, brand presets, collaboration

[Contact us](mailto:sales@montage.ai) for early access.

### Open Source Dependencies

We gratefully build on these open source projects:
- **FFmpeg** (LGPL) — Video processing backbone
- **OpenCV** (Apache 2.0) — Computer vision
- **Whisper** (MIT) — Speech recognition
- **Librosa** (ISC) — Audio analysis
- **OpenTimelineIO** (Apache 2.0) — Timeline interchange

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for complete attribution and license texts.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation & first montage |
| [Features](docs/features.md) | Deep dive on all capabilities |
| [Configuration](docs/configuration.md) | Every setting explained |
| [Architecture](docs/architecture.md) | How it works under the hood |
| [Strategy](docs/STRATEGY.md) | Product vision & roadmap |
| [Roadmap](docs/roadmap/ROADMAP_2026.md) | 12-month development plan |
| [Backlog](docs/BACKLOG.md) | Epics & user stories |
| [Contributing](CONTRIBUTING.md) | How to contribute |
| [Third-Party Licenses](THIRD_PARTY_LICENSES.md) | OSS dependencies & licenses |

---

## 🔧 Quick Troubleshooting

**Out of memory?**
```bash
MEMORY_LIMIT_GB=12 PARALLEL_ENHANCE=false ./montage-ai.sh run
```

**Want better quality?**
```bash
./montage-ai.sh run hitchcock --quality master
```

**Using cloud GPU?**
```bash
CLOUD_ACCELERATION=true ./montage-ai.sh run --upscale
```

**Create vertical shorts?**
```bash
./montage-ai.sh run viral --aspect 9:16 --captions
```

More help → [docs/troubleshooting.md](docs/troubleshooting.md)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick links:**
- [Report a bug](https://github.com/mfahsold/montage-ai/issues/new?template=bug_report.md)
- [Request a feature](https://github.com/mfahsold/montage-ai/issues/new?template=feature_request.md)
- [View the backlog](docs/BACKLOG.md)

---

## 📜 License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal use, commercial license available.

---

<p align="center">
  <strong>Montage AI</strong> — We polish pixels, we don't generate them.<br>
  <a href="https://montage.ai">Website</a> · 
  <a href="https://github.com/mfahsold/montage-ai">GitHub</a> · 
  <a href="mailto:hello@montage.ai">Contact</a>
</p>
