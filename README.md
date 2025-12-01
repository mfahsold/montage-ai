# 🎬 Montage AI - Intelligent Video Editor

**Version:** 0.3.0  
**Status:** Production-Ready

AI-powered automatic video montage creation with **natural language control**, **beat-synchronized editing**, **professional timeline export**, and **2024/2025 cutting techniques**.

> "Turn raw footage into cinematic montages with a single command."

---

## ✨ Features

### 🎯 Natural Language Control
```bash
# Just describe what you want
montage-ai "Edit this like Hitchcock - slow build-up with explosive climax"
montage-ai "Fast MTV-style cuts synced to the beat"
montage-ai "Documentary realism with natural pacing"
```

### 🎵 Beat-Synchronized Editing
- Automatic beat detection via librosa
- Energy analysis for dynamic pacing
- Fibonacci-based cut patterns
- Story arc awareness (Intro → Build → Climax → Outro)

### 🎬 Professional Techniques
- **Match Cuts**: Visual similarity detection for seamless transitions
- **Invisible Cuts**: Motion blur detection for hiding edits
- **Footage Management**: Professional "use once" principle
- **B-Roll Integration**: Smart cutaway placement

### 📦 Export Options
- **OpenTimelineIO**: Export to Premiere Pro, DaVinci Resolve, Final Cut Pro
- **EDL Format**: Industry-standard Edit Decision Lists
- **JSON Timeline**: Machine-readable timeline data

### 🖥️ Hardware Acceleration
- Vulkan GPU encoding
- V4L2 hardware encoding (Raspberry Pi, ARM)
- Multi-threaded CPU processing
- Real-ESRGAN AI upscaling

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/montage-ai.git
cd montage-ai

# Build Docker image
docker build -t montage-ai:latest .

# Or install locally (requires Python 3.10+)
pip install -e .
```

### Basic Usage

```bash
# Prepare your data folder
mkdir -p data/{input,music,output,assets}
# Put videos in data/input/
# Put music in data/music/

# Run with default settings
./montage-ai.sh dynamic

# Or with Docker
docker run --rm -v $(pwd)/data:/data montage-ai:latest
```

### CLI Shortcuts

```bash
./montage-ai.sh hitchcock      # Suspense style
./montage-ai.sh mtv            # Fast MTV cuts
./montage-ai.sh action         # Blockbuster action
./montage-ai.sh documentary    # Natural pacing
./montage-ai.sh wes            # Wes Anderson style
./montage-ai.sh minimalist     # Art film long takes
```

### Quality Presets

| Option | Speed | Quality | Effects |
|--------|-------|---------|---------|
| *(default)* | ~5 min | Good | Enhance only |
| `--fast` | ~2 min | Basic | None |
| `--hq` | ~10 min | High | Stabilize + Enhance |
| `--max` | ~30+ min | Maximum | All (AI Upscale) |

---

## 📁 Project Structure

```
montage-ai/
├── src/
│   ├── montage_ai/
│   │   ├── __init__.py
│   │   ├── editor.py           # Main editing engine
│   │   ├── creative_director.py # Natural language → parameters
│   │   ├── style_templates.py  # Pre-defined editing styles
│   │   ├── footage_manager.py  # Professional footage tracking
│   │   ├── footage_analyzer.py # Deep footage analysis
│   │   ├── timeline_exporter.py # OTIO/EDL export
│   │   └── monitoring.py       # Live progress tracking
│   └── cli.py                  # Command-line interface
├── data/                       # Default data directory
│   ├── input/                  # Source videos
│   ├── music/                  # Audio tracks
│   ├── assets/                 # Logos, overlays
│   └── output/                 # Generated videos
├── docs/
│   ├── EDITING_CONCEPTS.md     # Professional editing theory
│   ├── STYLE_GUIDE.md          # Style template reference
│   └── API.md                  # Python API documentation
├── tests/
├── Dockerfile
├── docker-compose.yml
├── montage-ai.sh               # CLI wrapper
├── pyproject.toml
└── README.md
```

---

## 🎨 Editing Styles

### Pre-defined Templates

| Style | Description | Best For |
|-------|-------------|----------|
| **Hitchcock** | Long takes building to explosive climax | Drama, suspense |
| **MTV** | Fast 1-2 beat cuts, high energy | Music videos |
| **Action** | Michael Bay rapid cuts, match cuts | Action content |
| **Documentary** | Natural pacing, observational | Vlogs, docs |
| **Wes Anderson** | Symmetrical, whimsical, centered | Artistic content |
| **Minimalist** | Very long takes, meditative | Art films |

### Custom Prompts

```bash
# The AI understands nuanced requests:
"Slow and meditative with occasional bursts of energy"
"80s music video aesthetic with lens flares"
"Nature documentary with David Attenborough pacing"
"TikTok-style vertical cuts, super fast"
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CREATIVE_PROMPT` | - | Natural language editing instructions |
| `CUT_STYLE` | `dynamic` | Fallback: fast/slow/hyper/dynamic |
| `STABILIZE` | `false` | Video stabilization |
| `UPSCALE` | `false` | AI upscaling (slow) |
| `ENHANCE` | `true` | Color/sharpness enhancement |
| `STRICT_FOOTAGE_MODE` | `true` | Use each clip only once |
| `DEEP_ANALYSIS` | `false` | AI scene analysis |
| `EXPORT_TIMELINE` | `false` | Export OTIO timeline |
| `NUM_VARIANTS` | `1` | Generate multiple versions |
| `FFMPEG_PRESET` | `medium` | Encoding speed/quality |

### Output Format

- **Resolution**: 1080x1920 (9:16 vertical, social media optimized)
- **Codec**: H.264/H.265
- **Audio**: AAC 192kbps

---

## 📊 Monitoring

Real-time progress tracking with phase breakdown:

```
🎬 Montage AI v0.3.0
═══════════════════════════════════════════
📊 Phase: assembling (2/5)
   Progress: ████████░░░░░░░░ 45%
   Cut #23: video_001.mp4 @ 34.5s
   Energy: 0.72 | Tempo: 128 BPM
   
📈 Stats:
   Clips Used: 23/45 (51%)
   Variety: 87%
   Duration: 1:45 / 3:20
═══════════════════════════════════════════
```

---

## 🔌 API Usage

```python
from montage_ai import MontageEditor, StyleTemplate

# Create editor
editor = MontageEditor(
    input_dir="./videos",
    music_path="./music/track.mp3",
    style=StyleTemplate.HITCHCOCK
)

# Or with natural language
editor = MontageEditor.from_prompt(
    prompt="Edit like a 90s music video",
    input_dir="./videos",
    music_path="./track.mp3"
)

# Generate montage
result = editor.create_montage(
    output_path="./output.mp4",
    stabilize=True,
    enhance=True
)

# Export timeline for NLE
editor.export_timeline("./timeline.otio")
```

---

## 🎯 Roadmap

- [x] Beat-synchronized editing
- [x] Natural language control
- [x] Professional footage management
- [x] OpenTimelineIO export
- [x] GPU acceleration
- [ ] Web UI interface
- [ ] Cloud processing API
- [ ] Multi-track audio support
- [ ] Automatic music selection
- [ ] AI-generated transitions

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [MoviePy](https://github.com/Zulko/moviepy) - Video editing engine
- [librosa](https://github.com/librosa/librosa) - Audio analysis
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) - Scene detection
- [OpenTimelineIO](https://github.com/AcademySoftwareFoundation/OpenTimelineIO) - Timeline interchange
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI upscaling

---

**Made with ❤️ for content creators**
