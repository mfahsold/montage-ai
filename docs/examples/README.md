# Examples & Demos

This directory contains example outputs, demo videos, and usage examples for Montage AI.

---

## 📸 Screenshots

### Web UI - Main Interface

![Web UI Hero](../images/web-ui-hero.png)

The main interface showing upload area, style selection, and quality profiles.

### Quality Profiles

![Quality Profiles](../images/quality-profiles.png)

One-click quality selection replacing multiple toggles:
- 🚀 **Preview** — Fast iteration
- 📺 **Standard** — Production-ready
- ✨ **High** — With stabilization
- 🎬 **Master** — Maximum quality

### Transcript Editor

![Transcript Editor](../images/transcript-editor.png)

Descript-style text editing (UI integration in progress; preview screenshot):
- Word-level timestamps
- Click to seek
- Delete text to cut video
- Filler word highlighting

### Shorts Studio

![Shorts Studio](../images/shorts-studio.png)

Vertical video creation (UI integration in progress; preview screenshot):
- Phone-frame preview
- Safe zone overlays
- Caption style selection
- Reframe mode options

---

## 🎬 Demo Videos

> **Note:** Demo videos are available on our [YouTube channel](https://youtube.com/@montageai) and [website](https://montage.ai/demos).

### Quick Start Demo (2 min)
Basic workflow: upload → style → create → download

### Style Comparison (5 min)
Side-by-side comparison of all built-in styles with the same footage

### Pro Handoff Workflow (3 min)
Creating a rough cut and exporting to DaVinci Resolve via OTIO

### Shorts Studio Tutorial (4 min)
Creating vertical content with captions and smart reframing

---

## 💻 Code Examples

### Basic CLI Usage

```bash
# Simple montage with default style
./montage-ai.sh run

# Specific style
./montage-ai.sh run hitchcock

# With quality profile
./montage-ai.sh run mtv --quality high

# Vertical shorts
./montage-ai.sh run viral --aspect 9:16 --captions
```

### Custom Creative Prompt

```bash
# Natural language direction
CREATIVE_PROMPT="energetic summer vibes, lots of movement, end on a calm shot" \
  ./montage-ai.sh run

# Specific mood
CREATIVE_PROMPT="noir thriller, slow build, dramatic reveal at the end" \
  ./montage-ai.sh run hitchcock
```

### Timeline Export

```bash
# Full export with proxies
./montage-ai.sh run documentary \
  --export-timeline \
  --generate-proxies \
  --quality high
```

### API Usage

```python
import requests

API_BASE = "http://<MONTAGE_API_HOST>"

# Create a job
response = requests.post(f'{API_BASE}/api/jobs', json={
    'style': 'dynamic',
    'quality_profile': 'high',
    'export_timeline': True,
    'cloud_acceleration': False
})
job_id = response.json()['id']

# Poll for completion
status = requests.get(f'{API_BASE}/api/jobs/{job_id}').json()
while status['status'] != 'completed':
    time.sleep(5)
    status = requests.get(f'{API_BASE}/api/jobs/{job_id}').json()

# Download result
output_url = status['output_url']
```

### Transcript API

```python
# Start transcription
response = requests.post(f'{API_BASE}/api/transcript', json={
    'video_id': 'video123',
    'language': 'en'
})
transcript_id = response.json()['id']

# Get transcript with word timestamps
transcript = requests.get(f'{API_BASE}/api/transcript/{transcript_id}').json()

# Apply edits (delete words 5-10)
requests.post(f'{API_BASE}/api/transcript/{transcript_id}/edit', json={
    'operations': [
        {'type': 'delete', 'start_word': 5, 'end_word': 10}
    ]
})
```

---

## 📁 Sample Assets

For testing, you can use:

### Free Stock Footage
- [Pexels Videos](https://www.pexels.com/videos/)
- [Pixabay Videos](https://pixabay.com/videos/)
- [Coverr](https://coverr.co/)

### Royalty-Free Music
- [Free Music Archive](https://freemusicarchive.org/)
- [YouTube Audio Library](https://studio.youtube.com/channel/audio)
- [Uppbeat](https://uppbeat.io/)

### Test Data Structure

```
data/
├── input/           # Your source videos
│   ├── clip1.mp4
│   ├── clip2.mov
│   └── clip3.mp4
├── music/           # Your audio tracks
│   └── track.mp3
└── output/          # Generated outputs
    ├── montage.mp4
    ├── montage.otio
    └── montage.edl
```

---

## 🎨 Style Gallery

### Dynamic (Default)
Best for: General purpose, adapts to music
![Dynamic Style](../images/style-dynamic.jpg)

### Hitchcock
Best for: Thrillers, reveals, dramatic content
![Hitchcock Style](../images/style-hitchcock.jpg)

### MTV
Best for: Music videos, dance, high energy
![MTV Style](../images/style-mtv.jpg)

### Wes Anderson
Best for: Aesthetic pieces, quirky content
![Wes Anderson Style](../images/style-wes-anderson.jpg)

---

## 📖 More Resources

- [Full Documentation](../README.md)
- [Feature Guide](../features.md)
- [Configuration Options](../configuration.md)
- [Troubleshooting](../troubleshooting.md)
- [Architecture Overview](../architecture.md)
