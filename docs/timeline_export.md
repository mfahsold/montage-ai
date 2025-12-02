# Timeline Export Guide

Export your Montage-AI edits to professional NLE software (DaVinci Resolve, Premiere Pro, Final Cut Pro, Avid).

---

## Quick Start

```bash
# Enable timeline export
./montage-ai.sh run hitchcock \
  --export-timeline \
  --generate-proxies

# Output files:
# - montage_XXX.otio     (OpenTimelineIO - recommended)
# - montage_XXX.edl      (CMX EDL - universal)
# - montage_XXX.csv      (Spreadsheet)
# - montage_XXX_metadata.json
# - montage_XXX_PROJECT/ (Complete package)
```

---

## What Gets Exported

### File Structure

```
data/output/
├── montage_20250102_123456_v1_hitchcock.mp4  # Final video
├── montage_20250102_123456_v1_hitchcock.otio  # Timeline
├── montage_20250102_123456_v1_hitchcock.edl
├── montage_20250102_123456_v1_hitchcock.csv
├── montage_20250102_123456_v1_hitchcock_metadata.json
└── montage_20250102_123456_v1_hitchcock_PROJECT/
    ├── README.txt
    ├── montage.otio
    ├── montage.edl
    ├── metadata.json
    └── proxies/          # If --generate-proxies was used
        ├── proxy_VID001.mp4
        └── proxy_VID002.mp4
```

### Exported Data

Each timeline export includes:

- **Cut points**: Exact timecodes (frame-accurate)
- **Source clips**: Paths to original video files
- **Clip metadata**:
  - Energy level (0.0-1.0)
  - Action level (low/medium/high)
  - Shot type (close-up/medium/wide)
  - Motion score
  - Story phase (intro/build/climax/outro)
- **Audio track**: Music file reference
- **Timeline duration**: Total montage length
- **Frame rate**: 30 fps (standard)
- **Resolution**: 1080x1920 (9:16 vertical)

---

## Import into NLE Software

### DaVinci Resolve (Recommended)

**OpenTimelineIO import:**

1. Open DaVinci Resolve
2. **File** → **Import** → **Timeline** → **Import AAF, EDL, XML...**
3. Select the `.otio` file
4. Choose import options:
   - ✅ **Automatically import source clips into Media Pool**
   - ✅ **Use original media**
   - ⚠️ **Relink media** if clips are in different location
5. Click **OK**

**Result:** Your montage appears as a timeline with all cuts preserved.

**EDL fallback (if OTIO not supported):**

1. **File** → **Import** → **Timeline** → **EDL...**
2. Select the `.edl` file
3. Manually relink media if needed

---

### Adobe Premiere Pro

**EDL import:**

1. Open Premiere Pro
2. **File** → **Import**
3. Select the `.edl` file
4. Premiere creates a new sequence
5. **Right-click timeline** → **Relink Media** (if paths changed)
6. Select original video files

**Note:** Premiere Pro doesn't fully support OpenTimelineIO yet. Use EDL.

---

### Final Cut Pro

**OpenTimelineIO import:**

1. Open Final Cut Pro
2. **File** → **Import** → **XML...**
3. Select the `.otio` file (FCP treats it as XML)
4. Choose import options
5. Relink media if needed

**Alternative:** Convert OTIO to FCPXML using OpenTimelineIO CLI:

```bash
otioconvert -i montage.otio -o montage.fcpxml
```

---

### Avid Media Composer

**EDL import:**

1. **File** → **Import**
2. Select the `.edl` file
3. Choose sequence settings
4. Manually relink media clips

---

## Proxy Workflow

For smooth editing, generate proxies:

```bash
./montage-ai.sh run --export-timeline --generate-proxies
```

### Proxy Settings

- **Resolution**: 960x540 (half of 1080x1920)
- **Codec**: H.264 (libx264)
- **Bitrate**: 5 Mbps
- **File size**: ~50% smaller than originals

### Import Workflow

1. Import timeline with proxies
2. Edit smoothly on laptop/lower-end hardware
3. When ready to export:
   - Relink to original high-res files
   - Export final video at full quality

**DaVinci Resolve Proxy Example:**

1. Import `.otio` timeline (uses proxies automatically)
2. **Media Pool** → Right-click clip → **Relink Clips**
3. Point to original high-res files
4. Export at full quality

---

## Metadata JSON Format

The `_metadata.json` file contains complete edit information:

```json
{
  "project_name": "montage_20250102_v1_hitchcock",
  "duration_sec": 150.5,
  "fps": 30,
  "resolution": [1080, 1920],
  "audio_file": "/data/music/soundtrack.mp3",
  "clips": [
    {
      "index": 1,
      "source_file": "/data/input/VID001.mp4",
      "proxy_file": "/data/output/proxies/proxy_VID001.mp4",
      "source_in": 5.2,
      "source_out": 7.8,
      "timeline_in": 0.0,
      "timeline_out": 2.6,
      "duration": 2.6,
      "metadata": {
        "energy": 0.75,
        "action": "high",
        "shot": "wide",
        "motion_score": 0.82,
        "story_phase": "intro"
      }
    },
    ...
  ]
}
```

### Use Cases

- **Analytics**: Track which clips were used
- **Archive**: Document edit decisions
- **Custom Tools**: Build your own processing pipeline
- **ML Training**: Train models on editing patterns

---

## CSV Export Format

Spreadsheet with all edit details:

| Clip # | Source File | Source In | Source Out | Timeline In | Duration | Energy | Action | Shot Type |
|--------|-------------|-----------|------------|-------------|----------|--------|--------|-----------|
| 1 | VID001.mp4 | 5.20 | 7.80 | 0.00 | 2.60 | 0.75 | high | wide |
| 2 | VID002.mp4 | 10.50 | 13.10 | 2.60 | 2.60 | 0.68 | medium | close-up |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Open in:**
- Excel
- Google Sheets
- LibreOffice Calc

**Use for:**
- Manual review
- Edit notes
- Shot logging
- Production reports

---

## Environment Variables

Control timeline export via environment variables:

```bash
# Enable/disable export
EXPORT_TIMELINE=true         # Default: false
GENERATE_PROXIES=true        # Default: false

# Run montage with export
docker-compose run \
  -e EXPORT_TIMELINE=true \
  -e GENERATE_PROXIES=true \
  montage-ai
```

---

## Troubleshooting

### "OpenTimelineIO not available"

**Solution:** Install the dependency:

```bash
pip install OpenTimelineIO>=0.16.0
```

Or rebuild Docker image:

```bash
./montage-ai.sh build
```

### "Cannot relink media in DaVinci Resolve"

**Problem:** Original video files moved or paths changed.

**Solution:**

1. Open `.otio` file in text editor
2. Update `target_url` paths to new locations
3. Re-import in Resolve

Or use absolute paths from the start:

```bash
# Use absolute paths for input
cp /absolute/path/to/videos/* data/input/
```

### "EDL import shows wrong timecodes"

**Problem:** Frame rate mismatch (e.g., 24fps vs 30fps).

**Solution:**

1. Check original video frame rate:
   ```bash
   ffprobe -v error -select_streams v:0 \
     -show_entries stream=r_frame_rate \
     -of default=noprint_wrappers=1:nokey=1 video.mp4
   ```

2. If not 30 fps, convert before import:
   ```bash
   ffmpeg -i input.mp4 -r 30 output_30fps.mp4
   ```

### "Proxy generation failed"

**Problem:** FFmpeg not found or insufficient disk space.

**Solution:**

1. Check FFmpeg:
   ```bash
   docker exec montage-ai ffmpeg -version
   ```

2. Check disk space:
   ```bash
   df -h data/output/
   ```

3. Disable proxies if space limited:
   ```bash
   GENERATE_PROXIES=false ./montage-ai.sh run
   ```

---

## Technical Details

### OpenTimelineIO vs EDL

| Aspect | OpenTimelineIO (.otio) | CMX EDL (.edl) |
|--------|------------------------|----------------|
| **Standard** | Academy Software Foundation (2017) | CMX Systems (1970s) |
| **Format** | JSON | Plain text |
| **Metadata** | ✅ Rich (effects, colors, etc.) | ❌ Minimal |
| **Multi-track** | ✅ Unlimited | ⚠️ Limited |
| **Compatibility** | Modern NLEs | All NLEs (universal) |
| **Future-proof** | ✅ Active development | ⚠️ Legacy (but works) |

**Recommendation:** Use `.otio` if your NLE supports it, otherwise `.edl`.

### Frame Rate & Timecodes

Montage-AI uses:

- **Frame rate**: 30 fps (NTSC standard)
- **Timecode format**: SMPTE (HH:MM:SS:FF)
- **Drop frame**: No (non-drop frame timecode)

Example timecode: `00:01:30:15` = 1 minute, 30 seconds, 15 frames

### Color Space

Exported clips retain original color space:

- **Rec.709** (standard HD video)
- **Rec.2020** (HDR, if source is HDR)

Color grading applied by Montage-AI is **baked into output video**, but **NOT** in timeline exports (you get original ungraded clips).

To apply the same grading in NLE:

1. Export grading LUT from Montage-AI (future feature)
2. Or manually recreate in NLE

---

## Advanced: Custom Timeline Processing

Use the Python API to customize exports:

```python
from montage_ai.timeline_exporter import TimelineExporter, Timeline, Clip

# Create custom timeline
clips = [
    Clip(
        source_path="/data/input/video1.mp4",
        start_time=5.0,
        duration=3.0,
        timeline_start=0.0,
        metadata={"energy": 0.8}
    ),
    Clip(
        source_path="/data/input/video2.mp4",
        start_time=10.0,
        duration=2.5,
        timeline_start=3.0,
        metadata={"energy": 0.6}
    )
]

timeline = Timeline(
    clips=clips,
    audio_path="/data/music/track.mp3",
    total_duration=5.5,
    project_name="custom_montage"
)

# Export
exporter = TimelineExporter(output_dir="/data/output")
files = exporter.export_timeline(
    timeline,
    generate_proxies=True,
    export_otio=True,
    export_edl=True,
    export_csv=True
)

print(f"Exported: {files}")
```

---

## FAQ

**Q: Can I edit the timeline in DaVinci and re-export?**

A: Yes! After importing, you can:
- Trim clips
- Add transitions
- Color grade
- Add effects
- Then export final video from Resolve

**Q: Do I need proxies for short videos?**

A: No. Proxies are mainly useful for:
- Long montages (5+ minutes)
- 4K source footage
- Editing on lower-end hardware

**Q: Can I import timeline into Blender?**

A: Yes! Blender supports OpenTimelineIO via addon:
1. Install addon: https://github.com/GPLgithub/blender-vse-ot
2. Import `.otio` file into Video Sequence Editor

**Q: Why use timeline export vs just editing the output MP4?**

A: Timeline export gives you:
- **Original clips** (uncompressed, highest quality)
- **Cut points** (re-edit without re-rendering)
- **Metadata** (energy, motion scores)
- **Flexibility** (change music, adjust timing, add effects)

---

## References

- [OpenTimelineIO Documentation](https://opentimelineio.readthedocs.io/)
- [CMX EDL Format Specification](http://xmil.biz/EDL-X/CMX3600.pdf)
- [Academy Software Foundation](https://www.aswf.io/)
- [DaVinci Resolve Timeline Import Guide](https://documents.blackmagicdesign.com/UserManuals/DaVinci-Resolve-Manual.pdf)

---

## Next Steps

- [Features Documentation](features.md) - All Montage-AI capabilities
- [Configuration Guide](configuration.md) - Environment variables
- [Architecture](architecture.md) - System design
