# Quick Start: Using Color Grading in Montage AI

## Overview

Color grading presets are now built into Montage AI's creative rendering pipeline. You can apply professional color grades to your generated trailers with a single parameter.

---

## Available Color Grades

| Grade | Look | Best For |
| --- | --- | --- |
| **none** | Neutral | Raw, unprocessed footage |
| **warm** | Golden, sunny | Outdoor, adventure, lifestyle content |
| **cool** | Blue, calm | Corporate, tech, dramatic content |
| **vibrant** | Saturated, pop | Travel, music, energetic content |
| **high_contrast** | Dramatic, moody | Trailers, action sequences |
| **cinematic** | Professional, desaturated | Dramatic, documentary, premium look |

---

## Web UI Method

### Step 1: Access Creative Cutter

```
http://localhost:8080/creative
```

### Step 2: Upload Footage

- Select your video files from `/data/input/`
- Or drag-drop into the interface

### Step 3: Analyze

1. Click **"Analyze Footage"**
2. Set target duration (default: 45s)
3. Wait for analysis to complete (~1-2 min for 55 videos)
4. Review suggested cuts

### Step 4: Render with Color Grade

1. In **"Rendering Options"**, select your color grade:
   - `none` (default)
   - `warm`
   - `cool`
   - `vibrant`
   - `high_contrast`
   - `cinematic`

2. Click **"Render Video"**
3. Wait for rendering (~30 seconds for 30s trailer)
4. Download finished video

---

## CLI Method

### Python Script

```python
from analyze_footage_creative import analyze_and_plan_creative_cut
from render_creative_cut import render_with_plan

# Step 1: Analyze footage
print("📊 Analyzing footage...")
cut_plan = analyze_and_plan_creative_cut(target_duration=45)

# Step 2: Render with color grade
print("🎬 Rendering with warm color grade...")
result = render_with_plan(cut_plan, color_grade="warm")

if result['success']:
    print(f"✅ Done! Saved to: {result['output_file']}")
    print(f"   Size: {result['file_size'] / (1024*1024):.1f}MB")
else:
    print(f"❌ Error: {result['error']}")
```

### Bash Command

```bash
#!/bin/bash
cd /home/codeai/montage-ai

# Generate trailer with cinematic grade
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/codeai/montage-ai')

from analyze_footage_creative import analyze_and_plan_creative_cut
from render_creative_cut import render_with_plan
import json

cut_plan = analyze_and_plan_creative_cut(target_duration=30)
result = render_with_plan(cut_plan, color_grade="cinematic")

print(json.dumps(result, indent=2))
EOF
```

---

## REST API Method

### 1. Start Analysis

```bash
curl -X POST http://localhost:8080/api/creative/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_session_123",
    "target_duration": 30
  }'
```

**Response:**
```json
{
  "job_id": "creative_user_session_123_1739145600",
  "status": "queued",
  "message": "Analysis job queued"
}
```

### 2. Poll Analysis Status

```bash
JOB_ID="creative_user_session_123_1739145600"
curl http://localhost:8080/api/creative/analyze/$JOB_ID
```

**Response (when complete):**
```json
{
  "job_id": "creative_user_session_123_1739145600",
  "status": "analyzed",
  "progress": 100,
  "cut_plan": {
    "target_duration": 30,
    "cuts": [...],
    "total_cuts": 31,
    ...
  }
}
```

### 3. Render with Color Grade

```bash
curl -X POST http://localhost:8080/api/creative/render \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_session_123",
    "cut_plan": {
      "cuts": [...],
      "target_duration": 30,
      ...
    },
    "color_grade": "warm"
  }'
```

**Response:**
```json
{
  "job_id": "creative_user_session_123_1739145600_render",
  "status": "queued",
  "message": "Rendering job queued (color_grade=warm)",
  "color_grade": "warm"
}
```

### 4. Poll Render Status

```bash
RENDER_JOB_ID="creative_user_session_123_1739145600_render"
curl http://localhost:8080/api/creative/render/$RENDER_JOB_ID
```

**Response (when complete):**
```json
{
  "job_id": "creative_user_session_123_1739145600_render",
  "status": "completed",
  "progress": 100,
  "output_file": "/home/codeai/montage-ai/data/output/gallery_montage_creative_trailer_rendered_warm.mp4",
  "file_size": 38567424,
  "color_grade": "warm"
}
```

---

## Output Files

All rendered videos are saved to `/data/output/`:

```
gallery_montage_creative_trailer_rendered_none.mp4          (36.8MB)
gallery_montage_creative_trailer_rendered_warm.mp4          (36.8MB)
gallery_montage_creative_trailer_rendered_cool.mp4          (36.8MB)
gallery_montage_creative_trailer_rendered_vibrant.mp4       (36.8MB)
gallery_montage_creative_trailer_rendered_high_contrast.mp4 (36.8MB)
gallery_montage_creative_trailer_rendered_cinematic.mp4     (36.8MB)
```

---

## Technical Details

### Grading Quality
- **Color Space:** Rec.709 (standard for web/broadcast)
- **Bit Rate:** CRF 18 (high quality)
- **Frame Rate:** CFR 30 fps (stable, no stuttering)
- **Audio:** AAC 192 kbps
- **Container:** MP4 (web-optimized with faststart)

### Performance
- **Analysis:** ~2 minutes for 55 videos (one-time)
- **Rendering:** ~30 seconds per color grade preset
- **Memory:** <500MB peak
- **Storage:** ~37MB per finished video

---

## Troubleshooting

### Issue: Color grade not applied

**Solution:** Check that color_grade parameter is spelled correctly:
- ✅ `"warm"`
- ✅ `"cool"`
- ✅ `"vibrant"`
- ✅ `"high_contrast"`
- ✅ `"cinematic"`
- ✅ `"none"`
- ❌ `"warm_tone"` (wrong)
- ❌ `"Warm"` (wrong - case-sensitive)

### Issue: Rendering fails with timeout

**Solution:** Increase timeout for large video files
```bash
# CLI: Rendering already has 15-min timeout
# API: Job will retry automatically
```

### Issue: Output file size very small or missing

**Solution:** Check `/tmp/montage_normalized/` for intermediate files
```bash
ls -lh /tmp/montage_normalized/
ls -lh /data/output/gallery_montage_creative_trailer_rendered_*.mp4
```

---

## Advanced Usage

### Batch Processing Multiple Grades

```python
from analyze_footage_creative import analyze_and_plan_creative_cut
from render_creative_cut import render_with_plan
import json
from pathlib import Path

# Analyze once
cut_plan = analyze_and_plan_creative_cut(target_duration=30)

# Render with all grades
grades = ['none', 'warm', 'cool', 'vibrant', 'high_contrast', 'cinematic']
results = {}

for grade in grades:
    print(f"Rendering {grade}...")
    result = render_with_plan(cut_plan, color_grade=grade)
    results[grade] = result

# Save results
output_file = Path("/data/output/batch_render_results.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ All renders complete! Results saved to {output_file}")
```

### Comparing Grades

```python
# Render with different grades and compare file sizes
from render_creative_cut import render_with_plan
from pathlib import Path

cut_plan = {...}
output_dir = Path("/data/output")

print(f"{'Grade':<20} {'File Size':<15} {'Status':<15}")
print("-" * 50)

for grade in ['none', 'warm', 'cool', 'vibrant', 'high_contrast', 'cinematic']:
    result = render_with_plan(cut_plan, color_grade=grade)
    if result['success']:
        size_mb = result['file_size'] / (1024*1024)
        print(f"{grade:<20} {size_mb:.1f}MB{'':<10} ✅ Success")
    else:
        print(f"{grade:<20} {'—':<15} ❌ {result['error']}")
```

---

## Next Steps

1. **Test** different color grades on your footage
2. **Collect feedback** on which grades look best
3. **Consider adding** custom presets if needed
4. **Optimize** based on your audience preferences

## Support

For issues or questions:
- Check logs: `kubectl logs -n montage-ai deployment/montage-ai-web`
- Review test results: [QUALITY_ENHANCEMENTS_TEST_REPORT_2026_02_10.md](./QUALITY_ENHANCEMENTS_TEST_REPORT_2026_02_10.md)
- Consult architecture docs: [docs/architecture.md](./docs/architecture.md)
