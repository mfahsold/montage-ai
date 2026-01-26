# Test Assets Strategy

This document describes the test asset strategy for Montage AI - how we handle test media files in a public repository without bloating the git history.

---

## Overview

| Category | Strategy | Git Tracked |
|----------|----------|-------------|
| Unit Tests | Mocking (no real media) | N/A |
| Integration Tests | On-demand download | No |
| Manual Testing | User-provided or downloaded | No |

**Key Principle**: No large media files (>100KB) are stored in the git repository.

---

## Repository Size Analysis

| Component | Size | Notes |
|-----------|------|-------|
| `.git/` | ~9 MB | Healthy, no large blobs |
| Source Code | ~3 MB | Python, JS, configs |
| Documentation | ~500 KB | Markdown files |
| **Total Tracked** | **~12 MB** | Very lean |

The `.gitignore` excludes all media:
```
data/input/
data/music/
data/output/
*.mp4
*.mp3
*.wav
*.mov
*.avi
```

---

## Test Asset Sources

### 1. NASA Public Domain (Primary)

An optional helper exists in `scripts/archive/prepare_trailer_assets.py` to download NASA footage:

```bash
# Download 4 video clips + audio (NASA public domain)
python scripts/archive/prepare_trailer_assets.py

# Custom queries
python scripts/archive/prepare_trailer_assets.py --query "apollo" --query "nebula"

# Limit download size
python scripts/archive/prepare_trailer_assets.py --max-video-mb 50 --videos 2
```

**Why NASA?**
- Public domain (U.S. Government work)
- No attribution required
- High-quality 4K footage available
- Variety of content (space, earth, technology)

### 2. Blender Open Movies (Alternative)

For higher-quality test footage, use Creative Commons films:

| Film | License | Download |
|------|---------|----------|
| Big Buck Bunny | CC-BY 3.0 | [peach.blender.org](https://peach.blender.org/download/) |
| Sintel | CC-BY 3.0 | [durian.blender.org](https://durian.blender.org/download/) |
| Tears of Steel | CC-BY 3.0 | [mango.blender.org](https://mango.blender.org/download/) |

```bash
# Manual download example
curl -L -o data/input/BigBuckBunny.mp4 \
  "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_surround-fix.avi"
```

### 3. Synthetic Test Fixtures

For minimal integration tests, generate synthetic media with FFmpeg:

```bash
mkdir -p tests/fixtures/video tests/fixtures/audio

# 5-second color bars (video)
ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=30 -t 5 \
  tests/fixtures/video/test_1080p_5s.mp4

# 10-second 120bpm sine (audio)
ffmpeg -f lavfi -i sine=frequency=440:duration=10 \
  tests/fixtures/audio/test_120bpm_10s.wav
```

---

## Test Infrastructure

### Unit Tests (Current State)

All unit tests use **mocking** - no real media files required:

```python
# Example from test_audio_analysis.py
@patch('src.montage_ai.audio_analysis.librosa')
def test_beat_detection(self, mock_librosa):
    mock_librosa.load.return_value = (np.zeros(22050), 22050)
    mock_librosa.beat.beat_track.return_value = (120.0, np.array([10, 20, 30]))
    # Test runs without any real audio file
```

**Benefits:**
- Tests run in CI without media downloads
- Fast execution (~2 seconds for full suite)
- Deterministic results

### Integration Tests (Optional)

For testing the full pipeline with real media:

```bash
# Download test assets first (optional helper)
python scripts/archive/prepare_trailer_assets.py

# Run integration tests
pytest tests/integration/ -v --run-slow
```

#### DEV opt-in e2e
- There is an opt-in developer e2e smoke test that runs a lightweight preview through the web → queue → worker flow. It is **skipped by default**; enable it with:

```bash
export RUN_DEV_E2E=true
pytest tests/integration/test_preview_e2e_dev.py -q
```

##### Preview curl (quick smoke)

For full preview SLO checks and benchmark steps, see docs/operations/preview-slo.md.

```bash
MONTAGE_API_BASE="https://<DEV_HOST>"

# Upload your clips to the DEV PVC (or use test fixtures)
curl -sS -X POST "${MONTAGE_API_BASE}/api/jobs" \
  -H 'Content-Type: application/json' \
  -d '{"video_files":["benchmark_clip_a.mp4"], "quality_profile":"preview"}' | jq

# Poll job
curl -sS "${MONTAGE_API_BASE}/api/jobs/<JOB_ID>" | jq

# Fetch metrics
curl -sS "${MONTAGE_API_BASE}/metrics" | grep montage_time_to_preview_seconds
```

#### Short-audio note / workaround ⚠️
- Short or synthetic WAVs (very short or extremely flat energy profiles) may trigger an edge-case in older versions and abort analysis. For reliable smoke runs, use a WAV >= 3–4s or the synthetic fixtures generated above.
- The codebase now includes a regression test to guard against this; if you still see failures, verify the WAV length and sample rate in `data/music/`.

Integration tests are marked with `@pytest.mark.slow` and skipped by default.

---

## Directory Structure

```
data/
  input/          # User video clips (gitignored)
  music/          # User audio tracks (gitignored)
  output/         # Rendered montages (gitignored)
  assets/         # Static assets like fonts (tracked if small)
  luts/           # LUT files + README (only README tracked)
  input_test/     # Minimal test clip (bunny_trailer.mp4, 143KB)

tests/
  fixtures/       # Synthetic test fixtures (generated, gitignored)
    video/
    audio/
  conftest.py     # Pytest fixtures
  test_*.py       # Unit tests (mocked)
  integration/    # Integration tests (require real media)
```

---

## CI/CD Considerations

### GitHub Actions

Unit tests run without any media downloads:

```yaml
- name: Run Tests
  run: pytest tests/ -v --ignore=tests/integration/
```

### Integration Tests (Manual)

For full integration testing:

```yaml
- name: Download Test Assets
  run: python scripts/archive/prepare_trailer_assets.py

- name: Run Integration Tests
  run: pytest tests/integration/ -v
```

---

## Cleanup Commands

```bash
# Remove archive folders (saves ~1GB)
rm -rf data/input/archive data/music/archive

# Remove all downloaded media
rm -rf data/input/* data/music/* data/output/*

# Remove generated fixtures
rm -rf tests/fixtures/
```

---

## Adding New Test Assets

1. **Never commit large media files** to the repository
2. Add download scripts to `scripts/` if new sources are needed
3. Update `.gitignore` if new patterns emerge
4. Document the source and license in this file

---

## License Compliance

| Source | License | Attribution Required |
|--------|---------|---------------------|
| NASA | Public Domain | No |
| Blender Films | CC-BY 3.0 | Yes (in derivative works) |
| SoundHelix | CC-BY 4.0 | Yes |
| Synthetic (FFmpeg) | N/A | No |

For commercial use, prefer NASA or synthetic assets.
