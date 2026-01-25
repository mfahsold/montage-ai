# Optional Dependencies

Montage AI installs with a minimal core set of dependencies by default. Additional features require optional dependency groups that can be installed selectively.

---

## Installation Methods

### Core Only (Lightweight)
```bash
pip install montage-ai
```
Includes: Video processing, FFmpeg integration, timeline export (no AI, web UI, or cloud features)

### With AI Enhancements
```bash
pip install montage-ai[ai]
```
Adds:
- **mediapipe** — Smart Reframing with face detection
- **scipy** — Path optimization for smooth camera motion
- **librosa** — Audio analysis fallback (FFmpeg is primary)
- **color-matcher** — Shot-to-shot color consistency

### With Web UI
```bash
pip install montage-ai[web]
```
Adds:
- **Flask** — Web framework
- **Werkzeug** — WSGI utilities
- **redis** + **rq** — Background job processing
- **msgpack** — Fast job serialization

### With Cloud GPU Support
```bash
pip install montage-ai[cloud]
```
Adds:
- **cgpu** — Cloud GPU orchestration for upscaling and analysis
- **soundfile** — Audio handling for cloud jobs

### For Development
```bash
pip install montage-ai[test]
```
Adds:
- **pytest** — Test framework
- **pytest-flask** — Flask testing utilities

### Everything (Development)
```bash
pip install montage-ai[all]
```
Includes all optional groups (equivalent to `pip install -r requirements.txt`)

---

## Feature Matrix

| Feature | Core | `[ai]` | `[web]` | `[cloud]` | `[test]` |
|---------|:----:|:-----:|:------:|:---------:|:--------:|
| **Video Editing** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FFmpeg Integration** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Timeline Export (OTIO)** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **CLI Interface** | ✅ | ✅ | ✅ | ✅ | ✅ |
| | | | | | |
| **Smart Reframing** | — | ✅ | ✅ | ✅ | ✅ |
| **Face Detection** | — | ✅ | ✅ | ✅ | ✅ |
| **Color Matching** | — | ✅ | ✅ | ✅ | ✅ |
| **Advanced Audio Analysis** | — | ✅ | ✅ | ✅ | ✅ |
| | | | | | |
| **Web UI (default port 8080)** | — | — | ✅ | ✅ | ✅ |
| **Async Job Queue** | — | — | ✅ | ✅ | ✅ |
| **REST API** | — | — | ✅ | ✅ | ✅ |
| | | | | | |
| **Cloud GPU Upscaling** | — | — | — | ✅ | ✅ |
| **Remote Analysis Jobs** | — | — | — | ✅ | ✅ |
| | | | | | |
| **Unit Tests** | — | — | — | — | ✅ |
| **Integration Tests** | — | — | — | — | ✅ |

---

## Dependency Details

### Core Dependencies (Always Installed)

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| **moviepy** | ≥2.2.1 | Video composition, effects, rendering | ~15 MB |
| **Pillow** | ≥10.0.0 | Image processing (embedded in moviepy) | ~8 MB |
| **opencv-python-headless** | ≥4.12.0.88 | Computer vision (scene/motion detection) | ~85 MB |
| **numpy** | ≥1.24, <2.0 | Numerical computing (pinned <2.0 for stability) | ~30 MB |
| **scenedetect** | ≥0.6.7.1 | Scene boundary detection | ~3 MB |
| **OpenTimelineIO** | ≥0.18.1 | Timeline/EDL interchange (ASWF standard) | ~2 MB |
| **tqdm** | ≥4.67.1 | Progress bars and logging | ~1 MB |
| **requests** | ≥2.32.5 | HTTP client (LLM APIs) | ~2 MB |
| **jsonschema** | ≥4.25.1 | JSON validation (Creative Director) | ~1 MB |
| **psutil** | ≥7.1.3 | System resource monitoring | ~1 MB |
| | | **Total Core** | **~148 MB** |

### AI Enhancements (`[ai]`)

| Package | Version | Purpose | Notes | Size |
|---------|---------|---------|-------|------|
| **mediapipe** | ≥0.10.0 | Face detection for smart reframing | Optional; fallback to bounding box detection | ~300 MB |
| **scipy** | ≥1.10.0 | Path optimization for camera motion | Optional; fallback to linear interpolation | ~45 MB |
| **librosa** | ≥0.10.0 | Audio analysis fallback | FFmpeg is primary; librosa is conditional | ~25 MB |
| **color-matcher** | ≥0.5.0 | Shot-to-shot color grading | Optional; fallback: no color matching | ~2 MB |

### Web UI (`[web]`)

| Package | Version | Purpose | Notes | Size |
|---------|---------|---------|-------|------|
| **Flask** | ≥3.0.0 | Web framework | — | ~2 MB |
| **Werkzeug** | ≥3.0.0 | WSGI utilities | Flask dependency | ~1 MB |
| **redis** | ≥5.0.0 | In-memory data store (job queue backend) | Requires external Redis service | ~2 MB |
| **rq** | ≥1.16.0 | Background job processing | Simple job queue using Redis | ~1 MB |
| **msgpack** | ≥1.0.0 | Binary serialization (22x faster than JSON) | Job serialization | ~1 MB |

### Cloud GPU (`[cloud]`)

| Package | Version | Purpose | Notes | Size |
|---------|---------|---------|-------|------|
| **cgpu** | ≥0.4.0 | Cloud GPU orchestration | Requires fluxibri/CGPU service | ~5 MB |
| **soundfile** | ≥0.12.0 | Audio file I/O | Cloud job audio handling | ~1 MB |

### Testing (`[test]`)

| Package | Version | Purpose | Notes | Size |
|---------|---------|---------|-------|------|
| **pytest** | ≥8.0.0 | Test framework | — | ~5 MB |
| **pytest-flask** | ≥1.3.0 | Flask testing fixtures | Provides `client` fixture | ~1 MB |

---

## Conditional Imports

All optional dependencies are wrapped in try/except blocks with graceful fallbacks. If a package is missing:

```python
try:
    import mediapipe
    face_detection_available = True
except ImportError:
    face_detection_available = False
    # Fall back to bounding box detection or skip reframing
```

---

## Common Scenarios

### Local Development (No GPU, No Web)
```bash
pip install montage-ai[ai,test]
```
Enables smart reframing, audio analysis, and unit tests locally.

### Production Server (Web UI + Job Queue)
```bash
pip install montage-ai[ai,web]
```
Requires separate Redis instance. Enables full web interface with background processing.

### Cloud Offloading (GPU Upscaling)
```bash
pip install montage-ai[ai,cloud]
```
Assumes CGPU service is available. Enables cloud-based upscaling and analysis.

### Docker / Full-Featured
```bash
pip install montage-ai[all]
```
All optional features included; can selectively enable via environment variables.

---

## Troubleshooting

### "No module named 'mediapipe'"
You're using the core installation. Install `[ai]` for smart reframing:
```bash
pip install montage-ai[ai]
```
Or disable smart reframing:
```bash
SMART_REFRAME_ENABLED=false ./montage-ai.sh run
```

### "ConnectionError: Can't connect to Redis"
You're using the web UI but Redis is not running. Either:
1. Start Redis: `redis-server`
2. Use core CLI instead: `./montage-ai.sh run [STYLE]` (no web UI)

### "ImportError in cgpu_jobs"
Cloud GPU features require the `[cloud]` group:
```bash
pip install montage-ai[cloud]
```
Or disable cloud jobs in config:
```bash
CGPU_ENABLED=false
```

---

## Size Estimates

| Installation | Size | Time (typical) |
|--------------|------|----------------|
| Core only | ~148 MB | 30s |
| Core + AI | ~420 MB | 90s |
| Core + Web | ~155 MB | 45s |
| Core + Cloud | ~155 MB | 45s |
| All | ~550 MB | 180s |

(Sizes are approximate; depends on platform and network speed.)

---

## Security Notes

### Redis Password Protection
If deploying with `[web]`, always configure Redis authentication:

```bash
redis-server --requirepass your-secure-password
export REDIS_PASSWORD=your-secure-password
```

### CGPU Service Authentication
Cloud GPU support requires valid CGPU credentials. Never commit credentials to git:

```bash
export CGPU_API_KEY=sk-...
export CGPU_ENDPOINT=https://cgpu.example.com
```

---

## References

- [pip Optional Dependencies](https://pip.pypa.io/en/latest/cli/pip_install/#install-extras)
- [setuptools Optional Dependencies](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [RQ (Job Queue) Documentation](https://python-rq.org/)
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
