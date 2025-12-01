# Architecture Decision Records (ADR)

This document records significant architectural decisions made for Montage-AI.

---

## ADR-001: Beat Detection Library

**Status:** Accepted  
**Date:** 2025-01

### Context

Montage-AI needs reliable beat detection to align video cuts with music rhythm. This is a core feature that directly impacts output quality.

### Decision

Use [librosa](https://librosa.org/) for beat detection.

### Alternatives Considered

| Library       | Accuracy | Speed  | Dependencies | Notes                        |
| ------------- | -------- | ------ | ------------ | ---------------------------- |
| **librosa** ✓ | High     | Medium | NumPy, SciPy | Pure Python, well-documented |
| Madmom        | Higher   | Fast   | TensorFlow   | Heavy dependencies           |
| Essentia      | Highest  | Fast   | C++ build    | Complex ARM compilation      |
| aubio         | Medium   | Fast   | C library    | Less accurate beat tracking  |

### Consequences

**Positive:**

- Portable across all platforms (ARM64, x86)
- No GPU required
- Extensive documentation and examples
- Actively maintained (8k+ stars)

**Negative:**

- Slower than native alternatives (~2-3s per 30s audio)
- Higher memory usage than C libraries

### References

- [librosa beat_track documentation](https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html)
- [SciPy 2015 Paper](https://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf)

---

## ADR-002: Scene Detection Approach

**Status:** Accepted  
**Date:** 2025-01

### Context

To build montages, we need to identify natural cut points within source clips. This enables selecting meaningful segments rather than arbitrary slices.

### Decision

Use [PySceneDetect](https://scenedetect.com/) with ContentDetector.

### Alternatives Considered

| Approach                                           | Quality | Integration | Notes              |
| -------------------------------------------------- | ------- | ----------- | ------------------ |
| **PySceneDetect** ✓                                | High    | Easy        | Industry standard  |
| FFmpeg filter (`-filter:v "select=gt(scene,0.4)"`) | Medium  | Complex     | Raw threshold only |
| Manual threshold on histogram diff                 | Low     | DIY         | High maintenance   |
| TransNetV2 (neural network)                        | Highest | Complex     | GPU required       |

### Configuration Choice

Default threshold: `27.0` (ContentDetector)

- Lower than library default (30.0) based on testing with travel/action footage
- Detects more cuts, better for fast-paced montages
- Configurable via style templates

### Consequences

**Positive:**

- Production-proven (used by Netflix)
- Multiple detector options (Content, Adaptive, Threshold)
- Built-in FFmpeg integration for video splitting

**Negative:**

- Python overhead vs native FFmpeg
- No built-in GPU acceleration

### References

- [PySceneDetect Benchmark](https://github.com/Breakthrough/PySceneDetect/blob/main/benchmark/README.md)
- [ContentDetector API](https://www.scenedetect.com/docs/latest/api/detectors.html)

---

## ADR-003: Video Composition Library

**Status:** Accepted  
**Date:** 2025-01

### Context

Need to concatenate clips, apply transitions, and render final video. Requires balance between ease of use and performance.

### Decision

Use [MoviePy](https://zulko.github.io/moviepy/) for video composition.

### Alternatives Considered

| Library           | API Complexity | Performance | Effects Support |
| ----------------- | -------------- | ----------- | --------------- |
| **MoviePy** ✓     | Simple         | Slow        | Built-in        |
| PyAV              | Complex        | Fast        | Manual          |
| Vapoursynth       | Complex        | Fast        | Plugin-based    |
| FFmpeg subprocess | Very Complex   | Fastest     | Filter syntax   |

### Rationale

For a batch processing tool where development velocity matters more than runtime performance:

1. **Simple clip concatenation** - `concatenate_videoclips([clip1, clip2])`
2. **Built-in effects** - Transitions, text overlays, color adjustments
3. **Good documentation** - Quick iteration on new features

### Future Consideration

Hybrid approach may be adopted:

- PyAV for frame extraction (fast)
- MoviePy for composition (easy)
- FFmpeg for final encoding (optimized)

### Consequences

**Positive:**

- Fast development iteration
- Readable code
- Good error messages

**Negative:**

- 2-3x slower than PyAV for large videos
- Higher memory usage
- Pulls in ImageMagick dependency

---

## ADR-004: AI Upscaling Model

**Status:** Accepted  
**Date:** 2025-01

### Context

Optional 4x video upscaling for high-quality output. Must work without NVIDIA GPU (Vulkan/CPU fallback).

### Decision

Use [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) with `realesr-animevideov3` model.

### Alternatives Considered

| Model                  | Quality | Speed  | License    | GPU Required |
| ---------------------- | ------- | ------ | ---------- | ------------ |
| **Real-ESRGAN ncnn** ✓ | High    | Medium | BSD-3      | No (Vulkan)  |
| Real-ESRGAN (PyTorch)  | High    | Fast   | BSD-3      | Yes (CUDA)   |
| Topaz Video AI         | Highest | Fast   | Commercial | Yes          |
| Waifu2x                | Medium  | Fast   | MIT        | No           |
| ESRGAN (original)      | High    | Medium | Apache 2.0 | Yes          |

### Model Choice: animevideov3

Selected `realesr-animevideov3` over `realesrgan-x4plus` because:

1. **Temporal consistency** - Designed for video, reduces flickering
2. **Smaller model** - Faster inference
3. **Good on real footage** - Despite "anime" name, works well on general video

### Consequences

**Positive:**

- Works on ARM64 (Apple Silicon, Raspberry Pi)
- Works on AMD GPUs via Vulkan
- No CUDA/PyTorch installation required
- State-of-the-art quality (ICCVW 2021)

**Negative:**

- Slower than CUDA version
- Requires building from source on ARM64
- Model files add ~50MB to Docker image

### References

- [Real-ESRGAN Paper](https://arxiv.org/abs/2107.10833)
- [Video Model Comparison](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md)

---

## ADR-005: LLM Backend Strategy

**Status:** Accepted  
**Date:** 2025-01

### Context

Creative Director component uses LLM to translate natural language prompts into structured editing parameters. Need flexibility between local (privacy) and cloud (speed).

### Decision

Dual backend support:

1. **Primary:** Ollama (local Llama 3.1)
2. **Alternative:** cgpu (cloud Gemini 2.0)

### Alternatives Considered

| Backend           | Privacy | Speed | Cost | Setup   |
| ----------------- | ------- | ----- | ---- | ------- |
| **Ollama** ✓      | Full    | 3-5s  | Free | Easy    |
| **cgpu/Gemini** ✓ | Cloud   | 0.3s  | Free | Easy    |
| OpenAI API        | Cloud   | 0.5s  | Paid | Easy    |
| vLLM              | Full    | 1-2s  | Free | Complex |
| llama.cpp         | Full    | 2-3s  | Free | Medium  |

### Model Selection

| Use Case   | Recommended      | Rationale                   |
| ---------- | ---------------- | --------------------------- |
| Production | Gemini 2.0 Flash | Fastest, most accurate JSON |
| Privacy    | Llama 3.1 8B     | Local, consumer hardware    |
| Offline    | Llama 3.1 8B     | No internet required        |

### Consequences

**Positive:**

- User choice between privacy and speed
- No API costs (both options free)
- Graceful fallback if one backend unavailable

**Negative:**

- Two code paths to maintain
- Different JSON accuracy between models
- Ollama requires 8GB+ RAM

---

## ADR-006: Container Strategy

**Status:** Accepted  
**Date:** 2025-01

### Context

Need consistent deployment across development, CI, and production. Video processing has complex dependencies (FFmpeg, Real-ESRGAN, Python libraries).

### Decision

Docker-first approach with multi-stage builds.

### Architecture

```text
Dockerfile (multi-stage)
├── Stage 1: Build Real-ESRGAN from source
├── Stage 2: Python dependencies
└── Stage 3: Runtime (slim image)
```

### Image Variants

| Tag      | Size   | Use Case              |
| -------- | ------ | --------------------- |
| `latest` | ~1.2GB | Full features         |
| `slim`   | ~800MB | No upscaling (future) |

### Consequences

**Positive:**

- Reproducible builds
- No "works on my machine" issues
- Easy Kubernetes deployment
- Cross-platform (linux/amd64, linux/arm64)

**Negative:**

- Large image size (~1.2GB)
- Slow initial build (10-15 min)
- No GPU passthrough by default

---

## ADR-007: Storage Architecture (Kubernetes)

**Status:** Accepted  
**Date:** 2025-12

### Context

Kubernetes deployment needs persistent storage for input footage, music, assets, and output.

### Decision

Separate PersistentVolumeClaims per data type.

### PVC Layout

| PVC              | Size  | Access Mode   | Purpose         |
| ---------------- | ----- | ------------- | --------------- |
| `montage-input`  | 50Gi  | ReadWriteOnce | Source footage  |
| `montage-music`  | 10Gi  | ReadWriteOnce | Audio tracks    |
| `montage-assets` | 5Gi   | ReadWriteOnce | Overlays, logos |
| `montage-output` | 100Gi | ReadWriteOnce | Rendered videos |

### Why ReadWriteOnce?

- Most storage classes don't support ReadWriteMany
- Job-based workload doesn't need concurrent access
- Simpler than NFS/shared storage

### Consequences

**Positive:**

- Works with any storage class
- Clear data organization
- Easy to size independently

**Negative:**

- Can't run multiple jobs concurrently on same PVCs
- Data loading requires helper pod or kubectl cp

---

## Summary

| Decision          | Choice             | Key Rationale                   |
| ----------------- | ------------------ | ------------------------------- |
| Beat detection    | librosa            | Portability, documentation      |
| Scene detection   | PySceneDetect      | Industry standard, configurable |
| Video composition | MoviePy            | Development speed               |
| AI upscaling      | Real-ESRGAN ncnn   | Works without CUDA              |
| LLM backend       | Ollama + cgpu      | Privacy + speed options         |
| Container         | Docker multi-stage | Reproducibility                 |
| K8s storage       | Separate RWO PVCs  | Compatibility                   |

All decisions prioritize:

1. **Open source** — No vendor lock-in
2. **Portability** — ARM64 + x86, no NVIDIA requirement
3. **Simplicity** — Maintainable by small team
