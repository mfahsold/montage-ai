# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Renderer observability: stdout/stderr are now tee’d to `/data/output/render.log` (pod PVC), so full render logs persist after Job cleanup.

- **Documentation overhaul for public release**
  - `README.md`: Complete rewrite with "Why Montage-AI?" section, comparison table, architecture diagram
  - `docs/models.md`: AI models documentation (librosa, Real-ESRGAN, Llama/Gemini) with rationale and sources
  - `docs/decisions.md`: Architecture Decision Records (ADR) for all major technical choices
  - Comparison with Adobe Premiere, DaVinci Resolve, Descript
  - Links to academic papers and official documentation
- Professional development workflow
  - `Makefile` with build, deploy, test commands
  - `CONTRIBUTING.md` for external contributors
  - `docs/INSTALL.md` comprehensive installation guide
- Kubernetes deployment support (`deploy/k3s/`)
  - Kustomize-based manifests for any K8s/K3s cluster
  - `base/` - generic manifests (namespace, configmap, pvc, job)
  - `overlays/dev/` - fast preview, low resources
  - `overlays/production/` - AMD GPU targeting, HQ settings
  - CronJob template for scheduled rendering
  - Full README with deployment guide
- cgpu integration for free cloud GPU and LLM access
  - `src/montage_ai/cgpu_upscaler.py`: New module for cloud GPU upscaling via cgpu/Google Colab
    - Offloads Real-ESRGAN to free T4/A100 GPUs
    - Automatic fallback to local Vulkan/CPU when cgpu unavailable
    - Frame extraction → cloud processing → reassembly pipeline
  - `creative_director.py`: Dual backend support (Ollama + cgpu/Gemini)
    - OpenAI-compatible client for cgpu serve endpoint
    - Automatic fallback from Gemini → Ollama on failure
  - `montage-ai.sh`: cgpu management commands
    - `cgpu-start`: Launch cgpu serve in background
    - `cgpu-stop`: Stop cgpu serve process
    - `cgpu-status`: Check cgpu availability
    - `--cgpu` flag: Enable Gemini LLM backend
    - `--cgpu-gpu` flag: Enable cloud GPU for upscaling
  - New environment variables in `docker-compose.yml`:
    - `CGPU_ENABLED`: Enable cgpu/Gemini LLM backend
    - `CGPU_HOST`: cgpu serve host (default: 127.0.0.1)
    - `CGPU_PORT`: cgpu serve port (default: 5021)
    - `CGPU_MODEL`: Gemini model (default: gemini-2.5-flash)
    - `CGPU_GPU_ENABLED`: Enable cloud GPU for upscaling
    - `CGPU_TIMEOUT`: Cloud operation timeout (default: 300s)

### Changed

- K3s dev overlay tuned for CPU-heavy renders: enable STABILIZE/UPSCALE/AI filter by default, raise parallel jobs to 6, and bump dev resources to 4c/8Gi requests with 6c/16Gi limits for montage-ai job.
- Real-ESRGAN Vulkan pipeline: reassemble from PNG frames (ncnn default) to avoid ffmpeg exit 254 and enable GPU upscaling; dev overlay stays privileged to access /dev/dri on fluxibriserver.
- **cgpu integration fixes** (tested 2025-12-01)
  - `creative_director.py`: Switch from chat.completions to responses API
    - cgpu serve uses `/v1/responses` endpoint, not `/v1/chat/completions`
    - Use `instructions` + `input` parameters instead of `messages` array
  - Default model changed from `gemini-2.0-flash` to `gemini-2.5-flash`
    - Gemini 2.0 doesn't support "thinking" feature required by gemini-cli
    - Gemini 2.5+ includes thinking support
  - Default port changed from 8080 to 5021 for consistency
  - `docs/CGPU_INTEGRATION.md`: Updated with correct settings and troubleshooting

- **cgpu GPU upscaling v2.0** (tested 2025-12-01)
  - `cgpu_upscaler.py`: Complete rewrite with working cloud GPU workflow
    - Uses `cgpu copy` for file upload (upload only, no download via copy)
    - Uses `cgpu run` for Real-ESRGAN on Tesla T4
    - Uses base64 stdout for result download
    - Added torchvision compatibility patch for Colab's newer torchvision
    - Added `upscale_image_with_cgpu()` for single image upscaling
    - Fixed model weight selection (always use x4plus, outscale controls factor)
  - Verified end-to-end: 320x240 → 1280x960 (4x) on Tesla T4 GPU
- `editor.py`: Updated upscale pipeline with cgpu priority
  - Priority order: cgpu cloud GPU → local Vulkan GPU → FFmpeg CPU fallback
  - Added cgpu_upscaler import and integration
- `requirements.txt`: Added `openai>=1.0.0` for cgpu/Gemini API compatibility

### Technical

- `.github/copilot-instructions.md`: Created AI coding agent instructions
  - Architecture overview and component documentation
  - cgpu integration documented as priority task
  - Code conventions and development workflow
  - Change documentation requirements added
- Documentation overhaul - created comprehensive SOTA documentation:
  - `docs/README.md`: Documentation index with table of contents
  - `docs/features.md`: Detailed feature documentation (beat sync, styles, Creative Director, story arc, enhancement pipeline, hardware acceleration)
  - `docs/configuration.md`: Complete environment variable reference
  - `docs/styles.md`: Style guide with all presets and customization instructions
  - `docs/architecture.md`: System design with data flow diagrams
  - `docs/CGPU_INTEGRATION.md`: Updated from planning doc to implementation status
- `README.md`: Streamlined as project overview with feature list and doc links
- `BACKLOG.md`: Created structured backlog for task tracking (local only, gitignored)
- `.gitignore`: Added exclusions for local business documentation
- Style preset system is now config-first (no hardcoded styles in code)
  - JSON presets ship in `montage_ai/styles/*.json` and are packaged
  - Override via env vars: `STYLE_PRESET_DIR` (folder) or `STYLE_PRESET_PATH` (single file)
  - Loader validates presets with `jsonschema`, supports multi-style files, and allows override precedence
  - Creative Director prompt pulls available styles dynamically from loaded presets

## [0.3.0] - 2024-XX-XX

### Features

- Initial release with core montage editing capabilities
- Beat-synchronized video editing via librosa
- Creative Director LLM integration (Ollama)
- Style templates: hitchcock, mtv, documentary, action, dynamic, minimalist
- Footage Manager with story arc awareness
- Real-time monitoring and decision logging
