# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **cgpu Cloud GPU in Docker** (`Dockerfile`, `docker-compose.web.yml`, `docker-compose.yml`)
  - Install Node.js 20 LTS and cgpu in Docker image for cloud GPU access
  - Mount cgpu credentials (`~/.config/cgpu`) into container
  - New environment variables: `CGPU_GPU_ENABLED=true`, `CGPU_TIMEOUT=600`
  - AI upscaling now uses Google Colab T4 GPU via cgpu instead of local Vulkan
  - Fallback chain: cgpu Cloud GPU → Local Vulkan GPU → FFmpeg Lanczos

- **Direct Google AI Integration** (`creative_director.py` v0.3.0)
  - New `GOOGLE_API_KEY` environment variable for direct Gemini API access
  - `_query_google_ai()` method bypasses cgpu serve / gemini-cli entirely
  - Uses `generativelanguage.googleapis.com` REST API with JSON output mode
  - Backend priority: Google AI (API Key) > cgpu serve > Ollama (local)
  - New env vars: `GOOGLE_API_KEY`, `GOOGLE_AI_MODEL=gemini-2.0-flash`
  - Automatic fallback to Ollama if Google AI fails

- **cgpu Utilities Module** (`src/montage_ai/cgpu_utils.py`)
  - Centralized cgpu cloud GPU utilities to eliminate code duplication
  - `CGPUConfig` dataclass for unified configuration (host, port, model, timeout)
  - `is_cgpu_available()`: Check if cgpu is installed and authenticated
  - `check_cgpu_gpu()`: Verify GPU access on Colab
  - `run_cgpu_command()`: Execute commands on cloud GPU
  - `cgpu_copy_to_remote()`: Upload files to Colab
  - `cgpu_download_base64()`: Download files via stdout
  - `setup_cgpu_environment()`: Install dependencies on Colab
  - `get_cgpu_llm_client()`: Get OpenAI-compatible client for cgpu serve

- **Open-Source AI Integration** (Sprint 1-2)
  - `docs/INTEGRATION_PLAN.md`: Comprehensive integration plan for VideoAgent, Wan2.1-VACE, Open-Sora
  - Architecture diagrams and sprint breakdown for implementation
  
- **FFmpeg Tools** (`src/montage_ai/ffmpeg_tools.py`)
  - LLM-callable FFmpeg wrapper replacing external FFmpeg-MCP dependency
  - 10 tools: extract_frames, create_segment, apply_transition, color_grade, mix_audio, resize, concatenate, get_video_info, thumbnail_grid, speed_change
  - OpenAI function calling schema for LLM integration
  - Built-in color grading presets (cinematic, vintage, cold, warm, noir, vivid)
  
- **VideoAgent Integration** (`src/montage_ai/video_agent.py`)
  - Memory-augmented clip analysis inspired by ECCV 2024 VideoAgent paper
  - Temporal Memory: Scene captions with embeddings for semantic search
  - Object Memory: SQL-backed object tracking across video
  - 4 Core Tools: caption_retrieval, segment_localization, visual_question_answering, object_memory_querying
  - Integration with footage_manager.py for story-arc aware clip selection
  
- **Wan2.1-VACE Service** (`src/montage_ai/wan_vace.py`)
  - Video generation and editing via Alibaba Wan2.1 model
  - Text-to-Video (T2V) B-Roll generation
  - cgpu cloud GPU execution for free T4 access
  - WanBRollGenerator with templates: transition, filler, establishing, detail, action
  - Supports 480p output with 1.3B model (8GB VRAM)
  
- **Open-Sora Generator** (`src/montage_ai/open_sora.py`)
  - Text-to-Video generation via HPC-AI Tech Open-Sora (Apache-2.0)
  - 256p generation on single T4 GPU
  - Automatic 4x upscaling with Real-ESRGAN to 1024p output
  - OpenSoraPromptEnhancer for better generation results
  - Diffusers fallback to ModelScope T2V if Open-Sora unavailable

### Changed

- **Package version** bumped to 0.4.0
- **`__init__.py`** updated with all new modules and feature flags
  - FFMPEG_TOOLS_AVAILABLE, VIDEO_AGENT_AVAILABLE, WAN_VACE_AVAILABLE, OPEN_SORA_AVAILABLE
  - CGPU_UTILS_AVAILABLE flag for shared utilities
  - Comprehensive docstrings with usage examples

### Refactored

- **cgpu code consolidation**: Eliminated duplicate `_run_cgpu` functions across modules
  - `cgpu_upscaler.py` v2.1.0: Now imports from `cgpu_utils`
  - `wan_vace.py` v1.1.0: Uses shared `run_cgpu_command()` and `cgpu_copy_to_remote()`
  - `open_sora.py` v1.1.0: Uses shared cgpu utilities
  - Reduced code duplication by ~150 lines

### Technical

- All new modules use consistent error handling with feature flags
- cgpu integration follows established patterns from cgpu_upscaler.py
- SQLite-backed storage for VideoAgent memory (persistent across sessions)
- Centralized cgpu configuration via `CGPUConfig` dataclass

---

## Previous Releases

### Added
- **Professional Timeline Export**: Fully integrated OTIO/EDL/CSV export for importing montages into professional NLE software
  - `src/montage_ai/timeline_exporter.py`: Complete export implementation with clip metadata collection
  - OpenTimelineIO (.otio) export - Academy Software Foundation standard for DaVinci Resolve, Premiere Pro, Final Cut Pro
  - CMX 3600 EDL (.edl) export - Universal format compatible with all professional NLEs
  - CSV spreadsheet (.csv) export - Human-readable timeline with timecodes, energy levels, shot types
  - Clip metadata collection during assembly: tracks source files, in/out points, timeline positions, energy, action levels, enhancements applied
  - Persistent export to PVC (`/data/output/*.{otio,edl,csv}`) alongside rendered video and logs
  - Enabled by default in K8s ConfigMap (`EXPORT_TIMELINE: "true"`)
- **Web Upload API** (initial implementation): User-friendly file upload interface for non-technical users
  - `src/montage_ai/api.py`: FastAPI server with multi-part file upload endpoints
  - Upload endpoints: `/api/upload/video`, `/api/upload/music`, `/api/upload/logo`
  - File management: list, delete, clear operations via REST API
  - Simple web interface at `/` for drag-and-drop uploads
  - Output file download via `/api/download/{filename}`
  - Health check endpoint for K8s probes
- Renderer observability: stdout/stderr are now tee'd to `/data/output/render.log` (pod PVC), so full render logs persist after Job cleanup.
- K3s dev defaults tuned for stability: UPSCALE off by default, MAX_PARALLEL_JOBS=4, dev job memory limit raised to 48Gi (was 16Gi).
- Clip variety: Pool bleibt flexibel, aber mit Reuse-Cap (`MAX_SCENE_REUSE`, default 3). AI darf Clips wiederverwenden, aber nicht exzessiv.
- FFmpeg MCP optional: `USE_FFMPEG_MCP` + `FFMPEG_MCP_ENDPOINT` schalten Subclip-Extraction auf einen MCP-Service um (Fallback: lokales ffmpeg).

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
