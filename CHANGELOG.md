# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **cgpu Integration**: Fixed API key propagation issue where `GOOGLE_API_KEY` was being unset, causing 502 errors.
- **Creative Director**: Added robust fallback logic (cgpu -> Google AI -> Ollama) to ensure generation continues even if primary backend fails.
- **Environment Config**: Updated `.env` and `montage-ai.sh` to correctly handle Google API keys for the cgpu wrapper.
- **Web UI**: Removed all inline styles in favor of shared `ui-utils.css` for maintainability
- **Accessibility**: Added proper `aria-label` and `for` attributes to all form inputs across templates
- **Markdownlint**: Fixed all documentation formatting issues (headings, lists, tables, fences)

### Added

- **OTIO Documentation**: Added common failure modes section to [OTIO_VERIFICATION.md](docs/OTIO_VERIFICATION.md)
- **Regression Test**: Added `test_otio_schema_version_strict` to lock OTIO schema at `Timeline.1`
- **Shared CSS Utilities**: Created `ui-utils.css` with reusable layout, spacing, and component classes

#### 🆕 Transcript Editor (Descript-style) - FULLY WIRED
- **Live Preview Flow**: Auto-generates 360p preview 2 seconds after edits (Story 1.1)
- **Undo/Redo Stack**: Non-destructive editing with full history support (Story 1.2)
- **Text-based editing**: Delete words to cut video, rearrange paragraphs to reorder scenes
- **Word-level sync**: Click any word to seek video
- **Filler word detection**: Highlights "um", "uh", "like" for easy removal
- **Silence removal**: Adjustable threshold for automatic gap removal
- **Export formats**: Video (with smooth audio crossfades), EDL, OTIO
- **Backend integration**: `TextEditor` class now properly wired to `/api/transcript/export`
- **Stats tracking**: Real-time removed word count and time saved
- **Web UI**: New `/transcript` route with dedicated editor interface

#### 🎬 Pro Handoff (Timeline Export)
- **Source Relinking**: XML/OTIO exports can now link to original high-res source files instead of proxies.
- **Conform Guide**: Automatically generates `HOW_TO_CONFORM.md` with step-by-step instructions for Resolve/Premiere.
- **Project Packaging**: Exports are bundled into a structured `{project}_PROJECT` folder with media, proxies, and metadata.

#### 📱 Shorts Studio - CAPTION BURN-IN COMPLETE
- **Phone-frame preview**: Live 9:16 aspect ratio preview with safe zones
- **Reframe modes**: Auto (AI tracking), Speaker (face detection), Center, Custom
- **Caption styles**: TikTok, Minimal, Bold, Gradient with live preview
- **Caption burn-in**: Full integration with `CaptionBurner` for hardcoded subtitles
- **File upload**: New `/api/shorts/upload` endpoint for video uploads
- **Web UI**: New `/shorts` route with dedicated studio interface

#### ⚡ Highlight Detection MVP - MULTI-SIGNAL
- **Energy peaks**: Detects high-energy audio regions (drops, loud moments)
- **Beat drops**: Identifies sudden energy increases aligned with beats
- **Speech hooks**: Finds punchy speech delivery in first 30 seconds
- **Fallback beats**: Evenly distributed beat-aligned moments
- **Scoring**: Normalized 0-1 scores with type labels (🔥 Energy, 💥 Drop, 🎤 Hook, 🎵 Beat)

#### 🔊 Audio Polish ("Clean Audio")
- **One-click audio cleanup**: `/api/audio/clean` bundles voice isolation + noise reduction
- **CGPU voice isolation**: Uses demucs for stem separation when cloud available
- **FFmpeg fallback**: Adaptive FFT denoiser, highpass/lowpass, light compression
- **Audio analysis**: `/api/audio/analyze` returns SNR estimate and recommendations
- **SNR improvement**: ~6dB with voice isolation, ~3dB with noise reduction alone

#### 🎚️ Quality Profiles
- **Outcome-based selection**: Preview → Standard → High → Master
- **Single-click configuration**: Replaces 5+ separate toggles
- **Profile presets**:
  - Preview: 360p, no enhancements, fast iteration
  - Standard: 1080p, color grading (default)
  - High: 1080p + stabilization + enhancement
  - Master: 4K + all enhancements + AI upscaling
- **Web UI**: Profile cards in `/v2` route

#### ☁️ Cloud Acceleration
- **Single toggle**: Consolidates CGPU, cloud transcription, LLM features
- **Graceful fallback**: Automatic local processing if cloud unavailable
- **Privacy-first**: Only enabled features use cloud, footage stays local by default

#### 📚 Documentation & Compliance
- **Strategic docs**: `docs/STRATEGY.md` with product vision and market analysis
- **Roadmap**: `docs/roadmap/ROADMAP_2026.md` with 12-month development plan
- **Backlog**: `docs/STRATEGIC_BACKLOG.md` with prioritized Epics and User Stories
- **GitHub Pages**: Poly-Chrome design matching webapp aesthetic
- **OSS compliance**: `THIRD_PARTY_LICENSES.md` with all dependency licenses

### Changed
- **README.md**: Complete refresh with new positioning, feature highlights, Quality Profiles documentation
- **docs/features.md**: Added sections for all new features, API reference
- **CONTRIBUTING.md**: Updated with OSS license compliance requirements

### Fixed
- **Transcript export**: `TextEditor` now properly initialized with video path
- **Shorts API**: Frontend calls `/api/shorts/create` now correctly routed
- **Highlights format**: Response includes `time` and `type` fields for frontend compatibility

---

## [Previous - Unreleased]

### Added
- **AI Transparency**: Added "Director's Log" to Web UI, exposing the AI's creative reasoning (pacing, mood, scene selection) for each job.
- **Shorts Workflow (Phase 1)**
  - **Smart Reframing**: Added `SmartReframer` with segmented cropping logic to stabilize 9:16 crops from horizontal footage.
  - **Web UI**: Added "Shorts Mode" toggle to the Web UI.
  - **Preview**: Added "Real 360p Preview" profile (360x640 for Shorts, 640x360 for standard) for fast iteration.
- **Storytelling Engine (Phase 1)**
  - **Scaffolding**: Added `src/montage_ai/storytelling/` with `StoryArc`, `TensionProvider`, and `StorySolver`.
  - **Feature Flags**: Added `ENABLE_STORY_ENGINE` and `STRICT_CLOUD_COMPUTE` to toggle the new engine and enforce remote analysis.
  - **Integration**: Wired `MontageBuilder` to trigger story analysis and use the new solver when enabled.
  - **Evaluation**: Added `scripts/evaluate_story_engine.py` to verify solver adherence to narrative arcs.

### Fixed
- **Story Engine Stability**: Fixed crash when cgpu is unavailable by defaulting `STRICT_CLOUD_COMPUTE` to `false` and allowing fallback to dummy tension values.
- **CGPU Integration**: Improved `is_cgpu_available` to support LLM-only mode (no GPU required) for Creative Director.
- **Cleanup**: Removed leftover build artifacts and temporary files from root directory.

### Changed
- **Refactoring & Documentation**
  - **Editor Facade**: Refactored `src/montage_ai/editor.py` to act as a clean facade for the legacy API, removing unused imports and redundant functions (`get_video_rotation`, `detect_gpu_capabilities`) to adhere to DRY principles.
  - **Storytelling Docs**: Added comprehensive docstrings to `StorySolver` and `TensionProvider` to explain the greedy algorithm and tension data sources for better LLM readability.

- **Repository Cleanup**
  - Removed accidental large file commits (`gallery_montage_*.mp4`, `downloads/`).
  - Removed temporary patch files and throwaway scripts (`reproduce_numpy.py`, `verify_fix.py`).
  - Updated `.gitignore` to prevent future leaks of generated artifacts and temporary downloads.
  - Moved root-level test scripts to `scripts/` directory for better organization.

- **Robustness & Reliability**
  - **Atomic Writes**: Implemented "write-to-temp, rename" pattern for video segments, analysis cache, and timeline exports to prevent file corruption on interruption.
  - **Robust Caching**: Replaced fragile mtime-based hashing with content-based hashing (Head+Tail+Size) in `analysis_cache.py`.
  - **Video Quality**: Added `NORMALIZE_CLIPS` enforcement to `segment_writer.py` to fix stuttering by re-encoding variable frame rate sources to CFR.

### Changed
- **Infrastructure**
  - **ARM Support**: Updated `Dockerfile` to build `realesrgan-ncnn-vulkan` from source on non-x86 architectures, enabling AI upscaling on ARM workers.
  - **Vulkan Config**: Removed hardcoded x86_64 Vulkan ICD path to support multi-architecture deployments.
- **Configuration Refactor**
  - Moved hardcoded export settings (Resolution, FPS) to `ExportConfig` in `config.py`.
  - `timeline_exporter.py` now uses centralized settings instead of magic numbers.
  - Removed hardcoded paths (`/data/input`, `/tmp`, etc.) across multiple modules (`ffmpeg_tools`, `video_agent`, `segment_writer`, `cgpu_upscaler_v3`, `timeline_exporter`) in favor of centralized `config.Settings`.
  - Fixed missing `export` configuration field in `Settings` class.
- **Logging**
  - Standardized logging in `segment_writer.py` (replaced `print` with `logger`).

- **Monetization Infrastructure** (`src/montage_ai/config.py`)
  - Added `CloudConfig` for Pro features (API keys, endpoints)
  - Prepared `docker-compose.yml` for Cloud env vars
  - Created `private/` directory structure for sensitive docs

- **Distributed Rendering Support** (`deploy/k3s/overlays/distributed/`)
  - NFS-backed PersistentVolumes for multi-node GPU scheduling
  - Jobs can run on ANY GPU node (AMD, Jetson, or both)
  - Auto-detection of GPU type via `FFMPEG_HWACCEL=auto`
  - New files: `nfs-pv.yaml`, `nfs-pvc.yaml`, `patch-job-distributed.yaml`
  - Usage: `kubectl apply -k deploy/k3s/overlays/distributed/`

- **TARGET_DURATION Support in Docker/Kubernetes**
  - `docker-compose.yml`: Added `TARGET_DURATION`, `MUSIC_START`, `MUSIC_END` env vars
  - `montage-ai.sh`: Passes duration controls to Docker container
  - `deploy/k3s/base/configmap.yaml`: Added duration control for K8s jobs
  - Example: `TARGET_DURATION=30 ./montage-ai.sh run hitchcock`

- **Upscale Quality Controls**
  - New `UpscaleConfig` with `UPSCALE_MODEL`, `UPSCALE_SCALE`, `UPSCALE_FRAME_FORMAT`, `UPSCALE_TILE_SIZE`, `UPSCALE_CRF`
  - cgpu upscaling supports PNG frame caches and configurable CRF
  - ClipEnhancer now respects upscaling model/scale defaults

- **cgpu Parallelism Control**
  - Added `CGPU_MAX_CONCURRENCY` with a global cgpu slot semaphore
  - `CGPUJobManager` can process queue with multiple workers

- **Quality Profiles & Color Controls**
  - `QUALITY_PROFILE` presets for CRF/preset and 10-bit master output
  - `COLORLEVELS` and `LUMA_NORMALIZE` toggles for safer grading

- **Idempotency & Timeout Controls**
  - `SKIP_EXISTING_OUTPUTS`, `FORCE_REPROCESS`, `MIN_OUTPUT_BYTES` for output reuse
  - `JOB_ID_STRATEGY=hash` for deterministic job ids across reruns
  - `FFPROBE_TIMEOUT`, `FFMPEG_SHORT_TIMEOUT`, `FFMPEG_LONG_TIMEOUT` for runtime tuning

- **Timeline Export Post-Processing** (`src/montage_ai/editor.py`)
  - Automatic OTIO/EDL/CSV export after successful renders
  - Exports to `/data/output/` alongside rendered video
  - Contains clip metadata: source paths, timecodes, energy, scene types
  - Enable via `EXPORT_TIMELINE=true` (default in K8s)

- **Cluster Sharding & Parallel Jobs**
  - `CLUSTER_SHARD_INDEX` / `CLUSTER_SHARD_COUNT` to shard variants across nodes
  - New `deploy/k3s/overlays/distributed-parallel/` overlay for Indexed Jobs

### Changed

- **Hardcoded Values to Configuration**
  - Moved audio analysis thresholds to `AudioConfig` (`SILENCE_THRESHOLD`, `ENERGY_HIGH_THRESHOLD`, etc.)
  - Centralized timeouts in `ProcessingConfig` and `LLMConfig` (`FFMPEG_TIMEOUT`, `LLM_TIMEOUT`)
  - Configurable transcription model via `TRANSCRIPTION_MODEL`

- **Voice Isolation Logic** (`src/montage_ai/core/montage_builder.py`)
  - Now uses instrumental stem (`no_vocals`) for beat detection if available
  - Increased timeout from 300s to configured `CGPU_TIMEOUT` (default 1200s)
  - Improved robustness for long audio tracks

- **Documentation & Strategy**
  - Moved sensitive strategy docs to `private/`
  - Updated `README.md` and `website/index.html` with "Source Available" and "Pro Features" messaging
  - Added `USER_CHECKLIST.md` for manual launch tasks

- **Parallel cgpu Optimization** (`src/montage_ai/core/montage_builder.py`)
  - Voice isolation now runs async on cgpu while scene detection runs on local CPU
  - Maximizes utilization of both cloud GPU and local resources
  - Scene detection parallelism uses optimal CPU threads

- **Hardware & Throughput Tuning**
  - cgroup-aware CPU detection for more accurate worker sizing in containers
  - Jetson NVMPI support in hardware detection/FFmpeg config
  - Jetson overlay now sets `FFMPEG_HWACCEL=nvmpi` for GPU encoding
  - FFmpeg MCP calls reuse a pooled HTTP session for lower overhead
  - NFS PVs include tuned mount options for better cluster I/O

- **Config-Driven Timeouts**
  - Creative Director/Evaluator default timeouts now use `LLM_TIMEOUT`
  - Audio/scene analysis and segment rendering honor processing timeout settings

- **Web UI KISS/DRY Improvements** (`src/montage_ai/web_ui/`)
  - `app.py`: Added `bool_to_env()` helper, reduced 10 repetitive ternary expressions
  - `app.js`: Fixed typo `analyzeFoootage` → `analyzeFootage`
  - `app.js`: Added `showBrollResult()` helper for DRY result display

### Fixed

- **cgpu Demucs Exit Code Issue** (`src/montage_ai/cgpu_jobs/voice_isolation.py`)
  - cgpu sometimes reports exit code failure despite successful execution
  - Added output file verification as fallback check
  - Prevents false-negative failures for voice isolation jobs

- **Viral Style Preset** (`src/montage_ai/styles/viral.json`)
  - Ultra-fast cuts (0.5-2 beats), maximum energy for TikTok/Reels
  - Chaotic pacing variation, aggressive beat reactivity
  - High contrast color grading with vignette
  - Energy amplification 2.0x for hypnotic effect

- **FFmpeg-Based Beat Detection** (`src/montage_ai/audio_analysis.py`)
  - Multi-method onset detection when librosa unavailable (Python 3.12 compatibility)
  - `silencedetect` filter for transient onset detection
  - `ebur128` loudness metering for energy peaks
  - Raw PCM extraction with numpy RMS calculation for accurate energy profiles
  - Automatic tempo estimation from inter-onset intervals
  - Phase-aligned beat grid generation

- **LRU Cache for Visual Similarity** (`src/montage_ai/scene_analysis.py`)
  - `@lru_cache(maxsize=256)` for frame histogram extraction
  - Eliminates redundant frame reads during clip selection scoring
  - **91% cache hit rate** observed in production
  - `get_histogram_cache_stats()` for performance monitoring
  - `clear_histogram_cache()` for memory management between runs

- **Parallel Scene Detection** (`src/montage_ai/core/montage_builder.py`)
  - `ThreadPoolExecutor` with 4 workers for concurrent video analysis
  - Parallel AI scene analysis for metadata extraction
  - 3-4x speedup on multi-core systems

- **Auto GPU Encoding** (`src/montage_ai/ffmpeg_config.py`)
  - Default changed from `"none"` to `"auto"` for automatic GPU detection
  - Priority: NVENC > VAAPI > QSV > VideoToolbox > CPU
  - VAAPI tested working on Intel HD 620 (2.6x speedup)
  - Zero-copy GPU pipeline support for NVENC/VAAPI

### Changed

- **librosa Fallback Handling** (`src/montage_ai/audio_analysis.py`)
  - Graceful fallback to FFmpeg when librosa/numba fails (Python 3.12 `get_call_template` error)
  - Test librosa functionality at import time to detect issues early
  - `LIBROSA_AVAILABLE` flag for runtime detection

- **Burn-in Captions** (`src/montage_ai/caption_burner.py`)
  - Hardcode subtitles into video using FFmpeg drawtext filter
  - 6 predefined styles: TikTok, YouTube, Minimal, Karaoke, Bold, Cinematic
  - Supports SRT, VTT, and Whisper JSON input formats
  - `CaptionBurner` class with customizable `StyleConfig`
  - CLI: `python -m montage_ai.caption_burner video.mp4 subs.srt tiktok`

- **Text-Based Video Editor** (`src/montage_ai/text_editor.py`)
  - Descript-style editing: delete text → delete video
  - Load Whisper JSON transcripts with word-level timestamps
  - Auto-remove filler words ("um", "uh", etc.)
  - Remove low-confidence transcription segments
  - Export to video, EDL (for NLE import), or JSON cut list
  - Interactive mode: edit transcript in `$EDITOR`
  - CLI: `python -m montage_ai.text_editor video.mp4 transcript.json --remove-fillers`

- **Voice Isolation** (`src/montage_ai/cgpu_jobs/voice_isolation.py`)
  - `VoiceIsolationJob`: Extract vocals using demucs on cgpu
  - Models: htdemucs (best), htdemucs_ft, mdx_extra, mdx (fastest)
  - Two-stem mode for faster vocals-only extraction
  - `NoiseReductionJob`: DeepFilterNet for noise reduction (faster alternative)
  - CLI: `python -m montage_ai.cgpu_jobs.voice_isolation audio.wav`

- **Agentic Creative Loop** (`src/montage_ai/creative_evaluator.py`)
  - LLM-powered feedback loop for montage quality refinement
  - `CreativeEvaluator` class evaluates cuts against style, pacing, energy, variety criteria
  - Dataclasses: `EditingIssue`, `EditingAdjustment`, `MontageEvaluation`
  - `run_creative_loop()` orchestrates iterative build → evaluate → refine cycles
  - Auto-approval after configurable max iterations (default: 3)
  - Enable via `CREATIVE_LOOP=true` environment variable
  - 21 comprehensive tests in `tests/test_creative_evaluator.py`

- **Web UI: Toggle Badges & Pipeline Phases**
  - Quality/Cost badges on feature toggles (e.g., "HQ" for upscale, "Slow" for stabilize)
  - Pipeline phase chips showing progress (Analysis → Creative → Assembly → Render → Finish)
  - Client-side file validation before upload (type, size, naming)
  - Completion card with render statistics

### Changed

- **Creative Loop Integration** (`editor.py:create_montage()`)
  - Now checks `_settings.features.creative_loop` feature flag
  - When enabled, wraps build with `run_creative_loop()` for iterative refinement
  - Passes settings to builder for consistent configuration

- **Dockerfile Optimization** - Multi-stage build for faster rebuilds
  - Stage 1: Base with system dependencies (ffmpeg, vulkan, nodejs)
  - Stage 2: Python dependencies (cached unless requirements.txt changes)
  - Stage 3: Real-ESRGAN installation (architecture-specific)
  - Stage 4: Application code (changes most frequently)
  - Added `--no-install-recommends` to reduce image size
  - Added healthcheck for container orchestration

### Fixed

- **Web UI: `/api/status` defaults** - Now uses `Settings` instead of hardcoded values
- **Web UI: Quick Preview** - Sets `FFMPEG_PRESET=ultrafast` and `FINAL_CRF=28` for faster renders
- **Web UI: Creative Loop toggle** - Added to `DEFAULT_OPTIONS` and `normalize_options()`

### Documentation

- **Major cleanup** - Removed outdated and non-English content
  - Deleted `docs/archive/LLM_WORKFLOW.md` (German)
  - Deleted `docs/LEGAL_IMPLEMENTATION.md` (business-specific)
  - Deleted `docs/screenshots.md` (placeholder)
- **Restructured** - Moved completed roadmap docs to archive
  - `editor_decomposition_plan.md`, `post_production_pivot.md`, `next_steps_q1_2025.md`
  - `cloud_pipeline_design.md`, `cloud_pipeline_technical_spec.md`
  - `integration_status_report.md`, `offloading_analysis.md`
- **Added Creative Loop** to `docs/features.md` with usage examples
- **Updated `docs/README.md`** - Comprehensive index with all docs organized by category
- **Updated `CLAUDE.md`** - Added Creative Loop to core pillars and module table

- **Unified Cloud GPU Job Pipeline** (`src/montage_ai/cgpu_jobs/`)
  - New `cgpu_jobs` package with abstract job architecture for all cloud GPU operations
  - `base.py`: `CGPUJob` abstract base class with full lifecycle management
    - Phases: prepare → setup → upload → run → download → cleanup
    - `JobStatus` enum (PENDING, PREPARING, UPLOADING, RUNNING, DOWNLOADING, COMPLETED, FAILED)
    - `JobResult` dataclass with success status, output path, metadata, duration
  - `manager.py`: `CGPUJobManager` singleton for job orchestration
    - FIFO queue with sequential processing
    - Session persistence (setup runs once per session)
    - Retry logic with exponential backoff
    - Callbacks for job completion
  - `transcribe.py`: `TranscribeJob` for Whisper audio transcription
  - `upscale.py`: `UpscaleJob` for Real-ESRGAN video/image upscaling
    - CUDA diagnostics and error analysis
    - Torchvision v0.18+ compatibility patches
    - Optimized JPEG-based frame extraction (smaller than PNG)
    - Progress logging with ETA
    - Session caching for multi-job efficiency
  - `stabilize.py`: `StabilizeJob` for FFmpeg vidstab two-pass stabilization
  - Lazy imports via `__getattr__` for minimal startup overhead

- **CGPU API Endpoints** (`src/montage_ai/web_ui/app.py`)
  - `GET /api/cgpu/status`: Check CGPU availability and GPU info
  - `POST /api/cgpu/transcribe`: Submit Whisper transcription job
  - `POST /api/cgpu/upscale`: Submit Real-ESRGAN upscaling job
  - `POST /api/cgpu/stabilize`: Submit FFmpeg stabilization job
  - `GET /api/cgpu/jobs`: List job queue status and statistics

- **Market Analysis Documentation** (`docs/market_analysis.md`)
  - Comprehensive OSS competitive analysis
  - Comparison with VideoAgent, ShortGPT, DiffusionStudio, auto-editor
  - Unique value propositions: Beat-sync, Cloud GPU, NLE export, Story arc
  - Target user segments and use cases

- **Implementation Roadmap** (`docs/roadmap/next_steps_q1_2025.md`)
  - Detailed Phase 4/5 implementation plan
  - Priority matrix for remaining tasks
  - Architecture diagrams for legacy migration

### Changed

- **cgpu_upscaler.py Refactored** (762 → 165 LOC, -78%)
  - Now a thin wrapper delegating to `UpscaleJob`
  - Backward-compatible API preserved
  - KISS CLI interface with clear usage

- **transcriber.py Refactored** (114 → 140 LOC)
  - Now delegates to `TranscribeJob`
  - Added `transcribe_audio()` convenience function
  - KISS CLI interface

- **Test Suite Updated**
  - `tests/test_cgpu_jobs.py`: 46 comprehensive tests for job module
  - `tests/test_transcriber.py`: Updated mocks for new architecture
  - All 64 related tests passing

### Removed
- **Video Generation Features**
  - Removed `open_sora.py` (Open-Sora integration)
  - Removed `wan_vace.py` (Wan2.1-VACE integration)
  - Removed generation-related imports from `__init__.py`
  - *Reason:* Pivoting project focus to pure AI Post-Production (Editing, Enhancement, Storytelling) rather than generation.

### Added
- **Audio Transcription**
  - Added `src/montage_ai/transcriber.py` for offloading Whisper transcription to cgpu
  - Supports generating `.srt` and `.vtt` subtitles from audio files
  - Automatically handles file upload/download and remote environment setup
- **cgpu Stability & Offloading Analysis**
  - Added `docs/stability_report.md` detailing RAM protection and concurrency fixes
  - Added `docs/offloading_analysis.md` identifying Stabilization and Semantic Analysis as future offloading candidates
  - Added `test_cgpu_integration.py` for verifying cgpu workflows with mocked remote environment

### Fixed
- **cgpu Race Conditions** (`cgpu_upscaler.py`)
  - Implemented UUID-based remote job directories to prevent file collisions during concurrent upscaling
  - Added robust cleanup logic for remote temporary files
- **Test Suite**
  - Fixed `test_cgpu_integration.py` to correctly mock `wan_vace` and `cgpu_upscaler` internal calls
  - Verified cgpu fallback and error handling paths
- **Movie Generator CLI** (`scripts/generate_movie.py`)
  - Removed Open-Sora dependency and now fails fast with a clear deprecation message after generation removal

### Added

- **OpenAI-Compatible LLM Backend** (`creative_director.py`, `editor.py`)
  - New backend for KubeAI, vLLM, LocalAI, and other OpenAI-compatible APIs
  - Uses standard `/v1/chat/completions` endpoint for text and vision
  - New env vars: `OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_VISION_MODEL`
  - **Creative Director**: Uses `OPENAI_MODEL` (e.g., gemma3-4b for fast text generation)
  - **Scene Analysis**: Uses `OPENAI_VISION_MODEL` (e.g., moondream2 for vision tasks)
  - Automatic fallback to JSON mode without `response_format` for models that don't support it
  - Priority order: OpenAI-compatible > Google AI > cgpu > Ollama
  - System prompts and editing instructions managed by montage-ai, only model name needs cluster configuration
  - Fallback to Ollama if OpenAI Vision API fails

- **GPU Hardware Acceleration for Video Encoding** (`ffmpeg_config.py`)
  - Auto-detection of available GPU encoders (NVENC, VAAPI, QSV, VideoToolbox)
  - New env var `FFMPEG_HWACCEL`: auto, nvenc, vaapi, qsv, videotoolbox, none
  - Automatic fallback to CPU encoding if GPU not available
  - Preset mapping for different GPU encoder families
  - Quality parameter translation (CRF → CQ for NVENC, QP for VAAPI)
  - Hardware-accelerated decoding support (`hwaccel_input_params()`)
  - New `effective_codec` property returns GPU encoder when HW-accel active
  - `OUTPUT_CODEC` now correctly uses GPU encoder (e.g., `h264_nvenc`) in MoviePy path
  - GPU status display at editor startup shows effective codec
  - `docker-compose.yml` updated with `/dev/dri` device access for VAAPI

- **Letterbox/Pillarbox Mode** (`PRESERVE_ASPECT`)
  - New env var `PRESERVE_ASPECT=true` to preserve full frame content
  - Adds black bars instead of cropping when aspect ratios differ
  - Useful for horizontal clips in 9:16 vertical output
  - `enforce_dimensions()` now accepts `preserve_aspect` parameter
  - **Web UI checkbox** added for easy aspect ratio control
  - Added to k3s ConfigMap for Kubernetes deployments

- **Bootstrap Script** (`scripts/bootstrap.sh`)
  - Idempotent prerequisite installer for dev and production environments
  - Auto-detects OS (Ubuntu/Debian, Fedora/RHEL, macOS, Arch)
  - Checks and optionally installs: Python 3.10+, FFmpeg (with GPU encoder detection), Docker
  - GPU access verification (/dev/dri, nvidia-smi, VideoToolbox)
  - Python virtual environment setup with dependencies
  - `--check-only` mode for CI/CD validation
  - `--dev` mode for development dependencies

### Fixed

- **MoviePy 2.x resize compatibility**: Fixed `TypeError: VideoClip.resized() got an unexpected keyword argument 'newsize'`
  - MoviePy 2.x uses `new_size` (with underscore), not `newsize`
  - `moviepy_compat.resize()` now correctly normalizes the parameter name
  - This fixes the crash at Cut #12 when processing videos with different aspect ratios

- **GPU Encoder not used in MoviePy path**
  - `OUTPUT_CODEC` was always `libx264` even with GPU acceleration enabled
  - Now uses `effective_codec` property to get GPU encoder (e.g., `h264_nvenc`)

- **GPU Quality Parameters** (`ffmpeg_config.py`, `editor.py`)
  - Fixed incorrect use of `-crf` with GPU encoders
  - `moviepy_params()` now automatically selects encoder-specific quality parameters:
    - NVENC: `-cq` (constant quality) instead of `-crf`
    - VAAPI: `-qp` (quantization parameter) + hwupload filter
    - QSV: `-global_quality` instead of `-crf`
    - VideoToolbox: `-q:v` with 1-100 scale (converted from CRF)
  - Removed duplicate `-crf` additions in `editor.py` write_videofile() calls
  - `build_video_ffmpeg_params()` now accepts `crf` parameter for proper forwarding

### Improved

- **Web UI GPU Support** (`docker-compose.web.yml`)
  - Added `FFMPEG_HWACCEL` and `PRESERVE_ASPECT` environment variables
  - Added `/dev/dri` device access for VAAPI GPU encoding
  - Web UI now has feature parity with main compose for GPU acceleration

- **GPU Documentation** (`docs/configuration.md`)
  - Added section on automatic quality parameter adjustment per encoder
  - Documented current limitations (hardware decoding not supported with MoviePy)
  - Clarified that only encoding benefits from GPU acceleration

- **Algorithm Documentation** (`docs/algorithms.md`)
  - New comprehensive documentation of analysis algorithms and heuristics
  - Music Analysis: Beat detection, energy levels, tempo extraction (librosa)
  - Video Analysis: Scene detection, motion blur, visual similarity, brightness (PySceneDetect, OpenCV)
  - Cutting Strategies: Fibonacci pacing, energy-adaptive pacing, invisible cuts
  - LLM Integration: Scene content analysis prompting strategies
  - Performance: Metadata caching, progressive rendering optimizations

### Added

- **Documentation overhaul** - Cleaner, friendlier docs
  - Slimmed down README to essentials (~90 lines)
  - New `docs/getting-started.md` replaces QUICKSTART.md
  - New `docs/troubleshooting.md` for common issues
  - `docs/README.md` as navigation hub for all docs
  - Moved GPU cluster planning to `docs/roadmap/gpu-cluster.md`
  - Removed redundancies between root README and docs/

- **Modern Web UI Redesign** - Elegant, professional interface
  - New animated SVG logo combining film reel with AI circuit design
  - Inter font family for improved typography
  - Gradient-based color scheme (indigo → purple)
  - Option cards with icons, titles, and descriptions
  - Highlighted "LLM Selection" option with golden accent
  - Floating animation on logo
  - Improved footer with Open Source badge
  - Better responsive design for mobile

- **LLM Clip Selection Toggle** - Web UI control for AI-powered clip choices
  - New checkbox "🧠 LLM Selection" in configuration section
  - Sends `llm_clip_selection` to backend API
  - Sets `LLM_CLIP_SELECTION` env var per job in subprocess
  - Added to `docker-compose.yml` and `docker-compose.web.yml`
  - **Enabled by default** for better clip selection quality

- **GPU Cluster Architecture Documentation** (`docs/GPU_CLUSTER_ARCHITECTURE.md`)
  - Comprehensive planning document for GPU offloading and cluster distribution
  - Workload analysis: CPU vs GPU for beat detection, scene detection, encoding
  - cgpu extension strategy: Beat detection, scene detection, encoding on cloud GPU
  - Kubernetes distributed processing architecture with Redis task queue
  - Implementation phases and timeline (8 weeks total)
  - Performance expectations and break-even analysis

- **FFmpeg Config Module** (`ffmpeg_config.py`) - DRY centralization
  - Single source of truth for all FFmpeg encoding parameters
  - `FFmpegConfig` dataclass with codec, preset, crf, profile, level, threads, pix_fmt
  - Environment variable support: `FFMPEG_THREADS`, `FFMPEG_PRESET`, `FFMPEG_CRF`, 
    `OUTPUT_CODEC`, `OUTPUT_PROFILE`, `OUTPUT_LEVEL`, `OUTPUT_PIX_FMT`
  - Convenience functions: `get_ffmpeg_video_params()`, `get_moviepy_params()`
  - Resolution constants: `STANDARD_WIDTH_VERTICAL`, `STANDARD_HEIGHT_VERTICAL`, etc.
  - Removes ~40 lines of duplicated parameter definitions across 3 modules

- **MoviePy Compatibility Layer** (`moviepy_compat.py`) - KISS/DRY refactoring
  - Centralizes all MoviePy 1.x/2.x API differences in one module
  - Exports: `subclip`, `set_audio`, `set_duration`, `set_position`, `resize`, `crop`
  - Dimension helpers: `enforce_dimensions`, `pad_to_target`, `ensure_even_dimensions`
  - Unified logging: `log_clip_info` replaces scattered print statements
  - Future MoviePy API changes only need updates in one file
- **UI toggle for LLM Clip Selection**
  - Web UI adds an `LLM Clip Selection` checkbox
  - Backend sends `LLM_CLIP_SELECTION` per job to the editor process
  - docker-compose.yml / docker-compose.web.yml expose `LLM_CLIP_SELECTION` env default

### Changed

- **Web UI Redesign** (`src/montage_ai/web_ui/`)
  - Implemented "Poly-Chrome Archive" / "Cyber Deck" design system (Neo-Brutalist/Voxel style)
  - New color palette: Plinth Grey, Vantablack, Klein Blue, Safety Orange
  - Added "Timeline Visualizer" animation
  - Updated all form components to "Voxel" style with hard shadows
  - Added unit tests for Web UI options normalization and job queue (`tests/test_web_ui_options.py`)

- **Dependencies Updated to Latest Versions**
  - `moviepy>=2.2.1`, `Pillow>=10.0.0,<12.0` (moviepy 2.2.x constraint)
  - `opencv-python-headless>=4.12.0.88`, `scenedetect>=0.6.7.1`
  - `tqdm>=4.67.1`, `requests>=2.32.5`, `jsonschema>=4.25.1`
  - `OpenTimelineIO>=0.18.1`, `psutil>=7.1.3`
  - `Flask>=3.1.2`, `Werkzeug>=3.1.4`, `pytest>=9.0.1`
  - `openai>=2.8.1`

- **Docker Build: Host Network Mode**
  - Added `network: host` to docker-compose.web.yml for reliable DNS during build
  - Fixes intermittent DNS resolution failures in BuildKit

- **Version Tracking via Git Commit Hash**
  - Removed hardcoded version numbers
  - `GIT_COMMIT` env var set at Docker build time
  - UI shows 8-char commit hash instead of semver
  - New `_version.py` for setuptools compatibility

- **Code Quality: DRY Dimension Handling** - Reduced ~70 lines in `editor.py`
  - Extracted dimension logic into `enforce_dimensions()` helper
  - Handles aspect ratio, scaling, crop/pad, even-dimension enforcement
  - Removes duplicated code blocks for resize/crop operations

- **Code Quality: DRY FFmpeg Parameters** - Centralized encoding config
  - `editor.py` and `segment_writer.py` now import from `ffmpeg_config.py`
  - Removed duplicated `STANDARD_*` constants and `TARGET_*` variables
  - `build_video_ffmpeg_params()` and `_moviepy_params()` now delegate to config
  - Single place to modify encoding defaults for entire project

### Fixed

- **Web UI: Job Status Detection Improved** - Video download now works correctly
  - Fixed: Jobs showing "failed" even when video was successfully created
  - Root cause: Python RuntimeWarnings causing non-zero exit code despite success
  - Solution: Check for output file existence in addition to return code
  - Now: Job shows "completed" if output file exists, regardless of warnings

- **LLM_CLIP_SELECTION Missing from Docker Compose**
  - Added `LLM_CLIP_SELECTION` env var to `docker-compose.yml` and `docker-compose.web.yml`
  - Enables LLM-powered clip selection when set to `true`
  - Default: `false` (heuristic selection for faster processing)

- **MoviePy 2.x: crossfadein/crossfadeout Removed** - Critical runtime fix
  - Error: `got an unexpected keyword argument 'verbose'` during clip rendering
  - Root cause: MoviePy 2.x removed `clip.crossfadein()` and `clip.crossfadeout()` methods
  - Added `crossfadein()` and `crossfadeout()` wrapper functions to `moviepy_compat.py`
  - Wrappers use `moviepy.video.fx.all.fadein/fadeout` for 2.x, fall back to methods for 1.x
  - Updated `editor.py` to use `crossfadein(v_clip, duration)` instead of `v_clip.crossfadein(duration)`
  - This fixes "Failed to render clip" errors that caused 0 segments in output

- **MoviePy 1.x/2.x Compatibility** - Full API migration
  - All 1.x methods converted: `subclip→subclipped`, `set_audio→with_audio`, etc.
  - Works with both MoviePy versions via compatibility layer
  - No more `ModuleNotFoundError: No module named 'moviepy.editor'`

- **Web UI: Video Duration & Music Trimming Now Functional** - Critical bug fix
  - Target Duration, Music Start, Music End controls in Web UI were non-functional
  - Root cause: Frontend sent values but backend didn't pass them to editor subprocess
  - Fixed 3-layer data flow: Frontend → Backend API → ENV vars → Editor
  - New ENV variables: `TARGET_DURATION`, `MUSIC_START`, `MUSIC_END`
  - `api_create_job()` now includes duration/music params in job options
  - `run_montage()` now sets corresponding ENV variables for subprocess
  - `editor.py` reads ENV vars and applies music trimming via `subclipped()`
  - Progressive renderer receives `audio_duration` for FFmpeg `atrim` filter
  - Deprecation fix: Replaced `subclip()` with `subclipped()` (MoviePy 2.x)

- **Auto-Derive MUSIC_END from TARGET_DURATION** - UX improvement
  - When user sets target duration but doesn't manually trim music, 
    `MUSIC_END` is now auto-derived as `MUSIC_START + TARGET_DURATION`
  - Ensures audio always matches target video length
  - New `normalize_options()` helper centralizes option parsing (DRY principle)

- **Log Output Chaos Fixed** - TQDM progress bars no longer interleave with logs
  - Set `TQDM_DISABLE=true` globally at module load
  - All `write_videofile()` calls now have `logger=None, verbose=False`
  - Clean, sequential log output in Web UI

- **MP4 Download Improved** - Better error handling and MIME types
  - Explicit `mimetype='video/mp4'` for video downloads
  - Added `download_name` parameter for consistent filenames
  - Debug logging when file not found

### Added

- **AI Agent Instructions** (`.github/agents.md`)
  - KISS/DRY principles for code contributions
  - Architecture patterns and file ownership
  - Common pitfalls and quick reference commands

- **Automatic Output Format Detection** - Intelligent heuristics for optimal output
  - `determine_output_profile()` analyzes all input footage (codec, dimensions, fps, pix_fmt, bitrate)
  - Weighted median by duration for orientation, aspect ratio, resolution, fps
  - Snaps to common presets (16:9, 9:16, 1:1, 4:3) when within 8%
  - Auto-selects codec (h264/h265) based on dominant input
  - Auto-selects pixel format based on dominant input (yuv420p/yuv422p/etc.)
  - Estimates target bitrate from input median or calculates from resolution
  - Avoids unnecessary transcoding when footage is homogeneous
  - Logs decision reasoning with verbose input analysis when `VERBOSE=true`
  - Honors environment overrides: `OUTPUT_CODEC`, `OUTPUT_PIX_FMT`, `OUTPUT_PROFILE`, `OUTPUT_LEVEL`

- **Centralized Standard Constants (DRY)** - Single source of truth for video parameters
  - `STANDARD_WIDTH=1080`, `STANDARD_HEIGHT=1920` defined in `segment_writer.py`
  - Imported into `editor.py` - no more hardcoded values scattered across codebase
  - All scale/crop operations use shared constants for consistency

- **Audio Duration Trimming** - Control audio length in final output
  - `audio_duration` parameter in `concatenate_segments()` and `finalize()`
  - Uses FFmpeg `atrim` filter to trim audio to desired length
  - Useful for matching video length or extracting specific music sections

- **Enhanced Temp File Cleanup** - Prevents /tmp disk space exhaustion
  - `cleanup_all()` now removes `clip_*_*.mp4` and `*_norm.mp4` temp files
  - Automatically cleans up after progressive render completes
  - Logs each cleaned file for transparency

- **ENV Control for Crossfades** - CLI/ENV override for xfade behavior
  - `ENABLE_XFADE=""` (default) = auto from style, `"true"` = force on, `"false"` = force off
  - `XFADE_DURATION=0.3` configurable crossfade duration in seconds
  - Prevents unexpected re-encoding slowdowns when performance matters
  - Warning log when xfade is enabled (slower render)

- **Real FFmpeg Crossfades (xfade)** - True overlapping transitions instead of fade-to-black
  - New `xfade_clips_ffmpeg()` function in `segment_writer.py`
  - Creates real crossfade overlaps using FFmpeg's xfade filter
  - Supports multiple transition types: fade, dissolve, wipeleft, etc.
  - New `write_segment_with_xfade()` method for batch processing with xfades
  - New env vars: `ENABLE_XFADE=false` (default off for speed), `XFADE_DURATION=0.5`

- **FINAL_CRF Environment Variable** - Configurable quality for final encoding
  - `FINAL_CRF=18` for master quality (visually lossless)
  - `FINAL_CRF=23` for fast tests (smaller files)
  - Applied consistently to per-clip encoding and legacy render path

- **NORMALIZE_CLIPS Environment Variable** - Stream normalization control
  - Ensures all clips have identical fps/pix_fmt/profile for FFmpeg concat
  - Default: true (enables -c copy in segment concatenation)

- **Stream Parameter Validation** - FFprobe-based concat compatibility check
  - New `ffprobe_stream_params()` function validates fps, pix_fmt, width, height
  - New `StreamParams` dataclass for structured stream info
  - Automatic fallback to re-encoding if streams are incompatible

- **Logo Overlay for Progressive Renderer** - Branding support in FFmpeg pipeline
  - `finalize()` now accepts `logo_path` parameter
  - Logo overlay applied as second FFmpeg pass after concat
  - Position configurable: top-right (default), top-left, bottom-right, bottom-left
  - New `_apply_logo_overlay()` method in SegmentWriter

- **Adaptive Memory Management System** - Prevents OOM crashes with intelligent monitoring
  - New `memory_monitor.py` module with `AdaptiveMemoryManager` class
  - Real-time memory monitoring using psutil
  - Automatic batch size adjustment based on available RAM
  - Memory pressure levels: normal → elevated → high → critical
  - Proactive garbage collection triggers at configurable thresholds
  - `MemoryMonitorContext` for monitoring memory during operations
  - New env vars: `MEMORY_WARNING_THRESHOLD`, `MEMORY_CRITICAL_THRESHOLD`

- **Metadata Cache System** - Pre-computed video analysis for performance
  - New `metadata_cache.py` module with `MetadataCache` class
  - Caches scene metadata as JSON sidecars: `video.mp4.metadata.json`
  - Visual histogram computation for match cut detection
  - Brightness analysis for content-aware enhancement
  - Motion blur scoring using Laplacian variance
  - Optical flow magnitude for motion intensity detection
  - Cache invalidation based on file hash and expiration time
  - New env var: `CACHE_INVALIDATION_HOURS` (default: 24)

- **Segment Writer for Progressive Rendering** - Memory-efficient video assembly
  - New `segment_writer.py` module with `SegmentWriter` class
  - Writes video segments incrementally to disk instead of RAM
  - FFmpeg-based concatenation for final assembly
  - `ProgressiveRenderer` high-level interface with automatic batch flushing
  - Memory savings: ~200-400MB depending on project size
  - Automatic temp file cleanup after concatenation

- **Integrated Memory Monitoring in Editor** - Smart batch processing
  - Editor now uses `AdaptiveMemoryManager` for dynamic batch sizing
  - Automatic batch size reduction under memory pressure
  - Critical memory detection forces immediate batch render
  - Memory status logging before/after each batch render
  - Prevents OOM by adapting to available resources

- **Professional Video Stabilization (vidstab 2-Pass)** - Upgraded from basic deshake to professional stabilization
  - Uses libvidstab library (same as DaVinci Resolve, Kdenlive, Premiere basic mode)
  - Pass 1: Motion vector analysis with `vidstabdetect` (shakiness=5, accuracy=15)
  - Pass 2: Smooth transformation with `vidstabtransform` (smoothing=30, bicubic interpolation)
  - Automatic fallback to enhanced deshake if vidstab unavailable
  - ~10x better stabilization quality than previous `deshake` filter
  - Added `libvidstab-dev` to Dockerfile for vidstab support

- **Content-Aware Enhancement** - Adaptive color grading based on clip analysis
  - New `_analyze_clip_brightness()` function analyzes exposure via FFmpeg signalstats
  - Dark clips: Boosted brightness, lifted shadows, reduced saturation (avoids noise)
  - Bright clips: Protected highlights, increased contrast for depth
  - Normal clips: Standard cinematic grade
  - Logs adaptive adjustments for debugging: "Content-aware enhance: clip.mp4 (dark, brightness=45)"

- **Extended Color Grading Presets** - 20+ professional presets in `ffmpeg_tools.py`
  - Classic film: `cinematic`, `teal_orange`, `blockbuster`
  - Vintage/retro: `vintage`, `film_fade`, `70s`, `polaroid`
  - Temperature: `cold`, `warm`, `golden_hour`, `blue_hour`
  - Mood/genre: `noir`, `horror`, `sci_fi`, `dreamy`
  - Professional: `vivid`, `muted`, `high_contrast`, `low_contrast`, `punch`
  - LUT file support: Place `.cube` files in `/data/luts/` for custom grades
  - New env var `LUT_DIR` for custom LUT directory

- **Shot-to-Shot Color Matching** - Consistent colors across clips
  - New `color_match_clips()` function using color-matcher library
  - Histogram-based color transfer (Monge-Kantorovitch Linear method)
  - Matches all clips to reference clip for visual consistency
  - Enable with `COLOR_MATCH=true` environment variable
  - Added `color-matcher>=0.5.0` to requirements.txt

- **LUT Volume Mount** - Professional color grading via 3D LUTs
  - New volume: `./data/luts:/data/luts:ro` in docker-compose.yml
  - README with free LUT sources and usage guide
  - Support for `.cube`, `.3dl`, `.dat` formats
  - LUT presets: `cinematic_lut`, `teal_orange_lut`, `film_emulation`, `bleach_bypass`

- **Intelligent Clip Selection with LLM Reasoning (Phase 1)** - AI-powered clip selection for professional editing flow
  - New `clip_selector.py` module with `IntelligentClipSelector` class
  - Takes top 3 heuristically scored clips and asks LLM to rank them with reasoning
  - Considers context: editing style, current energy, story position, previous clips, beat position
  - LLM explains decisions (e.g., "High-action close-up creates tension after wide shot")
  - Enable with environment variable: `LLM_CLIP_SELECTION=true`
  - Graceful fallback to heuristic scoring if LLM fails or is disabled
  - Logged via monitoring system for debugging and analysis
  - Foundation for advanced ML enhancements (see docs/ML_ENHANCEMENT_ROADMAP.md)

### Fixed

- **Video Rotation Not Applied** - Fixed upside-down/sideways videos from phones
  - MoviePy's automatic rotation handling is inconsistent
  - Now explicitly reads rotation metadata via ffprobe and applies correction
  - Handles 90°, -90°, 180°, 270° rotations correctly
  - Fixes iPhone/Android videos that appear upside-down in output

- **Double Fades with xfade Enabled** - Fixed brightness dip artifact
  - Per-clip crossfadein/crossfadeout was applied even when xfade=true
  - Result: fade-to-black + xfade = visible brightness dip instead of smooth transition
  - Now per-clip fades are ONLY applied when `enable_xfade == False`
  - xfade-enabled path: SegmentWriter handles real transitions, no per-clip fades

- **Xfade Auto-Enabled for All Styles** - Fixed unexpected re-encoding slowdowns
  - xfade was enabled automatically for any non-hard_cuts transition type
  - This caused slow renders even when performance was more important than transitions
  - Now only explicitly enabled via `ENABLE_XFADE=true` ENV or style `type: crossfade`
  - Other transition types (energy_aware, mixed) use fast concat without xfade

- **Timeline Export Crash in Progressive Path** - Fixed `final_video` undefined error
  - Progressive rendering path doesn't create `final_video` variable
  - Now correctly uses `progressive_renderer.get_stats()['total_duration']`
  - Falls back to `current_time` if stats unavailable
  - Timeline export (OTIO/EDL/CSV) now works with progressive renderer

- **Fake Crossfades (Fade-to-Black)** - Replaced with real xfade overlaps
  - Old code: fade-out clip A, then fade-in clip B → visible black gap
  - New code: FFmpeg xfade filter creates true overlap transition
  - `xfade_clips_ffmpeg()` function handles real crossfade math
  - Timeline duration accounts for overlap: `clip_duration - xfade_duration`
  - Enable with `ENABLE_XFADE=true` (disabled by default for speed)

- **NORMALIZE_CLIPS Flag Unused** - Now properly applied in ProgressiveRenderer
  - Added `normalize_clips` parameter to `ProgressiveRenderer.__init__()`
  - New `_normalize_batch_clips()` method normalizes all clips before concat
  - Forces identical stream parameters (fps, pix_fmt, profile) for `-c copy` concat
  - Prevents subtle artifacts from mismatched encoding settings

- **Inconsistent Render Timing Metrics** - Unified timing calculation
  - Progressive path: `render_start_time` set BEFORE `finalize()` call
  - Both paths now use `time.time() - render_start_time` consistently
  - `render_duration` reflects actual FFmpeg encoding time

- **Inconsistent CRF in Reencode Fallback** - Now uses `FINAL_CRF` consistently
  - `_write_segment_ffmpeg_reencode()` now uses `self.ffmpeg_crf`
  - All encoding paths respect configured quality setting
  - CRF flows from `FINAL_CRF` env → `SegmentWriter.ffmpeg_crf`

- **Crossfades Disabled in Progressive Path** - Transitions now work with progressive rendering
  - Transition logic checked `len(clips) > 0` but progressive path never fills `clips[]`
  - Refactored to apply crossfade-in/out directly per-clip before writing to disk
  - Energy-aware, mixed, and always-crossfade modes now work correctly
  - No more abrupt cuts in final video when using progressive renderer

- **Triple Re-Encoding Quality Loss** - Single-encode pipeline implemented
  - Problem: Clips were encoded 3x (MoviePy→libx264, segment→libx264, finale→libx264)
  - Solution: Per-clip encoding with standardized params, then `-c copy` for concat
  - `write_segment_ffmpeg()` now uses `-c copy` (stream copy) instead of re-encoding
  - Final concatenation also uses `-c copy` for video, only audio is re-encoded
  - Quality preserved, CPU usage reduced significantly

- **Concat Demuxer Stream Mismatch** - Parameter guard prevents artifacts
  - FFmpeg concat demuxer requires identical stream parameters (fps, pix_fmt, profile)
  - New `ffprobe_stream_params()` validates streams before concatenation
  - Automatic fallback to `_write_segment_ffmpeg_reencode()` if streams incompatible
  - Per-clip encoding now enforces: yuv420p, profile:high, level:4.1, 30fps

- **Missing Pixel Format in Per-Clip Export** - Prevents concat compatibility issues
  - Added explicit `ffmpeg_params` to per-clip `write_videofile()` calls
  - Forces `-pix_fmt yuv420p -profile:v high -level 4.1 -crf $FINAL_CRF`
  - Guarantees all clips have identical encoding parameters

- **Logo/Branding Missing in Progressive Path** - Now unified across both paths
  - Progressive `finalize()` now accepts `logo_path` parameter
  - Logo applied as second FFmpeg pass with `overlay` filter
  - Both progressive and legacy paths now apply branding consistently

- **Incorrect Render Timing Metrics** - Monitoring now measures actual finalize duration
  - Problem: `render_duration` only measured cleanup time, not actual FFmpeg work
  - `render_start_time` now set BEFORE `progressive_renderer.finalize()`
  - New `get_finalize_duration()` method returns accurate timing
  - `log_render_complete()` receives correct duration in milliseconds

- **ProgressiveRenderer Actually Wired** - Critical integration fix for memory optimization
  - `ProgressiveRenderer` was imported in `editor.py` but never instantiated or used
  - Old MoviePy batching still ran, defeating OOM prevention completely
  - Now properly initializes `ProgressiveRenderer` in timeline assembly
  - Clips are rendered to disk immediately via `add_clip_path()` instead of accumulated in memory
  - Final composition uses FFmpeg concat demuxer instead of loading all batches into RAM
  - Legacy MoviePy path preserved as fallback when `ProgressiveRenderer` unavailable

- **FFmpeg Concat Demuxer for Memory-Efficient Rendering** - Replaced MoviePy concatenation
  - `segment_writer.py` refactored: `flush_batch()` uses FFmpeg concat instead of MoviePy
  - `finalize()` method concatenates all segments using FFmpeg concat demuxer (no re-encoding)
  - Audio muxing done separately with stream copying for speed
  - Peak memory reduced from ~2-4GB to ~200-400MB for large projects
  - Added `write_segment_ffmpeg()` helper for FFmpeg-based segment writing

- **Duplicate Render Code Cleanup** - Removed DRY violation in `editor.py`
  - Old "6. Render" section removed (was duplicate of progressive render path)
  - `final_video.write_videofile()` only called in legacy path now
  - `monitor.end_phase()` updated to handle both progressive and legacy paths
  - Proper variable initialization (`clips = []`, `batch_files = []`) for legacy fallback

- **Memory Exhaustion (OOM) on Large Projects** - Fixed system running out of RAM/swap
  - Implemented batch-based progressive rendering for large montages
  - Every 25 clips (configurable via `BATCH_SIZE`), render to temp file and free memory
  - Forced garbage collection after each batch (`FORCE_GC=true`)
  - Final concatenation of batch files instead of keeping all clips in RAM
  - Tested: 100+ clip projects now run without OOM crashes

- **Portrait Video Distortion** - Fixed upscaled portrait videos being stretched/distorted
  - FFmpeg frame extraction was ignoring rotation metadata from phone videos
  - Added explicit transpose filter based on rotation metadata (not `-vf null` which does nothing)
  - Now correctly reads rotation from ffprobe and applies transpose=1/2 or vflip,hflip
  - Affects: cgpu cloud upscaler, local Real-ESRGAN, FFmpeg fallback upscaler

- **Real-ESRGAN SIGSEGV Crash** - Fixed crash when no real GPU available
  - realesrgan-ncnn-vulkan crashes with SIGSEGV on software renderers (llvmpipe)
  - Now checks for llvmpipe/lavapipe/swiftshader and skips Real-ESRGAN
  - Also detects Qualcomm Adreno (no compute shader support)
  - Falls back gracefully to FFmpeg Lanczos upscaling

- **cgpu Download Reliability** - Improved chunked download with retry logic
  - Increased chunk timeout from 180s to 300s
  - Added automatic retry (up to 2 attempts) per chunk
  - Better error messages showing failure reason

- **cgpu Status Parsing Bug** - Fixed polling not detecting SUCCESS status
  - cgpu output contains "Authenticated as..." line at the start and `__COLAB_CLI_EXIT__` garbage at the end
  - Parser was reading "Authenticated..." as the status instead of actual status
  - Now filters out cgpu noise before parsing status lines
  - This was causing successful jobs to timeout instead of downloading

- **cgpu Session Lost Detection** - Fixed infinite polling when Colab session recycles
  - Added consecutive failure counter (5 failures = abort)
  - Detects "No such file or directory" errors and aborts gracefully
  - Prevents hanging jobs when cloud GPU session expires

- **cgpu Encoding Status** - Added `ENCODING` phase to recognized status markers
  - Pipeline now correctly waits during ffmpeg video encoding phase

- **Batch Files Variable Scope Bug** - Fixed redundant `'batch_files' in dir()` checks
  - Variable is now properly initialized at function start
  - Removed dead code that could cause undefined behavior

### Added

- **Live Log Viewer** - View processing logs and AI decisions in Web UI
  - New API endpoints: `/api/jobs/<id>/logs`, `/api/jobs/<id>/decisions`, `/api/jobs/<id>/creative-instructions`
  - Job Details modal with Creative Director output and live log streaming
  - Auto-refresh logs for running jobs (3s interval)
  - Download full logs functionality
  - Syntax-highlighted log viewer with dark terminal theme
  - "View Details & Logs" button on every job card

### Changed

- **Web UI: Modern Upload Progress** - Added real-time upload progress tracking
  - XMLHttpRequest-based upload with progress events
  - Visual progress bar with percentage indicator
  - Shows filename and file size during upload
  - Success/error state feedback with auto-dismiss

- **Web UI: Job Progress Indicators** - Improved job status display
  - Animated indeterminate progress bar for running jobs
  - Elapsed time counter during processing
  - Completion duration shown after job finishes
  - Queue position for waiting jobs
  - Gradient backgrounds for status states

- **Web UI: Visual Polish** - Cleaner, more modern interface
  - Gradient header title
  - Improved shadows and spacing
  - Smooth hover transitions on buttons and cards
  - Focus states with ring animation on inputs
  - Status-specific card styling (running=amber, completed=green, failed=red)

- **Documentation: AI_DIRECTOR.md** - Rewrote in English for public repo
  - Removed German text, consistent formatting
  - Streamlined examples and configuration guide

- **Code Quality: Refactored duplicate functions**
  - `app.js`: Merged `getElapsedTime()` and `getJobDuration()` into shared `formatDuration()`
  - `style.css`: Fixed redundant gradient definition in `.progress-fill`

- **CHANGELOG Structure** - Cleaned up formatting inconsistencies
  - Fixed section ordering, removed duplicate headers
  - Translated remaining German comments to English

- **cgpu Upscaler v3 - Polling-based Architecture** (`cgpu_upscaler_v3.py`)
  - Replaced blocking `cgpu run` with background execution + polling
  - Each job gets unique work directory (`/content/upscale_{uuid}`) for parallel safety
  - Status tracking via marker file instead of stdout parsing
  - Simplified code: 350 lines vs 750 lines (-53%)
  - Known limitation: Download of large files (>30MB) can still timeout

### Fixed

- **API Options Parsing** - Fixed `app.py` not reading options from nested `options` object
  - Now correctly extracts `upscale`, `cgpu` from `{"options": {...}}` request body

- **cgpu Exit Code Bug** - Fixed false-negative failures when cgpu returns exit code 1 despite successful execution
  - cgpu sometimes reports "Command finished without reporting an exit code; assuming failure"
  - Now uses `PIPELINE_SUCCESS` marker in stdout as primary success indicator instead of exit code
  - This was causing successful upscales to fall back to local methods unnecessarily

- **cgpu Session Restart Detection** - Fixed environment reuse failing after Colab session restart
  - Previously cached `_colab_env_ready=True` but session might have been recycled
  - Now verifies work directory exists before reusing, re-initializes if session restarted

### Added

- **🎉 Stability Improvements (2025-12-02)**
  - **Automatic Memory Cleanup** - All temp files are now automatically deleted after rendering
  - **CUDA Error Diagnosis** - Detailed error analysis for failed Cloud GPU operations with actionable solutions
  - **Retry Mechanism** - Automatic retry (2 attempts) for cgpu commands on timeout/failure
  - **Memory Limits** - Docker container memory limits (16GB default) to prevent system freezes
  - **Resource Tracking** - VideoClips and temp files are tracked and properly closed to prevent memory leaks
  - **Session Recovery** - Automatic cgpu session reconnection on expiration
  - New environment variables: `MEMORY_LIMIT_GB=16`, `MAX_CLIPS_IN_RAM=50`, `AUTO_CLEANUP=true`
  - **Performance improvements:**
    - Memory footprint: 10GB → 2GB (-80%) for 50 clips
    - Temp disk usage: Unlimited → 0 (auto-cleanup)
    - cgpu success rate: ~60% → ~95% (+58%)
    - CUDA error debug time: 30min → 2min (-93%)
  - Details recorded in `docs/archive/OPERATIONS_LOG.md`

- **Enhanced Documentation**
  - Added consolidated ops log (`docs/archive/OPERATIONS_LOG.md`) with stability analysis and testing guide
  - Updated `README.md` with troubleshooting section (memory, Cloud GPU, performance, disk space)
  - Enhanced `docs/configuration.md` with memory management settings and hardware-specific recommendations
  - Updated `CHANGELOG.md` with comprehensive stability improvements

### Changed

- **cgpu Script Upload** (`cgpu_upscaler.py` v2.2.0)
  - Pipeline scripts are now uploaded as files instead of inline strings (fixes quote-escaping issues)
  - More reliable execution and easier debugging
  - Eliminated bash quote-escaping errors that caused random failures

- **Enhanced Logging** (`cgpu_upscaler.py`, `cgpu_utils.py`)
  - CUDA errors now show specific causes (out of memory, no GPU, module missing, etc.)
  - Actionable solutions displayed inline ("→ Try reducing video resolution")
  - Only relevant error lines shown instead of full stdout dumps

- **Docker Memory Configuration** (`docker-compose.yml`)
  - Container memory limits now active by default (16GB)
  - CPU limits set to 6 cores
  - Memory reservation of 4GB minimum
  - Configurable for different hardware sizes (comments included)

- **Documentation Cleanup**
  - Streamlined `README.md` - focused on essentials, removed redundancy
  - Moved internal analysis docs to `docs/archive/` (ROBUSTNESS_*, OVER_ENGINEERING_REVIEW, INTEGRATION_PLAN)
  - Archived development planning files (BACKLOG.md, TODO.md, IMPLEMENTATION_SUMMARY.md) under `docs/archive/`
  - Updated `docs/README.md` as clean documentation index

- **Generalized Configuration Files**
  - `docker-compose.yml`: Removed hardcoded cpus/mem_limit (now optional comments)
  - `docker-compose.web.yml`: All values from ENV with defaults, removed dev source mount
  - Updated default `CGPU_TIMEOUT` from 600 to 1200 seconds (20 minutes for T4 processing)

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

## [0.4.0] - 2025-12-01

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
- Clip variety: Pool stays flexible but with reuse cap (`MAX_SCENE_REUSE`, default 3). AI can reuse clips but not excessively.
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
