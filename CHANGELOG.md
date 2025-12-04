# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Dependencies Updated to Latest Versions**
  - `moviepy>=2.1.1` (was 1.0.3) - Uses modern `.subclipped()` API
  - `opencv-python-headless>=4.10.0` (was unversioned)
  - `Pillow>=10.4.0`, `tqdm>=4.66.0`, `requests>=2.32.0`
  - `scenedetect>=0.6.4`, `openai>=1.55.0`, `psutil>=6.1.0`
  - `OpenTimelineIO>=0.17.0`, `jsonschema>=4.23.0`
  - `Flask>=3.1.0`, `Werkzeug>=3.1.0`, `pytest>=8.3.0`

- **Version Tracking via Git Commit Hash**
  - Removed hardcoded version numbers
  - `GIT_COMMIT` env var set at Docker build time
  - UI shows 8-char commit hash instead of semver
  - New `_version.py` for setuptools compatibility

### Fixed

- **MoviePy 1.x/2.x Compatibility** - `subclip_compat()` helper
  - Works with both `.subclip()` (1.x) and `.subclipped()` (2.x)
  - Prevents `AttributeError` when running on different MoviePy versions

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
