# Montage AI Quick Start & Status Guide

## Quick Commands

### Check Job Status
```bash
./montage-status.sh status          # Show current job + output
./montage-status.sh logs            # Show last 50 log lines
./montage-status.sh logs 200        # Show last 200 log lines
```

### Run Montage
```bash
# Simple run (default settings)
./montage-ai.sh run

# With custom style
export CUT_STYLE=hitchcock
./montage-ai.sh run

# Full speed, maximum parallelism
export MAX_PARALLEL_JOBS=16
export MAX_CONCURRENT_JOBS=16
export QUALITY_PROFILE=preview
./montage-ai.sh run

# High-quality output (slower)
export QUALITY_PROFILE=standard
export FFMPEG_PRESET=slow
./montage-ai.sh run
```

### Manage Jobs
```bash
./montage-status.sh stop            # Stop running job
./montage-status.sh cleanup         # Clean temp files
./montage-status.sh restart         # Stop + cleanup + ready for new run
```

## Environment Variables

### Performance
- `MAX_PARALLEL_JOBS` - Number of parallel processors (default: auto-detect)
- `MAX_CONCURRENT_JOBS` - Concurrent jobs (default: auto-detect)
- `FFMPEG_PRESET` - FFmpeg speed (ultrafast|superfast|veryfast|faster|fast|medium|slow|slower|veryslow)
- `FFMPEG_THREADS` - Threads per FFmpeg process (0=auto)

### Quality
- `QUALITY_PROFILE` - preview|standard|high (default: standard)
- `STABILIZE` - Enable video stabilization (true|false)
- `UPSCALE` - Enable AI upscaling (true|false)

### Style
- `CUT_STYLE` - dynamic|hitchcock|mtv|action|documentary|minimalist|wes_anderson
- `CREATIVE_PROMPT` - Natural language editing instructions

### Paths
- `INPUT_DIR` - Source footage location (default: /data/input/)
- `OUTPUT_DIR` - Final video output (default: /data/output/)
- `MUSIC_DIR` - Background music (default: /data/music/)

## Typical Workflows

### Fast Preview Run
```bash
export QUALITY_PROFILE=preview
export CUT_STYLE=dynamic
export MAX_PARALLEL_JOBS=16
./montage-ai.sh run

# Check status every 15 seconds:
watch -n 15 './montage-status.sh status'
```

### High-Quality Single Output
```bash
export QUALITY_PROFILE=high
export CUT_STYLE=hitchcock
export FFMPEG_PRESET=slow
./montage-ai.sh run

# Monitor logs in another terminal:
tail -f /tmp/montage_run.log | grep -E "📊|✓|Phase"
```

### Batch Processing (multiple styles)
```bash
for style in dynamic action documentary; do
    echo "Processing: $style"
    export CUT_STYLE=$style
    ./montage-ai.sh run
    sleep 60  # Wait for job to start
    ./montage-status.sh logs  # Check progress
done
```

## Output Files

Videos appear in `/data/output/` once rendering completes:
```bash
ls -lh /data/output/*.mp4

# Check output quality
ffprobe /data/output/montage_*.mp4
```

## Troubleshooting

### Job not starting
```bash
./montage-status.sh cleanup    # Remove stale temp files
./montage-status.sh logs 50    # Check error messages
./montage-ai.sh run            # Try again
```

### Job stuck or slow
```bash
# Check CPU/memory usage in logs
tail -100 /tmp/montage_run.log | grep "📊"

# Stop and restart with fewer jobs
./montage-status.sh stop
export MAX_PARALLEL_JOBS=4
./montage-ai.sh run
```

### No output files
```bash
# Check if job completed
./montage-status.sh status

# Increase timeout and allow more time
export RENDER_TIMEOUT=7200
./montage-ai.sh run
```

## Resource Requirements

**Minimum:**
- 2 CPU cores
- 4 GB RAM
- 10 GB free disk space

**Recommended (for cluster):**
- 8+ CPU cores
- 16 GB RAM
- 50+ GB free disk space

**For 4K files:**
- 16+ CPU cores
- 32 GB RAM
- 100+ GB free disk space

## Architecture

```
Your Video Files
       ↓
   Scene Detection (parallel)
       ↓
   Audio Analysis (beat sync)
       ↓
   Intelligent Clip Selection (ML)
       ↓
   Timeline Assembly
       ↓
   FFmpeg Rendering
       ↓
Final Montage 🎬
```

## Monitoring Resource Usage

The logs show CPU/Memory every 5 seconds during long phases:
```
📊 Scene Detection [33%]: CPU process=12.5% sys=45.2% | Memory process=345MB sys=72.3%
```

This helps you see:
- **Progress** - percentage complete
- **Process CPU** - how much this job uses
- **System CPU** - cluster load
- **Memory** - RAM consumption

Use this to tune `MAX_PARALLEL_JOBS` and `MAX_CONCURRENT_JOBS` for your hardware.
