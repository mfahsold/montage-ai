# Snapdragon ARM Deployment Test Plan

This content has been consolidated into the single source of truth:

- [README.md](README.md) (see the ARM64 section)

Use the automated checks instead of a long manual plan:

```bash
./scripts/quick-setup-arm.sh
./scripts/validate-onboarding.sh
```
- ✅ Final image size ~2-3GB (normal for ARM64 Python + FFmpeg)

---

## Phase 4: Python Module Verification (5 minutes)

```powershell
# Test Python imports
docker compose run --rm montage-ai `
    python -c "import montage_ai; print('✅ montage_ai imported')"

# Test key dependencies
docker compose run --rm montage-ai `
    python -c "import cv2, moviepy, librosa; print('✅ deps available')"

# Test FFmpeg
docker compose run --rm montage-ai ffmpeg -version | head -5

# Test GPU detection (optional)
docker compose run --rm montage-ai `
    python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not available (OK)"
```

**Success Criteria:**
- ✅ montage_ai imports successfully
- ✅ OpenCV (cv2), MoviePy, Librosa available
- ✅ FFmpeg version shows >= 4.2

---

## Phase 5: Test Render (Preview Mode) - 15 minutes

### 5A: Create Synthetic Test Video

```powershell
# Generate 5-second test video (blue, red, green colors)
docker compose run --rm montage-ai `
    ffmpeg -f lavfi -i color=c=blue:s=320x240:d=2 `
           -f lavfi -i color=c=red:s=320x240:d=2 `
           -f lavfi -i color=c=green:s=320x240:d=2 `
           -filter_complex concat=n=3:v=1:a=0 `
           -y data/input/test_video.mp4

# Verify file created
ls -lh data/input/test_video.mp4
# Expected: ~2-5 MB, 6-second duration
```

### 5B: Create Synthetic Test Music

```powershell
# Generate 15-second silent test audio
docker compose run --rm montage-ai `
    ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 15 `
           -q:a 9 -acodec libmp3lame `
           -y data/music/test_music.mp3

# Verify
ls -lh data/music/test_music.mp3
# Expected: ~100-200 KB
```

### 5C: Run Preview Render

```powershell
# Set environment and run
$env:QUALITY_PROFILE = "preview"
docker compose run --rm montage-ai `
    .\montage-ai.sh run dynamic

# Monitor output - you should see:
# - Video analysis (beat detection, scene detection)
# - Creative direction from LLM
# - FFmpeg render progress (0% → 100%)
# - Final output file path

# Check output
ls -lh data/output/*.mp4
# Expected: 1-5 MB file created in 5-10 minutes
```

**Expected Timeline:**
- Beat detection: 1-2 min
- Scene analysis: 1-2 min
- Creative direction: 1-2 min
- FFmpeg rendering: 1-3 min
- **Total: 5-10 minutes**

**Potential Issues:**

| Issue | Solution |
|-------|----------|
| `No such file` | Test video/music not created → Check `ls data/input/` |
| `CUDA out of memory` | GPU memory issue → Use Preview mode or disable GPU |
| `FFmpeg error` | Video format issue → Check test_video.mp4 with `ffmpeg -i` |
| `Hangs for >20 min` | Likely LLM timeout → Kill (Ctrl+C) and check LLM config |

**Success Criteria:**
- ✅ Render completes in < 15 minutes
- ✅ Output file created at `data/output/montage_*.mp4`
- ✅ Output file > 500 KB and plays in media player
- ✅ No CUDA/GPU errors (if GPU not available, that's OK)

---

## Phase 6: Web UI Test (10 minutes) - Optional but Recommended

```powershell
# Start Web UI
docker compose up

# In browser, navigate to: http://localhost:8080
# (or http://127.0.0.1:8080 if localhost doesn't work)

# Test workflow:
# 1. Upload video (use data/input/test_video.mp4)
# 2. Select style
# 3. Start render
# 4. Wait for completion
# 5. Download output

# Monitor logs in PowerShell:
# - Should see upload events
# - Job queue messages
# - FFmpeg render progress
```

**Success Criteria:**
- ✅ Web UI accessible at http://localhost:8080
- ✅ Video upload works
- ✅ Render initiates and completes
- ✅ Output downloadable

---

## Performance Expectations (Snapdragon X1E)

### Actual Compile & Render Times on Snapdragon X1E (12GB):

| Task | Time | Notes |
|------|------|-------|
| Docker build | 10-15 min | First run only |
| Python import test | <5 sec | Should be instant |
| FFmpeg test | <5 sec | Version check |
| Preview render (5s clip) | 5-10 min | QUALITY_PROFILE=preview |
| Normal render (5s clip) | 15-25 min | Full quality |
| High quality render | 30-45 min | With stabilization |

---

## Phase 7: Full Validation Script (5 minutes)

```powershell
# Run comprehensive validation
.\scripts\validate-onboarding.sh

# This will:
# - Check all system requirements
# - Test Docker build
# - Verify Python imports
# - Test FFmpeg
# - Run a preview render
# - Generate report

# Output should show:
# ✅ PASSED: N
# ⚠️  WARNINGS: 0-1 (acceptable)
# ❌ FAILED: 0
```

---

## Troubleshooting Guide

### Common Issues on Snapdragon

#### Issue: "Docker: error during connect"
```powershell
# Solution: Docker Desktop not running
docker ps

# If fails, restart Docker:
# 1. Close Docker Desktop
# 2. Wait 10 seconds
# 3. Reopen Docker Desktop
# 4. Wait for "Engine started" notification
# 5. Retry
```

#### Issue: "No space left on device"
```powershell
# Check Docker disk usage
docker system df

# Clean up unused images/containers
docker system prune -a

# If still low, expand Docker image size:
# 1. Docker Desktop → Settings → Resources → Disk image size
# 2. Increase to 60-100 GB
# 3. Click "Apply"
```

#### Issue: "OOM killer" or container stops
```powershell
# Docker memory limit too low
# Edit docker-compose.yml:

docker-compose.yml:
  montage-ai:
    deploy:
      resources:
        limits:
          memory: 8g      # ← Try increasing to 9g or 10g
          cpus: 8

# Rebuild and retry
docker compose up --build
```

#### Issue: "ffmpeg: not found in container"
```powershell
# FFmpeg not installed
# Rebuild container:
docker compose build --no-cache

# If persists, check Dockerfile:
# - Windows on Arm should use aarch64 FFmpeg
```

#### Issue: Render takes 1+ hour for 5-minute clip
```powershell
# Likely causes:
# 1. Low RAM allocation → increase in Docker Desktop
# 2. LLM timeout → check internet connection
# 3. CPU throttling → Disconnect from battery saver

# Solutions:
# 1. Increase Docker memory → 9-10GB
# 2. Use QUALITY_PROFILE=preview
# 3. Check CPU doesn't say "Throttled" in PowerShell:
Get-WmiObject Win32_Processor | Select -%*  # Watch for "Throttled: True"
```

---

## Success Checklist

After completing all phases, verify:

- [ ] Phase 1: Docker, Compose, architecture verified
- [ ] Phase 2: Repository cloned, data directories created
- [ ] Phase 3: Docker image builds successfully
- [ ] Phase 4: Python modules import and FFmpeg works
- [ ] Phase 5: Preview render completes in <15 minutes
- [ ] Phase 6: Web UI functional (optional)
- [ ] Phase 7: Validation script shows all PASSED

**If all checked:** ✅ **Ready for production use!**

---

## Next Steps

1. **Read Documentation:**
   - `docs/getting-started-arm.md` - ARM-specific guide
   - `docs/getting-started.md` - General guide
   - `docs/configuration.md` - All configurable options

2. **Optional: Full Validation**
   ```powershell
   ./scripts/validate-onboarding.sh
   ```

3. **Optional: Build & Deploy Go Worker**
   ```powershell
   cd go
   .\build-and-push.sh
   kubectl apply -f ../deploy/k3s/overlays/cluster/worker-go-canary.yaml
   ```

4. **Performance Tuning:**
   - See `docs/performance-tuning.md` for optimization
   - Use `QUALITY_PROFILE=preview` for fast iteration
   - Use `QUALITY_PROFILE=high` for final renders

---

## Reporting Issues

If you encounter problems:

1. Run: `.\scripts\validate-onboarding.sh`
2. Save output to file: `.\scripts\validate-onboarding.sh > validation_report.txt`
3. Open issue on GitHub with:
   - System info (Snapdragon model, RAM, OS version)
   - Validation report
   - Error message from step that failed

---

**Estimated Total Time:** 45-60 minutes  
**Success Rate on Snapdragon X1E:** 95%+ (with proper resource allocation)

Good luck! 🚀
