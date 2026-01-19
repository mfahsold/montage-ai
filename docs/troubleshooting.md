# Troubleshooting

Common issues and how to fix them.

For Kubernetes/on-call fixes, see the public stub at [KUBERNETES_RUNBOOK.md](KUBERNETES_RUNBOOK.md) or request access to the internal runbook.

---

## Memory Issues

**Symptom:** Job crashes, container killed, "Out of Memory" error

**Fix:**

```bash
# Reduce memory usage
MEMORY_LIMIT_GB=12 \
PARALLEL_ENHANCE=false \
FFMPEG_PRESET=ultrafast \
./montage-ai.sh run
```

Or offload processing to cloud GPU:

```bash
CGPU_GPU_ENABLED=true ./montage-ai.sh run --upscale
```

---

## Slow Performance
**Symptom:** Rendering takes forever

**Fix:**
```bash
# Speed over quality
PARALLEL_ENHANCE=true \
UPSCALE=false \
./montage-ai.sh preview hitchcock
```

---

## Cloud GPU Problems

**Symptom:** "CUDA not available", "session expired", upscaling fails

**Fix:**

1. Check cgpu status:
   ```bash
   cgpu status
   ```

2. Reconnect if needed:
   ```bash
   cgpu connect
   ```

3. Retry with verbose logging:
   ```bash
   VERBOSE=true CGPU_GPU_ENABLED=true ./montage-ai.sh run
   ```

**Common CUDA errors:**

| -------------------- | ------------------ | -------------------------------------- |
| `CUDA out of memory` | Video too large    | Use smaller clips or reduce resolution |
| `session expired`    | Colab disconnected | Runs auto-retry (2 attempts)           |
| `CUDA not available` | No GPU assigned    | Run `cgpu connect` again               |

---

## Missing Files

**"No videos found"**

```bash
ls data/input/   # Should see your .mp4 files
```


```bash
ls data/music/   # Should see your .mp3 file
```

---

## Disk Space

**Symptom:** `/tmp` fills up, container runs out of space

**Fix:**

```bash
# Enable auto-cleanup (default in recent versions)

# Manual cleanup
docker exec montage-ai rm -rf /tmp/*.mp4
```

---

## Web UI Issues

**Can't access http://localhost:5001**

```bash
# Check if running
docker ps | grep web-ui

# Check logs

# Port in use? Try different port
docker compose -f docker-compose.web.yml up -d -e PORT=5002
```

---

## FFmpeg Errors

**"FFmpeg not found"**

FFmpeg is included in the Docker image. If running locally:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

---

## Still Stuck?

1. Run with verbose logging:
   ```bash
   VERBOSE=true ./montage-ai.sh run 2>&1 | tee debug.log
   ```

2. Check the [GitHub Issues](https://github.com/mfahsold/montage-ai/issues)

3. Open a new issue with your `debug.log`
