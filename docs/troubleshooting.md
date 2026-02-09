# Troubleshooting

Common issues and how to fix them.

For Kubernetes/on-call fixes, see the public stub at [KUBERNETES_RUNBOOK.md](KUBERNETES_RUNBOOK.md) or request access to the internal runbook.

---

## Docker Startup Issues

### "OCI runtime error", "Docker not starting"

**Symptom:** `docker compose up` fails immediately with:
```
error: container exited with error: [error] systemd-executor 
OCI runtime error: memory cgroup out of memory
```

**Cause:** Your `memory:` limit in `docker-compose.yml` exceeds your system RAM.

**Fix:**

```bash
# Check your actual RAM:
free -h | grep Mem
  # If you see: 16Gi total

# Edit docker-compose.yml and set memory to:
# memory: 12g (leaves 4GB for OS)

# Then retry:
docker compose up
```

### "docker: command not found"

**Fix:**

```bash
# macOS:
brew install docker docker-compose

# Ubuntu/Debian:
sudo apt-get install docker.io docker-compose

# Then restart Docker:
sudo systemctl restart docker

# Or on macOS:
open /Applications/Docker.app

# Verify:
docker --version
```

### "Cannot connect to Docker daemon"

**Fix:**

```bash
# Ensure Docker is running
sudo systemctl start docker   # Linux
# or
open /Applications/Docker.app # macOS

# Add yourself to docker group (Linux, restart shell after):
sudo usermod -aG docker $USER
newgrp docker

# Verify:
docker ps
```

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

| Error                | Cause              | Solution                               |
| -------------------- | ------------------ | -------------------------------------- |
| `CUDA out of memory` | Video too large    | Use smaller clips or reduce resolution |
| `session expired`    | Colab disconnected | Runs auto-retry (2 attempts)           |
| `CUDA not available` | No GPU assigned    | Run `cgpu connect` again               |

---

## Kubernetes Deploy Errors

### PVC Patch Fails ("spec is immutable")

**Symptom:** `make -C deploy/k3s deploy-cluster` fails with:

```text
PersistentVolumeClaim "montage-ai-input-nfs" is invalid: spec is immutable
```

**Cause:** Existing PVCs in the cluster have a different `storageClassName`,
`accessModes`, or `storage` size than the values rendered from
`config-global.yaml`.

**Fix:**

```bash
# 1. Check existing PVC settings
kubectl get pvc -n montage-ai -o custom-columns=\
'NAME:.metadata.name,CLASS:.spec.storageClassName,ACCESS:.spec.accessModes[0],SIZE:.spec.resources.requests.storage'

# 2. Update config-global.yaml to match:
#    - storage.classes.default → must match existing storageClassName
#    - storage.pvc.* → must match existing PVC names

# 3. Re-render and deploy
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster
```

If the PVCs are no longer needed, delete them first (data loss!):

```bash
kubectl delete pvc <pvc-name> -n montage-ai
```

### Ingress Fails ("host is invalid")

**Symptom:** `kubectl apply` rejects the Ingress with:

```text
host: montage-ai.<YOUR_DOMAIN> is invalid
```

**Cause:** The `cluster.hostnames.montage` value in `config-global.yaml` still
contains placeholder characters (`<>`), which violate RFC1123.

**Fix:** Set a valid hostname in `config-global.yaml`:
```yaml
cluster:
  hostnames:
    montage: "montage-ai.example.com"   # or "montage-ai.local" for dev
```

Then re-render: `make -C deploy/k3s config`

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

**Can't access the Web UI**

```bash
# Check if running
docker ps | grep montage-ai

# Check logs

# Port in use? Try different port
WEB_PORT=5002 docker compose up -d
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
