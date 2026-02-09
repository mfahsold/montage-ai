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

## Kubernetes Pod Issues

### Pod Stuck in `Init:0/2`

**Symptom:** Pod stays in `Init:0/2` for minutes or hours:
```
NAME                          READY   STATUS     RESTARTS   AGE
montage-ai-render-8h65q       0/1     Init:0/2   0          2d6h
```

**Common causes and fixes:**

1. **Storage not ready** (most common) — init container waits for `/data/input/.ready`:
   ```bash
   # Run bootstrap to create .ready markers
   ./deploy/k3s/bootstrap.sh

   # Or manually create the marker
   kubectl exec -it -n montage-ai <pod> -c wait-for-nfs-ready -- touch /data/input/.ready
   ```

2. **PVC not bound** — storage class missing or misconfigured:
   ```bash
   kubectl get pvc -n montage-ai
   # All PVCs should show "Bound". If "Pending", check storage class:
   kubectl get storageclass
   ```

3. **Image pull failure** — init container image not available:
   ```bash
   kubectl describe pod <pod> -n montage-ai | grep -A 5 "Init Containers"
   ```

4. **Init container crash** — check init container logs:
   ```bash
   kubectl logs <pod> -c wait-for-nfs-ready -n montage-ai
   kubectl logs <pod> -c init-dirs -n montage-ai
   ```

See [deploy/k3s/README.md](../deploy/k3s/README.md#the-ready-file-issue) for detailed NFS/bootstrap setup.

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

### ImagePullBackOff on Mixed ARM/AMD Clusters

**Symptom:**
```
NAME                     READY   STATUS             RESTARTS   AGE
montage-ai-web-abc123    0/1     ImagePullBackOff   0          2m
```

**Cause:** Node architecture in `config-global.yaml` doesn't match the actual node, so the wrong image platform is pulled.

**Fix:**
```bash
# Check actual architecture
kubectl get nodes -o wide
# See ARCH column: arm64, amd64, etc.

# Update config-global.yaml to match
# nodes:
#   - name: "node1"
#     arch: "arm64"  # ← Must match actual node

# Rebuild multi-arch image
cd deploy/k3s
BUILD_MULTIARCH=true ./build-and-push.sh
```

### Pods Stuck in `Pending` (Storage)

**Symptom:** `kubectl get pods` shows pods stuck in `Pending`.

**Cause:** No StorageClass available, or StorageClass doesn't support the required access mode.

**Fix:**
```bash
# Check StorageClass
kubectl get storageclass

# Check PVC status
kubectl get pvc -n montage-ai
# All should show "Bound". If "Pending":

kubectl describe pvc -n montage-ai <pvc-name>
# Look for: "waiting for first consumer" or "no persistent volumes available"
```

For multi-node clusters, you need an RWX-capable StorageClass (NFS, Longhorn, etc.). See [Cluster Deployment: Storage Setup](cluster-deploy.md#storage-setup).

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

## LLM / AI Issues

### "No LLM backend available"

**Symptom:** Log shows `No LLM backend available (OpenAI API, Ollama)`.

**This is NOT an error.** LLM is optional. Montage AI works fully without it — style templates, beat-synced editing, and all video effects function without LLM. The message is informational.

**To enable LLM features** (natural language prompts, Creative Loop):
- **Ollama (local):** `OLLAMA_HOST=http://localhost:11434`
- **OpenAI-compatible:** `OPENAI_API_BASE=http://your-api/v1 OPENAI_MODEL=auto`
- **Google Gemini:** `GOOGLE_API_KEY=your-key`
- **cgpu:** `CGPU_ENABLED=true`

See [Configuration: AI/LLM](configuration.md#ai--llm-settings) for all options.

---

## Web UI Issues

**Can't access the Web UI**

```bash
# Check if running
docker ps | grep montage-ai

# Check logs
docker logs montage-ai

# Port in use? Try different port
WEB_PORT=8081 docker compose up -d
```

### Port Conflicts

**Symptom:** "Address already in use" or Web UI not reachable

```bash
# Find what's using port 8080
lsof -i :8080          # Linux/macOS
ss -tlnp | grep 8080   # Linux alternative

# Use a different port
WEB_PORT=8081 docker compose up
# Then open http://localhost:8081
```

---

## Docker Build Issues

### Build Takes > 30 Minutes

**Cause:** Slow network or disk I/O bottleneck.

**Fix:**
```bash
# Increase timeout
export DOCKER_CLIENT_TIMEOUT=600
export COMPOSE_HTTP_TIMEOUT=600
docker compose build

# Use verbose output to see progress
docker compose build --progress=plain
```

### "no space left on device"

**Fix:**
```bash
# Clean Docker cache
docker system prune -a --volumes -f

# Check available space (needs >= 3 GB)
df -h /var/lib/docker
```

### Dependency Download Fails

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement opencv-python-headless
```

**Fix:**
```bash
# Retry with plain progress for visibility
docker compose build --progress=plain

# Check network connectivity
ping registry-1.docker.io
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
