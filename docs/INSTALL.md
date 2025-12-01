# Installation Guide

Complete installation instructions for Montage AI.

## Table of Contents

- [Quick Start (Docker)](#quick-start-docker)
- [Local Development](#local-development)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud GPU (cgpu)](#cloud-gpu-cgpu)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (Docker)

The fastest way to get started. Works on any system with Docker.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2+)

### Installation

```bash
# Clone the repository
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai

# Build the image
./montage-ai.sh build

# Add your media files
cp /path/to/your/videos/* data/input/
cp /path/to/your/music.mp3 data/music/

# Create your first montage
./montage-ai.sh run
```

### Verify Installation

```bash
# Check Docker is working
docker --version
docker compose version

# Test the build
./montage-ai.sh build

# List available styles
./montage-ai.sh list
```

---

## Local Development

For developers who want to modify the code or run without Docker.

### Prerequisites

- Python 3.10+
- FFmpeg
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai

# Create conda environment
conda create -n montage-ai python=3.10 -y
conda activate montage-ai

# Install dependencies
conda install -c conda-forge librosa numba numpy scipy ffmpeg -y
pip install -r requirements.txt
pip install -e .

# Install FFmpeg (if not via conda)
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: choco install ffmpeg

# Verify installation
python -c "from montage_ai import editor; print('OK')"
```

### Running Locally (without Docker)

```bash
# Set environment variables
export CUT_STYLE=dynamic
export VERBOSE=true

# Run the editor
python -m montage_ai.editor
```

### Development Workflow

```bash
# Use Make for common tasks
make help           # Show all commands
make build          # Build Docker image
make run            # Run montage
make shell          # Interactive container shell
make test           # Run tests
make validate       # Validate K8s manifests
```

---

## Kubernetes Deployment

Deploy Montage AI as batch jobs on any Kubernetes cluster.

### Prerequisites

- Kubernetes 1.28+ (K3s, K8s, EKS, GKE, etc.)
- kubectl configured
- Container registry access

### Option A: Using Pre-built Images

```bash
# Deploy directly from GitHub Container Registry
kubectl apply -k deploy/k3s/base/

# Start a render job
kubectl apply -f deploy/k3s/base/job.yaml

# Watch logs
kubectl logs -n montage-ai -f job/montage-ai-render
```

### Option B: Building Your Own Image

```bash
# Build for amd64 (most clusters)
make build-amd64

# Push to your registry
docker tag ghcr.io/mfahsold/montage-ai:latest your-registry/montage-ai:latest
docker push your-registry/montage-ai:latest

# Update image reference in kustomization.yaml
# Then deploy
kubectl apply -k deploy/k3s/base/
```

### Deployment Variants

```bash
# Base - generic amd64 deployment
kubectl apply -k deploy/k3s/base/

# Development - fast preview, low resources
kubectl apply -k deploy/k3s/overlays/dev/

# Production - AMD GPU targeting, high quality
kubectl apply -k deploy/k3s/overlays/production/
```

### Loading Media Data

```bash
# Option 1: Use a data loader pod
kubectl run -it --rm data-loader \
  --image=busybox \
  --namespace=montage-ai \
  --overrides='{"spec":{"containers":[{"name":"data-loader","image":"busybox","volumeMounts":[{"name":"input","mountPath":"/data/input"}]}],"volumes":[{"name":"input","persistentVolumeClaim":{"claimName":"montage-ai-input"}}]}}' \
  -- sh

# Option 2: Copy from local machine (requires running pod)
kubectl cp ./my-videos/ montage-ai/<pod-name>:/data/input/
kubectl cp ./my-music.mp3 montage-ai/<pod-name>:/data/music/
```

### Cluster Requirements

| Resource | Minimum  | Recommended              |
| -------- | -------- | ------------------------ |
| CPU      | 2 cores  | 4+ cores                 |
| Memory   | 8 GB     | 16+ GB                   |
| Storage  | 50 GB    | 200+ GB                  |
| GPU      | Optional | AMD/NVIDIA for upscaling |

---

## Cloud GPU (cgpu)

Use free cloud GPUs via [cgpu](https://github.com/RohanAdwankar/cgpu) for:
- **LLM Access**: Free Gemini API for Creative Director
- **GPU Compute**: Free Google Colab GPUs for upscaling

### Prerequisites

1. **Install cgpu**
   ```bash
   npm install -g cgpu
   ```

2. **Install gemini-cli** (for LLM features)
   ```bash
   # Follow instructions at:
   # https://github.com/google-gemini/gemini-cli
   ```

3. **Authenticate with Google**
   ```bash
   gemini auth login
   ```

### Using cgpu for LLM (Creative Director)

```bash
# Start the cgpu server
./montage-ai.sh cgpu-start

# Run with Gemini LLM
./montage-ai.sh run --cgpu

# Check status
./montage-ai.sh cgpu-status

# Stop when done
./montage-ai.sh cgpu-stop
```

### Using cgpu for GPU Upscaling

```bash
# Enable cloud GPU for upscaling
./montage-ai.sh run --cgpu-gpu --upscale

# Full cloud mode (LLM + GPU)
./montage-ai.sh hq hitchcock --cgpu --cgpu-gpu
```

### cgpu Configuration

| Variable           | Default          | Description                 |
| ------------------ | ---------------- | --------------------------- |
| `CGPU_ENABLED`     | false            | Enable Gemini LLM           |
| `CGPU_HOST`        | localhost        | cgpu server host            |
| `CGPU_PORT`        | 8080             | cgpu server port            |
| `CGPU_MODEL`       | gemini-2.0-flash | Gemini model                |
| `CGPU_GPU_ENABLED` | false            | Enable cloud GPU            |
| `CGPU_TIMEOUT`     | 600              | Operation timeout (seconds) |

### Fallback Behavior

cgpu features gracefully fall back when unavailable:

| Feature           | Primary          | Fallback           |
| ----------------- | ---------------- | ------------------ |
| Creative Director | Gemini (cgpu)    | Ollama (local)     |
| Upscaling         | Cloud GPU (cgpu) | Local Vulkan → CPU |

---

## Troubleshooting

### Docker Issues

```bash
# Permission denied
sudo usermod -aG docker $USER
newgrp docker

# Build fails
docker system prune -af
./montage-ai.sh build

# Out of disk space
docker system df
docker image prune -af
```

### Kubernetes Issues

```bash
# Pod stuck in Pending
kubectl describe pod -n montage-ai -l app.kubernetes.io/name=montage-ai

# Image pull errors
kubectl get events -n montage-ai --sort-by='.lastTimestamp'

# Check PVC binding
kubectl get pvc -n montage-ai

# View logs
kubectl logs -n montage-ai -l app.kubernetes.io/name=montage-ai --tail=100
```

### cgpu Issues

```bash
# cgpu not found
npm install -g cgpu

# Gemini auth failed
gemini auth login

# Server won't start
./montage-ai.sh cgpu-stop
./montage-ai.sh cgpu-start
```

### Common Errors

| Error                       | Solution                       |
| --------------------------- | ------------------------------ |
| "No input videos found"     | Add videos to `data/input/`    |
| "No music files found"      | Add audio to `data/music/`     |
| "CUDA not available"        | Use `--cgpu-gpu` for cloud GPU |
| "Ollama connection refused" | Start Ollama or use `--cgpu`   |

---

## Next Steps

- [Configuration Guide](configuration.md) - All environment variables
- [Style Guide](styles.md) - Style templates and customization
- [Architecture](architecture.md) - System design
- [Contributing](../CONTRIBUTING.md) - How to contribute
