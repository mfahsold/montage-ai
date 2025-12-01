# Installation Guide

Get Montage-AI running on your system.

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Docker | ✅ | Recommended method |
| Docker Compose | ✅ | v2.0+ |
| 8GB RAM | ✅ | 16GB for HQ mode |
| 10GB disk | ✅ | Plus space for media |

**Optional:**

| Component | For | Installation |
|-----------|-----|--------------|
| Ollama | Local AI Director | [ollama.ai](https://ollama.ai/) |
| cgpu | Cloud LLM/GPU | `npm install -g cgpu` |
| kubectl | K8s deployment | [kubernetes.io](https://kubernetes.io/docs/tasks/tools/) |

---

## Quick Start (Docker)

**Works on:** Linux, macOS, Windows (WSL2)

### 1. Install Docker

<details>
<summary><strong>Linux (Ubuntu/Debian)</strong></summary>

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
```

</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
# Install Docker Desktop
brew install --cask docker

# Or download from: https://www.docker.com/products/docker-desktop

# Start Docker Desktop, then verify
docker --version
```

</details>

<details>
<summary><strong>Windows</strong></summary>

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Enable WSL2 backend in Docker Desktop settings
3. Open PowerShell or WSL2 terminal

```powershell
docker --version
```

</details>

### 2. Clone & Build

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai
./montage-ai.sh build
```

### 3. Add Media

```bash
# Add your video clips
cp /path/to/your/videos/*.mp4 data/input/

# Add background music
cp /path/to/your/music.mp3 data/music/
```

**Supported formats:**

- Video: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- Audio: `.mp3`, `.wav`, `.flac`, `.m4a`

### 4. Create Montage

```bash
# Basic run (uses 'dynamic' style)
./montage-ai.sh run

# Choose a style
./montage-ai.sh run hitchcock

# Fast preview
./montage-ai.sh preview

# High quality + effects
./montage-ai.sh hq
```

**Output:** `data/output/montage.mp4`

### 5. Verify It Works

```bash
# List available commands
./montage-ai.sh help

# List styles
./montage-ai.sh list

# Check logs if something fails
./montage-ai.sh logs
```

---

## Optional: AI Director (Ollama)

The AI Director uses an LLM to interpret creative prompts. Without it, Montage-AI uses preset styles only.

### Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download

# Pull the model
ollama pull llama3.1:8b
```

### Use AI Director

```bash
# Set a creative prompt
export CREATIVE_PROMPT="Fast-paced travel montage with dramatic builds"
./montage-ai.sh run
```

**Note:** Ollama runs on your machine. For cloud LLM (faster, no local resources), see [cgpu section](#cloud-gpu-cgpu).

---

## Optional: Cloud GPU (cgpu)

Use free cloud resources via [cgpu](https://github.com/RohanAdwankar/cgpu):

- **Gemini LLM** — Faster than local Ollama
- **Cloud GPU** — For AI upscaling without local GPU

### Setup

```bash
# 1. Install cgpu
npm install -g cgpu

# 2. Install gemini-cli
# Follow: https://github.com/google-gemini/gemini-cli

# 3. Authenticate
gemini auth login
```

### Usage

```bash
# Start cgpu server
./montage-ai.sh cgpu-start

# Run with cloud LLM
./montage-ai.sh run --cgpu

# Run with cloud GPU for upscaling
./montage-ai.sh run --upscale --cgpu-gpu

# Stop when done
./montage-ai.sh cgpu-stop
```

---

## Kubernetes Deployment

For production or batch processing on a cluster.

### Prerequisites

- Kubernetes cluster (K3s, K8s, EKS, GKE)
- `kubectl` configured
- Container image available

### Deploy

```bash
# 1. Apply base manifests
kubectl apply -k deploy/k3s/base/

# 2. Check resources
kubectl get all -n montage-ai

# 3. Load your media (see deploy/k3s/README.md for methods)

# 4. Start render job
kubectl apply -f deploy/k3s/base/job.yaml

# 5. Watch progress
kubectl logs -n montage-ai -f job/montage-ai-render
```

### Variants

```bash
# Development (fast preview)
kubectl apply -k deploy/k3s/overlays/dev/

# Production (HQ + AMD GPU)
kubectl apply -k deploy/k3s/overlays/production/
```

→ **[Full K8s Guide](../deploy/k3s/README.md)**

---

## Local Development (No Docker)

For contributors or advanced users.

### Prerequisites

- Python 3.10+
- FFmpeg
- Git

### Setup

```bash
# Clone
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install FFmpeg
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: choco install ffmpeg

# Verify
python -c "from montage_ai import editor; print('OK')"
```

### Run

```bash
# Set paths
export INPUT_DIR=./data/input
export MUSIC_DIR=./data/music
export OUTPUT_DIR=./data/output

# Run editor
python -m montage_ai.editor
```

### Development Commands

```bash
make help      # Show all commands
make test      # Run tests
make lint      # Check code style
make shell     # Container shell
```

---

## Troubleshooting

### Docker

| Problem | Solution |
|---------|----------|
| Permission denied | `sudo usermod -aG docker $USER && newgrp docker` |
| Build fails | `docker system prune -af && ./montage-ai.sh build` |
| Out of disk | `docker image prune -af` |

### Media

| Problem | Solution |
|---------|----------|
| "No input videos found" | Add `.mp4`/`.mov` files to `data/input/` |
| "No music files found" | Add `.mp3`/`.wav` to `data/music/` |
| Corrupt output | Try different input codec, re-encode with FFmpeg |

### AI Director

| Problem | Solution |
|---------|----------|
| "Ollama connection refused" | Start Ollama: `ollama serve` |
| Slow generation | Use `--cgpu` for cloud LLM |
| JSON parse error | Update Ollama model: `ollama pull llama3.1:8b` |

### Kubernetes

| Problem | Solution |
|---------|----------|
| Pod pending | Check PVC: `kubectl get pvc -n montage-ai` |
| Image pull error | Check registry access, image exists |
| Job failed | `kubectl logs -n montage-ai job/montage-ai-render` |

---

## Next Steps

- **[Configuration](configuration.md)** — Environment variables
- **[Styles](styles.md)** — Style templates
- **[Architecture](architecture.md)** — System design
- **[Contributing](../CONTRIBUTING.md)** — How to contribute
