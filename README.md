# Montage AI

AI-assisted rough-cutting for video creators, with CLI and web UI workflows.

This README is the single source of truth for onboarding and local setup.

## Quick Start (Docker)

```bash
# Build image (first run)
docker compose build

# Start Web UI
docker compose up
```

Open http://localhost:8080.

> **Port conflict?** If port 8080 is already in use, override with: `WEB_PORT=8081 docker compose up` (then open http://localhost:8081).

CLI run (uses data/input + data/music):

```bash
docker compose run --rm montage-ai /app/montage-ai.sh run
```

**Output:** `data/output/montage_<timestamp>.mp4`
**Duration:** ~2-5 min for 3x 30s clips (system-dependent)
**Progress:** Logs show beat detection -> scene analysis -> clip assembly -> rendering

Preview mode (faster):

```bash
QUALITY_PROFILE=preview docker compose run --rm montage-ai /app/montage-ai.sh run
# → 360p output, ~60% faster
```

> **See also:** [Installation Test Guide](docs/INSTALLATION_TEST.md) to verify your setup, [Configuration](docs/configuration.md) for all environment variables.

## System Requirements

- Docker + Docker Compose v2
- 16 GB RAM recommended (8 GB minimum for preview)
- 4+ CPU cores recommended (2 cores minimum)
- 10+ GB free disk space

Quick checks:

```bash
docker --version
docker compose version
free -h | grep Mem
nproc
df -h /
```

Windows (PowerShell):

```powershell
docker --version
docker compose version
RAM: (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
CPU: (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
```

**Resource limits:** Docker defaults to 12 GB memory / 4 CPUs (optimized for 16 GB systems). Override for your system:

```bash
# 32 GB system (recommended for best performance):
DOCKER_MEMORY_LIMIT=24g DOCKER_CPU_LIMIT=8 docker compose up

# 8 GB system (minimum, use preview mode):
DOCKER_MEMORY_LIMIT=6g DOCKER_CPU_LIMIT=2 QUALITY_PROFILE=preview docker compose up
```

If Docker fails to start with "OCI runtime error", reduce the memory limit below your system RAM.

## First-Time Setup

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai

# Run setup script (required before first build)
./scripts/setup.sh

# Generate test media (optional; or provide your own)
./scripts/ops/create-test-video.sh
```

**What `setup.sh` does:**

| Check | Action |
|-------|--------|
| Data directories | Creates `data/input`, `data/music`, `data/output`, `data/assets`, `data/luts` |
| Permissions (Linux) | Fixes ownership if directories are owned by root |
| Docker | Verifies Docker and Compose v2 are installed |
| Disk space | Warns if < 30 GB free, fails if < 5 GB |
| RAM | Reports available system memory |
| Architecture | Detects ARM64 and notes MediaPipe limitation |

The script is idempotent — safe to re-run at any time.

Add your own media (alternative):

```bash
# Copy your videos to data/input/
cp ~/Videos/*.mp4 data/input/

# Copy music to data/music/
cp ~/Music/track.mp3 data/music/
```

**Permission note:**
- **Linux:** If you see "Permission denied" errors, the setup script will help fix `data/` directory ownership. If needed, run: `sudo chown -R $USER:$USER data/`
- **macOS / Windows (Docker Desktop):** Permissions are handled automatically by Docker Desktop. No manual fix needed.

Run:

```bash
docker compose up
# Then open http://localhost:8080
```

## ARM64 (Snapdragon, Apple Silicon)

ARM64 is supported via multi-arch Docker images. Use the same commands.

> **ARM64 limitation:** MediaPipe (face detection for auto-reframe) is not available on ARM64. Auto-reframe uses **center-crop fallback**, which works well for most content. The `[WARN] MediaPipe not installed` log message is safe to ignore. See [Optional Dependencies](docs/OPTIONAL_DEPENDENCIES.md) for details.

Verify architecture:

```bash
uname -m
```

Recommended Docker resources (examples):

- Snapdragon 12 GB: memory 8g, cpus 8
- Apple Silicon 16 GB: memory 12g, cpus 8

If you want an automated check and a preview render test:

```bash
./scripts/quick-setup-arm.sh
./scripts/validate-onboarding.sh
```

## Troubleshooting

- Port in use: `WEB_PORT=8081 docker compose up`
- Low RAM (8 GB): `DOCKER_MEMORY_LIMIT=6g QUALITY_PROFILE=preview docker compose up`
- Docker OCI error: lower memory limit: `DOCKER_MEMORY_LIMIT=6g docker compose up`
- High-performance system: `DOCKER_MEMORY_LIMIT=24g DOCKER_CPU_LIMIT=8 docker compose up`
- aarch64 local venv: `mediapipe` is unavailable on Python >= 3.13; use Docker or skip `[ai]`

## Documentation Navigator

**Where should I start?**

```
New user?
├── Just want to try it → docs/quickstart.md (5 min)
├── Full setup guide    → docs/getting-started.md
└── ARM64 device?       → docs/getting-started-arm.md

Already running?
├── Configure settings  → docs/configuration.md
├── See all features    → docs/features.md
├── Fix an error        → docs/troubleshooting.md
└── Tune performance    → docs/performance-tuning.md

Deploying to K8s?
├── Cluster setup       → docs/cluster-deploy.md
├── Full K8s reference  → deploy/k3s/README.md
└── Operations          → docs/operations/README.md

Developing?
├── Architecture        → docs/architecture.md
├── Contributing        → CONTRIBUTING.md
└── Full docs index     → docs/README.md
```

## LLM Backend: Optional

Montage AI works **without any LLM backend**. Style templates, beat-synced editing, and all video effects work out of the box. LLM adds natural language creative direction but is not required.

| Capability | No LLM | With LLM (Ollama/Gemini/OpenAI) |
|-----------|--------|--------------------------------|
| Style templates (7 built-in) | Yes | Yes |
| Beat-synced editing | Yes | Yes |
| All video effects | Yes | Yes |
| Natural language prompts | — | Yes |
| Creative Loop (iterative refinement) | — | Yes |
| Custom editing instructions | — | Yes |

If you see `No LLM backend available` in the logs, this is informational — not an error. To enable LLM features, see [Configuration: AI/LLM Settings](docs/configuration.md#ai--llm-settings).

## How It Works

```
Input Videos + Music ─> Beat Detection ─> Scene Analysis ─> Clip Assembly ─> FFmpeg Render ─> Final Video
                              │                                    ▲
                              └── Creative Director (LLM) ─────────┘
                                  Style Template (JSON)
```

See [Architecture](docs/architecture.md) for the full component diagram.

## Key Features

- Beat-synced editing with 7 built-in style templates
- Smart reframing (16:9 to 9:16) for TikTok/Shorts
- AI denoising, stabilization, upscaling, color grading, film grain
- Dialogue ducking, audio normalization, voice isolation
- Caption burning (TikTok, Minimal, Bold, Cinematic, Karaoke)
- Timeline export (OTIO, EDL, CSV) for DaVinci Resolve / Premiere Pro
- Story engine with narrative arc optimization
- Cloud GPU acceleration via cgpu (optional)
- Quality profiles: Preview (360p fast) / Standard / High / Master (4K)

See [Features](docs/features.md) for the complete feature matrix and CLI examples.

## Development

```bash
./scripts/ci.sh
make code-health
```
