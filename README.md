# Montage AI

AI-assisted rough-cutting for video creators, with CLI and web UI workflows.

This README is the single source of truth for onboarding and local setup.

## Quick Start (Docker)

```bash
# Build image (first run)
docker compose build

# Start Web UI
docker compose up

# If 8080 is busy
WEB_PORT=8081 docker compose up
```

Open http://localhost:8080 (or your chosen port).

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

If Docker fails to start, reduce the memory limit in docker-compose.yml.

## First-Time Setup

```bash
git clone https://github.com/mfahsold/montage-ai.git
cd montage-ai

# Run setup script (handles data/ directory creation + permissions)
./scripts/setup.sh

# Generate test media (optional; or provide your own)
./scripts/ops/create-test-video.sh
```

Add your own media (alternative):

```bash
# Copy your videos to data/input/
cp ~/Videos/*.mp4 data/input/

# Copy music to data/music/
cp ~/Music/track.mp3 data/music/
```

**Permission note:** On Linux, if you see "Permission denied" errors, the setup script will help fix `data/` directory ownership. If needed, run:

```bash
sudo chown -R $USER:$USER data/
```

Run:

```bash
docker compose up
# Then open http://localhost:8080
```

## ARM64 (Snapdragon, Apple Silicon)

ARM64 is supported via multi-arch Docker images. Use the same commands.

Verify architecture:

```bash
uname -m
```

Recommended Docker resources (examples):

- Snapdragon 12 GB: memory 8g, cpus 8
- Apple Silicon 16 GB: memory 12g, cpus 8

**MediaPipe note:** On ARM64, MediaPipe (face detection for smart reframing) may not be available. This is expected — auto-reframe falls back to center-crop mode, which works well for most content. The `[WARN] MediaPipe not installed` log message is safe to ignore.

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

## Docs Index

- [Quick Start Guide](docs/quickstart.md) — 5-minute setup
- [Configuration](docs/configuration.md) — All environment variables and settings
- [Features](docs/features.md) — Styles, effects, enhancements, export
- [Troubleshooting](docs/troubleshooting.md) — Common issues and fixes
- [Performance Tuning](docs/performance-tuning.md) — Optimization for your hardware
- [Cluster Deployment](docs/cluster-deploy.md) — Kubernetes/K3s deployment
- [Full Docs Index](docs/README.md) — All documentation

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
