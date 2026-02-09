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
docker compose run --rm montage-ai ./montage-ai.sh run
```

Preview mode (faster):

```bash
QUALITY_PROFILE=preview docker compose run --rm montage-ai ./montage-ai.sh run
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

mkdir -p data/input data/music data/output data/assets
```

Add media:

```bash
cp ~/Videos/*.mp4 data/input/
cp ~/Music/track.mp3 data/music/
```

Run:

```bash
docker compose up
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

If you want an automated check and a preview render test:

```bash
./scripts/quick-setup-arm.sh
./scripts/validate-onboarding.sh
```

## Troubleshooting

- Port in use: `WEB_PORT=8081 docker compose up`
- Low RAM: use `QUALITY_PROFILE=preview`
- Docker start error: lower memory in docker-compose.yml
- aarch64 local venv: `mediapipe` is unavailable on Python >= 3.13; use Docker or skip `[ai]`

## Docs Index

- [docs/README.md](docs/README.md)
- [docs/troubleshooting.md](docs/troubleshooting.md)
- [docs/performance-tuning.md](docs/performance-tuning.md)
- [deploy/README.md](deploy/README.md)

## Development

```bash
./scripts/ci.sh
make code-health
```
