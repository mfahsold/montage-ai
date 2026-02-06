# Quick Start: Montage AI

This is a command cheat sheet. For the full walkthrough, see [Getting Started](docs/getting-started.md).

## Run (Local Docker)

```bash
# Web UI
docker compose up
# Open http://localhost:8080 (or WEB_PORT)

# CLI (uses data/input + data/music)
docker compose run --rm montage-ai ./montage-ai.sh run
```

## Status & Logs

```bash
./montage-status.sh status
./montage-status.sh logs 200
```

## Common Flags

```bash
QUALITY_PROFILE=preview docker compose run --rm montage-ai ./montage-ai.sh run
CGPU_ENABLED=true docker compose run --rm montage-ai ./montage-ai.sh run
```

## Testing

```bash
./scripts/ci.sh
pytest tests/test_montage_builder.py -v
```

## Deployment (Kubernetes)

```bash
make -C deploy/k3s config
make -C deploy/k3s deploy-cluster
```

See `deploy/README.md` and `deploy/k3s/README.md` for full deployment guidance.
