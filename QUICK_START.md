# Quick Start: Montage AI

This is a command cheat sheet. For the full walkthrough, see [Getting Started](docs/getting-started.md).
For pip/uv install options and optional extras, see [Optional Dependencies](docs/OPTIONAL_DEPENDENCIES.md).

## Run (Local)

```bash
# Web UI
./montage-ai.sh web
# Open http://localhost:8080

# CLI (uses data/input + data/music)
./montage-ai.sh run
```

## Status & Logs

```bash
./montage-status.sh status
./montage-status.sh logs 200
```

## Common Flags

```bash
QUALITY_PROFILE=preview ./montage-ai.sh run
CGPU_ENABLED=true ./montage-ai.sh run
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
