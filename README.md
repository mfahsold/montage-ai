# Montage AI

AI-assisted rough-cutting for video creators, with CLI and web UI workflows.

Quickstart:
```bash
# Web UI (local Docker)
docker compose up

# CLI run (uses data/input + data/music)
docker compose run --rm montage-ai ./montage-ai.sh run
```

Documentation:
- `docs/README.md` for the full public index.
- `QUICK_START.md` for common commands.
- `deploy/README.md` for deployment options.

Development:
```bash
./scripts/ci.sh
make code-health
```
