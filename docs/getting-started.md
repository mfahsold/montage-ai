# Getting Started

For local setup (Docker), see the [README](../README.md).

For ARM64 platforms (Snapdragon, Apple Silicon, Raspberry Pi), see [Getting Started on ARM](getting-started-arm.md).

---

## Kubernetes

For cluster deployments:

```bash
# Render config
make -C deploy/k3s config

# Canonical cluster overlay
make -C deploy/k3s deploy-cluster
```

The canonical cluster overlay is `deploy/k3s/overlays/cluster`.

Check job status:

```bash
python3 -m montage_ai.cli jobs --api-base http://<cluster-service> list
```

See [deploy/k3s/README.md](../deploy/k3s/README.md) for the full cluster deployment guide.

---

## Next Steps

- **[Configuration](configuration.md)** — All environment variables and settings
- **[Features](features.md)** — Styles, effects, timeline export
- **[Troubleshooting](troubleshooting.md)** — Common issues and fixes
