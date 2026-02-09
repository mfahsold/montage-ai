## Getting Started

This file is now a pointer to the single source of truth.

Please use [README.md](../README.md) for onboarding, local setup, and ARM64 notes.

If running in Kubernetes, use the provided manifests in `deploy/k3s`.

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

---

## Next Steps

- **[Configuration](configuration.md)** — Tweak every setting
- **[Features](features.md)** — Learn about styles, effects, timeline export
- **[Troubleshooting](troubleshooting.md)** — When things go wrong
