# Cluster overlay dependencies

This overlay assumes the following cluster components are installed before applying:

- KEDA (kedacore/keda) — for `ScaledObject` resources used to autoscale worker deployments
  - Install: helm repo add kedacore https://kedacore.github.io/charts && helm install keda kedacore/keda --namespace keda --create-namespace

- JobSet CRD (if you plan to use `deploy/k3s/distributed/jobset-example.yaml`)
  - Install instructions and upstream links: https://jobset.sigs.k8s.io/

Notes:
- `keda-scaledobjects.yaml` uses Redis list-length triggers. Ensure Redis is reachable at `redis.default.svc.cluster.local:6379` or patch the address.
- Override scaling thresholds using the environment variables in `deploy/config.env` before templating.

## Dev overlay (low-resource/dev-friendly)

If you're deploying to a small or heterogeneous dev cluster, use the provided `dev` overlay which applies smaller resource requests and adds a toleration for GPU node taints where local PVs are often bound. This helps schedule web/worker/redis onto nodes hosting local-path volumes.

Usage:

- Deploy the canonical cluster overlay in dev mode:

```bash
make -C deploy/k3s deploy-dev
```

This is intended as a convenience for developer clusters and **should not** replace production tuning. See "Troubleshooting" below if pods remain pending.

## Troubleshooting Scheduling Issues

Common causes for pods remaining `Pending`:

- PersistentVolume bound to a specific node (local-path). Check PVC annotations `volume.kubernetes.io/selected-node` and ensure pods can schedule there.
- Untolerated node taints (e.g., `gpu=amd-rocm:NoSchedule`). Add the appropriate toleration or change node taints.
- Insufficient requested CPU/memory on available nodes. Consider reducing requests via the `dev` overlay or increasing cluster capacity.

Tips:

- Inspect events: `kubectl describe pod <pod> -n montage-ai` → look for "didn't match PersistentVolume's node affinity" or "untolerated taint(s)".
- Check nodes/taints: `kubectl get nodes -o wide` and `kubectl describe node <node>`.
- Temporary dev fix: `make -C deploy/k3s deploy-dev` will apply the dev-friendly patches (reduced requests + a common toleration).

Note: If jobs start but complete immediately with `No videos found in input directory`, ensure test content is available on the `montage-input` PVC (e.g., copy a small test clip to `/data/input` on the node hosting that PVC, or mount test fixtures into the `montage-ai-web`/`montage-ai-worker` pods during dev runs).
