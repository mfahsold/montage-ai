# Cluster overlay dependencies

This overlay assumes the following cluster components are installed before applying:

- KEDA (kedacore/keda) — for `ScaledObject` resources used to autoscale worker deployments
  - Install: helm repo add kedacore https://kedacore.github.io/charts && helm install keda kedacore/keda --namespace keda --create-namespace

- JobSet CRD (if you plan to use `deploy/k3s/distributed/jobset-example.yaml`)
  - Install instructions and upstream links: https://jobset.sigs.k8s.io/

Notes:
- `keda-scaledobjects.yaml` uses Redis list-length triggers. Set `REDIS_HOST`/`REDIS_PORT` via `deploy/config.env` or `deploy/k3s/config-global.yaml` (rendered into `cluster-config.env`).

## Troubleshooting Scheduling Issues

Common causes for pods remaining `Pending`:

- PersistentVolume bound to a specific node (local-path). Check PVC annotations `volume.kubernetes.io/selected-node` and ensure pods can schedule there.
- Untolerated node taints (e.g., `gpu=amd-rocm:NoSchedule`). Add the appropriate toleration or change node taints.
- Insufficient requested CPU/memory on available nodes. Consider adjusting requests/limits in `deploy/k3s/base/worker.yaml` or increasing cluster capacity.

Tips:

- Inspect events: `kubectl describe pod <pod> -n montage-ai` → look for "didn't match PersistentVolume's node affinity" or "untolerated taint(s)".
- Check nodes/taints: `kubectl get nodes -o wide` and `kubectl describe node <node>`.

Note: If jobs start but complete immediately with `No videos found in input directory`, ensure test content is available on the `montage-input` PVC (e.g., copy a small test clip to `/data/input` on the node hosting that PVC, or mount test fixtures into the `montage-ai-web`/`montage-ai-worker` pods during dev runs).
