# Cluster overlay dependencies

This overlay assumes the following cluster components are installed before applying:

- KEDA (kedacore/keda) — for `ScaledObject` resources used to autoscale worker deployments
  - Install: helm repo add kedacore https://kedacore.github.io/charts && helm install keda kedacore/keda --namespace keda --create-namespace

- JobSet CRD (if you plan to use `deploy/k3s/distributed/jobset-example.yaml`)
  - Install instructions and upstream links: https://jobset.sigs.k8s.io/

Notes:
- `keda-scaledobjects.yaml` uses Redis list-length triggers. Ensure Redis is reachable at `redis.default.svc.cluster.local:6379` or patch the address.
- Override scaling thresholds using the environment variables in `deploy/config.env` before templating.
