KEDA / Metrics / Cluster‑Autoscaler — quick install & verification (staging)

Purpose
- Provide the minimal infra required to safely run queue‑driven autoscaling (KEDA) and HPA fallback.

Prereqs
- kubectl context with cluster-admin rights for the staging namespace
- Helm 3 installed
- `deploy/k3s/config-global.yaml` configured for your local registry and namespace

1) Metrics Server (required for HPA CPU)
- Install (k3s: metrics-server may already be present):
  helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/
  helm repo update
  helm upgrade --install metrics-server metrics-server/metrics-server --namespace kube-system
- Verify:
  kubectl top nodes

2) KEDA (queue‑driven scaling)
- Add & install:
  helm repo add kedacore https://kedacore.github.io/charts
  helm repo update
  helm upgrade --install keda kedacore/keda --namespace keda --create-namespace
- Verify CRDs:
  kubectl get crds | rg ScaledObject

3) Prometheus Adapter (external/custom metrics for HPA)
- If you run Prometheus via the repo stack, install/prometheus-adapter or use kube-prometheus-stack which provides adapter.
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
  helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter --namespace monitoring
- Verify:
  kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1" | jq .

4) Cluster Autoscaler (optional; cloud dependent)
- For k3s/lab use: provision an autoscaler that understands your infra (or use VM autoscaling scripts).

5) Post‑install checks (staging)
- Apply staging overlay (we keep a DRY overlay):
  kubectl apply -k deploy/k3s/overlays/staging
- Create a ScaledObject smoke (already in repo). Verify:
  CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-montage-ai}"
  kubectl -n "$CLUSTER_NAMESPACE" get scaledobject
  kubectl -n "$CLUSTER_NAMESPACE" describe scaledobject montage-ai-worker-scaledobject

6) Rollback (if needed)
- helm uninstall keda -n keda
- kubectl delete -k deploy/k3s/overlays/staging

Notes & safety
- Install into `keda` / `monitoring` namespaces in staging first. Do not apply to production until smoke tests are green.
- Use conservative thresholds and set `cooldownPeriod` to 120s+ to avoid flapping.

Commands are intentionally parameterized; do not hardcode registry/namespace values — use `deploy/k3s/config-global.yaml` for environment overrides.
