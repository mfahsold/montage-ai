# Automate configuration of deploy/k3s/config-global.yaml

**Description:**

During installation verification, it was noted that `deploy/k3s/config-global.yaml.example` requires manual replacement of placeholders like `<REGISTRY_HOST>`, `<CLUSTER_NAMESPACE>`, etc. before `make config` can be run. Verification failed initially until these were manually edited.

This hinders automated deployment testing and increases friction for new developers.

**Steps to Reproduce:**

1. Clone repo.
2. Run `cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml`.
3. Run `make -C deploy/k3s config`.
4. Observe failure in `render_cluster_config_env.sh` locally or invalid config generation if variables aren't set.

**Suggestion:**

- Update `render_cluster_config_env.sh` to accept environment variables or flags to generate this file automatically.
- Provide a `make init-config` target that asks interactively or uses sensible defaults for `k3d`/`minikube` (e.g., "interactive setup" argument).
- Allow "No Registry" mode for local development where images are side-loaded, without needing to specify a dummy registry URL in the config.

**Priority:** Medium
**Type:** Developer Experience / Automation

**Verification:**

Deployed successfully to `montage-ai-test` namespace after manual configuration of `deploy/k3s/config-global.yaml`.
Deployment checks passed and `deploy.sh` correctly handled existing resources (idempotency confirmed).
