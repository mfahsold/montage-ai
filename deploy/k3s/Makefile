# ==============================================================================
# Montage-AI K3s/K8s - Canonical Makefile (Fluxibri-aligned)
# Single entry point for cluster operations
# ==============================================================================

.PHONY: help config validate diff deploy deploy-cluster status health

GREEN  := $(shell tput -Txterm setaf 2 2>/dev/null || echo '')
YELLOW := $(shell tput -Txterm setaf 3 2>/dev/null || echo '')
CYAN   := $(shell tput -Txterm setaf 6 2>/dev/null || echo '')
RED    := $(shell tput -Txterm setaf 1 2>/dev/null || echo '')
RESET  := $(shell tput -Txterm sgr0 2>/dev/null || echo '')

KUBECTL ?= kubectl
KUSTOMIZE_BUILD ?= kustomize build --load-restrictor LoadRestrictionsNone
CONFIG_GLOBAL ?= ./config-global.yaml
CLUSTER_NAMESPACE ?= montage-ai

CONFIG_ENV_SCRIPT := ../../scripts/ops/render_cluster_config_env.sh

help: ## Show this help message
	@echo '${CYAN}Montage-AI Cluster Management${RESET}'
	@echo ''
	@echo 'Usage: ${YELLOW}make <target>${RESET}'
	@echo ''
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${YELLOW}%-25s${RESET} %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ''

config: ## Render deploy/k3s/base/cluster-config.env from config-global.yaml
	@if [ -x "$(CONFIG_ENV_SCRIPT)" ]; then \
		CONFIG_GLOBAL="$(CONFIG_GLOBAL)" ENV_OUT=./base/cluster-config.env bash "$(CONFIG_ENV_SCRIPT)"; \
	else \
		echo "${YELLOW}Missing $(CONFIG_ENV_SCRIPT); using cluster-config.env.example${RESET}"; \
		cp ./base/cluster-config.env.example ./base/cluster-config.env; \
	fi

validate: config ## Validate kustomize builds
	@echo "${GREEN}Validating base manifests...${RESET}"
	@$(KUSTOMIZE_BUILD) ./base >/dev/null
	@echo "${GREEN}Validating cluster overlay...${RESET}"
	@$(KUSTOMIZE_BUILD) ./overlays/cluster >/dev/null
	@echo "${GREEN}✓ Validation complete${RESET}"

diff: config ## Show kubectl diff for cluster overlay
	@$(KUSTOMIZE_BUILD) ./overlays/cluster | $(KUBECTL) diff -f - || true

deploy: deploy-cluster ## Deploy cluster overlay (default)

deploy-cluster: config ## Deploy cluster overlay
	@echo "${GREEN}Deploying cluster overlay...${RESET}"
	@$(KUSTOMIZE_BUILD) ./overlays/cluster | $(KUBECTL) apply -f -

status: ## Show montage-ai pod status
	@$(KUBECTL) get pods -n $(CLUSTER_NAMESPACE)

health: ## Basic health checks
	@$(KUBECTL) get pods -n $(CLUSTER_NAMESPACE)
	@$(KUBECTL) rollout status deployment/montage-ai-web -n $(CLUSTER_NAMESPACE) --timeout=120s || true
