# ==============================================================================
# Montage-AI K3s/K8s - Canonical Makefile (Fluxibri-aligned)
# Single entry point for cluster operations
# ==============================================================================

.PHONY: help config validate diff deploy deploy-dev deploy-staging deploy-production deploy-app status health

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
	@echo "${GREEN}Validating production overlay...${RESET}"
	@$(KUSTOMIZE_BUILD) ./overlays/production >/dev/null
	@echo "${GREEN}✓ Validation complete${RESET}"

diff: config ## Show kubectl diff for production overlay
	@$(KUSTOMIZE_BUILD) ./overlays/production | $(KUBECTL) diff -f - || true

deploy: deploy-production ## Deploy production overlay (default)

deploy-dev: config ## Deploy dev overlay
	@echo "${GREEN}Deploying dev overlay...${RESET}"
	@$(KUSTOMIZE_BUILD) ./overlays/dev | $(KUBECTL) apply -f -

deploy-staging: config ## Deploy staging overlay
	@echo "${GREEN}Deploying staging overlay...${RESET}"
	@$(KUSTOMIZE_BUILD) ./overlays/staging | $(KUBECTL) apply -f -

deploy-production: config ## Deploy production overlay
	@echo "${GREEN}Deploying production overlay...${RESET}"
	@$(KUSTOMIZE_BUILD) ./overlays/production | $(KUBECTL) apply -f -

deploy-app: config ## Deploy full app manifests (ingress, monitoring)
	@echo "${GREEN}Deploying full app manifests...${RESET}"
	@$(KUSTOMIZE_BUILD) ./app | $(KUBECTL) apply -f -

status: ## Show montage-ai pod status
	@$(KUBECTL) get pods -n $(CLUSTER_NAMESPACE)

health: ## Basic health checks
	@$(KUBECTL) get pods -n $(CLUSTER_NAMESPACE)
	@$(KUBECTL) rollout status deployment/montage-ai-web -n $(CLUSTER_NAMESPACE) --timeout=120s || true
