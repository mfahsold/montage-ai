# Montage AI - Development Makefile
# Lightweight dev workflow for local and cluster deployments

.PHONY: help build push deploy test clean logs shell ci

# Configuration
IMAGE_NAME ?= ghcr.io/mfahsold/montage-ai
IMAGE_TAG ?= latest
REGISTRY ?= ghcr.io/mfahsold
# Default: GHCR for reliable public registry (multi-arch)
# For cluster-only deployments, override with CLUSTER_REGISTRY=192.168.1.12:30500
CLUSTER_REGISTRY ?= ghcr.io/mfahsold
NAMESPACE ?= montage-ai
PLATFORM ?= linux/amd64
PLATFORMS ?= linux/amd64,linux/arm64
CACHE_DIR ?= /tmp/buildx-cache

# Colors
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

help: ## Show this help
	@echo "$(CYAN)Montage AI - Golden Path$(RESET)"
	@echo ""
	@echo "$(GREEN)📖 Read First: deploy/README.md$(RESET)"
	@echo ""
	@echo "$(GREEN)Local Development (5 sec feedback):$(RESET)"
	@echo "  make dev            Build base image with cache (once)"
	@echo "  make dev-test       Run with volume mounts (instant)"
	@echo "  make dev-shell      Interactive shell for debugging"
	@echo ""
	@echo "$(GREEN)Cluster Deployment (2-15 min):$(RESET)"
	@echo "  make cluster           Build + push + deploy (all-in-one)"
	@echo "  make cluster-build     Multi-arch build (ARM64+AMD64) (set BUILDKIT_SSH=1 to enable BuildKit SSH forwarding)"
	@echo "  make cluster-build-fast Single-arch quick build"
	@echo ""
	@echo "$(GREEN)Builder Management:$(RESET)"
	@echo "  make builder-setup  Setup distributed builder"
	@echo "  make builder-status Show builder status"
	@echo ""
	@echo "$(GREEN)CI / Validation:$(RESET)"
	@echo "  make ci             Run vendor-agnostic CI checks locally"
	@echo ""
	@echo "$(GREEN)Releases:$(RESET)"
	@echo "  make release-ghcr   Build + push to GitHub Container Registry"
	@echo ""
	@echo "$(GREEN)Maintenance:$(RESET)"
	@echo "  make clean          Clean local caches"
	@echo "  make test           Run all tests"
	@echo ""
	@echo "$(YELLOW)Legacy commands available, see: make help-all$(RESET)"

# ============================================================================
# WEB UI (SELF-HOSTED)
# ============================================================================

web: build ## Start web UI (http://localhost:5000)
	@echo "$(CYAN)Starting Montage AI Web UI...$(RESET)"
	@echo "$(GREEN)Web interface: http://localhost:5000$(RESET)"
	docker-compose -f docker-compose.web.yml up

web-bg: build ## Start web UI in background
	@echo "$(CYAN)Starting Montage AI Web UI (background)...$(RESET)"
	docker-compose -f docker-compose.web.yml up -d
	@echo "$(GREEN)Web interface: http://localhost:5000$(RESET)"

web-stop: ## Stop web UI
	docker-compose -f docker-compose.web.yml down

web-deploy: ## Deploy web UI to Kubernetes
	@echo "$(CYAN)Deploying Web UI to Kubernetes...$(RESET)"
	kubectl apply -f deploy/k3s/base/web-service.yaml
	@echo "$(GREEN)Web UI deployed. Check service: kubectl get svc -n $(NAMESPACE) montage-ai-web$(RESET)"

# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================

# Get current git commit hash for version tracking
GIT_COMMIT := $(shell git rev-parse --short=8 HEAD 2>/dev/null || echo "dev")

# ============================================================================
# GOLDEN PATH: LOCAL DEVELOPMENT
# ============================================================================

dev: ## Build base image with local cache (run once)
	@echo "$(CYAN)Building base image with local cache...$(RESET)"
	@echo "$(YELLOW)First build: 8-12 min | Subsequent: ~1 min$(RESET)"
	CACHE_DIR=$(CACHE_DIR) TAG=dev LOAD=true ./scripts/build_local_cache.sh
	@echo "$(GREEN)✓ Ready for development!$(RESET)"
	@echo "$(GREEN)→ Run: make dev-test$(RESET)"

dev-test: ## Run with volume mounts (instant code changes)
	@echo "$(CYAN)Running with volume mounts (instant reload)...$(RESET)"
	docker run --rm -it \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/data:/data \
		montage-ai:dev \
		montage-ai run --style dynamic

dev-shell: ## Interactive shell for debugging
	@echo "$(CYAN)Opening interactive shell...$(RESET)"
	docker run --rm -it \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/data:/data \
		--entrypoint /bin/bash \
		montage-ai:dev

# ============================================================================
# GOLDEN PATH: CLUSTER DEPLOYMENT
# ============================================================================

cluster: cluster-build cluster-deploy ## Build + deploy to cluster (all-in-one)

cluster-build: ## Build multi-arch with distributed builder + registry cache
	@echo "$(CYAN)Building for cluster (multi-arch, distributed)...$(RESET)"
	@echo "$(YELLOW)First build: 15-20 min | Incremental: ~2 min$(RESET)"
	REGISTRY=$(CLUSTER_REGISTRY) TAG=$(IMAGE_TAG) ./scripts/build-distributed.sh
	@echo "$(GREEN)✓ Image pushed to registry$(RESET)"

cluster-build-fast: ## Quick single-arch build for current platform
	@echo "$(CYAN)Building for cluster (single-arch, fast)...$(RESET)"
	REGISTRY=$(CLUSTER_REGISTRY) TAG=$(IMAGE_TAG) PLATFORMS=linux/amd64 ./scripts/build-distributed.sh
	@echo "$(GREEN)✓ Image pushed to registry$(RESET)"

cluster-deploy: ## Deploy to Kubernetes cluster
	@echo "$(CYAN)Deploying to cluster...$(RESET)"
	kubectl set image deployment/montage-ai-worker \
		montage-ai=$(CLUSTER_REGISTRY)/montage-ai:$(IMAGE_TAG) \
		-n $(NAMESPACE)
	kubectl rollout status deployment/montage-ai-worker -n $(NAMESPACE)
	@echo "$(GREEN)✓ Deployment complete$(RESET)"

# ============================================================================
# BUILDER SETUP
# ============================================================================

builder-setup: ## Setup distributed builder (parallel ARM64+AMD64)
	@echo "$(CYAN)Setting up distributed builder...$(RESET)"
	@./scripts/setup_distributed_builder.sh
	@echo "$(GREEN)✓ Distributed builder ready$(RESET)"

builder-status: ## Show buildx builder status
	@echo "$(CYAN)Builder Status$(RESET)"
	@docker buildx ls
	@echo ""
	@echo "Active builder platforms:"
	@docker buildx inspect --bootstrap 2>/dev/null | grep -E "Platforms|Name|Driver" || echo "No active builder"

# ============================================================================
# RELEASES (GitHub Container Registry)
# ============================================================================

release-ghcr: ## Build + push to GitHub Container Registry
	@echo "$(CYAN)Building for GHCR release...$(RESET)"
	REGISTRY=ghcr.io/mfahsold TAG=$(IMAGE_TAG) ./scripts/build-distributed.sh
	@echo "$(GREEN)✓ Image pushed to ghcr.io/mfahsold/montage-ai:$(IMAGE_TAG)$(RESET)"

# ============================================================================
# MAINTENANCE
# ============================================================================

clean: ## Clean local caches and Docker resources
	@echo "$(CYAN)Cleaning local resources...$(RESET)"
	rm -rf $(CACHE_DIR)
	docker compose down -v 2>/dev/null || true
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	docker buildx prune -f
	@echo "$(GREEN)✓ Caches cleared$(RESET)"

.PHONY: cleanup
cleanup: ## Archive large proxies, compress old monitoring JSONs, rotate logs
	@echo "$(CYAN)Running cleanup script...$(RESET)"
	./scripts/cleanup.sh

.PHONY: registry-check
registry-check: ## Run registry health checks against default host (192.168.1.12)
	@echo "$(CYAN)Running registry checks...$(RESET)"
	./scripts/registry_check.sh

validate-deps: ## Validate optional dependencies installation
	@echo "$(CYAN)Validating optional dependencies...$(RESET)"
	@./scripts/validate_optional_deps.sh

.PHONY: deps-lock
deps-lock: ## Generate/refresh uv.lock using uv
	@echo "$(CYAN)Generating uv.lock...$(RESET)"
	@./scripts/uv-lock.sh

benchmark: ## Run performance baseline benchmarks
	@echo "$(CYAN)Running performance baseline...$(RESET)"
	@python3 scripts/benchmark_baseline.py

# ============================================================================
# LEGACY COMMANDS (use golden path above)
# ============================================================================

build: ## Build Docker image for local architecture
	@echo "$(YELLOW)⚠ Legacy command. Use 'make dev' instead.$(RESET)"
	docker build --build-arg GIT_COMMIT=$(GIT_COMMIT) -t $(IMAGE_NAME):$(IMAGE_TAG) .

build-multiarch: ## [LEGACY] Use 'make cluster' instead
	@echo "$(YELLOW)⚠ Legacy command. Use 'make cluster' instead.$(RESET)"
	docker buildx build --platform linux/amd64,linux/arm64 \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--cache-from type=registry,ref=$(IMAGE_NAME):build-cache \
		--cache-to type=registry,ref=$(IMAGE_NAME):build-cache,mode=max \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		--push .

build-amd64: ## [LEGACY] Use 'make cluster' instead
	@echo "$(YELLOW)⚠ Legacy command. Use 'make cluster' instead.$(RESET)"
	docker buildx build --platform linux/amd64 \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		--load .

run: build ## [LEGACY] Use 'make dev-test' instead
	@echo "$(YELLOW)⚠ Legacy command. Use 'make dev-test' instead.$(RESET)"
	./montage-ai.sh run

preview: build ## [LEGACY] Use 'make dev-test' instead
	@echo "$(YELLOW)⚠ Legacy command. Use 'make dev-test' instead.$(RESET)"
	./montage-ai.sh preview

hq: build ## [LEGACY] Use 'make dev-test' instead
	@echo "$(YELLOW)⚠ Legacy command. Use 'make dev-test' instead.$(RESET)"
	./montage-ai.sh hq

shell: ## [LEGACY] Use 'make dev-shell' instead
	@echo "$(YELLOW)⚠ Legacy command. Use 'make dev-shell' instead.$(RESET)"
	docker compose run --rm --entrypoint /bin/bash montage-ai

push: build-amd64 ## [LEGACY] Use 'make cluster' instead
	@echo "$(YELLOW)⚠ Legacy command. Use 'make cluster' instead.$(RESET)"
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/montage-ai:$(IMAGE_TAG)
	docker push $(REGISTRY)/montage-ai:$(IMAGE_TAG)

push-ghcr: build-amd64 ## Push image to GitHub Container Registry
	@echo "$(CYAN)Pushing to ghcr.io...$(RESET)"
	docker push $(IMAGE_NAME):$(IMAGE_TAG)

deploy: ## Deploy base resources to Kubernetes
	@echo "$(CYAN)Deploying montage-ai to Kubernetes...$(RESET)"
	kubectl apply -k deploy/k3s/base/

deploy-prod: ## Deploy production overlay (AMD GPU node)
	@echo "$(CYAN)Deploying montage-ai (production)...$(RESET)"
	kubectl apply -k deploy/k3s/overlays/production/

deploy-dev: ## Deploy dev overlay (fast preview)
	@echo "$(CYAN)Deploying montage-ai (dev)...$(RESET)"
	kubectl apply -k deploy/k3s/overlays/dev/

job: ## Start a render job in cluster
	@echo "$(CYAN)Starting render job...$(RESET)"
	kubectl delete job -n $(NAMESPACE) montage-ai-render 2>/dev/null || true
	kubectl apply -f deploy/k3s/base/job.yaml
	@echo "$(GREEN)Job started. Use 'make logs' to watch progress.$(RESET)"

logs: ## View job logs
	@echo "$(CYAN)Fetching logs for most recent job...$(RESET)"
	@# Try to find the most recent job pod
	@POD=$$(kubectl get pods -n $(NAMESPACE) --sort-by=.metadata.creationTimestamp -o name | grep montage-ai-distributed | tail -1); \
	if [ -z "$$POD" ]; then \
		echo "$(YELLOW)No running jobs found. Try 'make job' first.$(RESET)"; \
	else \
		echo "$(GREEN)Streaming logs from $$POD...$(RESET)"; \
		kubectl logs -n $(NAMESPACE) -f $$POD; \
	fi

status: ## Show cluster deployment status
	@echo "$(CYAN)Montage AI Cluster Status$(RESET)"
	@echo ""
	@kubectl get all,pvc -n $(NAMESPACE) 2>/dev/null || echo "Namespace not found"

# ============================================================================
# TESTING
# ============================================================================

test: validate test-local test-unit ## Run all tests
	@echo "$(GREEN)All tests passed!$(RESET)"

ci-local: ## Run CI locally (uv-based). Avoids GitHub Actions costs.
	@echo "$(CYAN)Running local CI (uv)...$(RESET)"
	@./scripts/ci-local.sh
	@echo "$(GREEN)Local CI finished$(RESET)"

test-unit: ## Run unit tests with pytest
	@echo "$(CYAN)Running unit tests...$(RESET)"
	PYTHONPATH=src pytest tests/ -v --ignore=tests/integration/
	@echo "$(GREEN)✓ Unit tests passed$(RESET)"

test-assets: ## Download NASA test footage (public domain)
	@echo "$(CYAN)Downloading NASA test assets...$(RESET)"
	python scripts/prepare_trailer_assets.py --videos 3 --max-video-mb 50
	@echo "$(GREEN)✓ Test assets downloaded to data/input/ and data/music/$(RESET)"

test-fixtures: ## Generate synthetic test fixtures (no download)
	@echo "$(CYAN)Generating synthetic test fixtures...$(RESET)"
	@mkdir -p tests/fixtures/video tests/fixtures/audio
	@# Generate 5-second 1080p test video with color bars
	ffmpeg -y -f lavfi -i "testsrc=duration=5:size=1920x1080:rate=30" \
		-f lavfi -i "sine=frequency=440:duration=5" \
		-c:v libx264 -preset ultrafast -crf 23 -c:a aac -b:a 128k \
		tests/fixtures/video/test_1080p_5s.mp4 2>/dev/null
	@# Generate 10-second 120 BPM beat track
	ffmpeg -y -f lavfi -i "sine=frequency=80:duration=0.1,apad=pad_dur=0.4[kick]; \
		sine=frequency=200:duration=0.05,apad=pad_dur=0.45[snare]; \
		[kick][snare]amix=inputs=2,aloop=loop=20:size=22050" \
		-t 10 -ar 44100 tests/fixtures/audio/test_120bpm_10s.wav 2>/dev/null
	@echo "$(GREEN)✓ Fixtures created in tests/fixtures/$(RESET)"

clean-data: ## Remove all downloaded media (saves ~1GB)
	@echo "$(YELLOW)Removing downloaded media...$(RESET)"
	rm -rf data/input/archive data/music/archive
	rm -rf data/output/*
	@echo "$(GREEN)✓ Media cleaned$(RESET)"

test-local: ## Test local Docker workflow
	@echo "$(CYAN)Testing local Docker workflow...$(RESET)"
	@./montage-ai.sh build
	@echo "$(GREEN)✓ Docker build successful$(RESET)"

test-k8s: validate ## Test Kubernetes manifests
	@echo "$(CYAN)Testing Kubernetes deployment...$(RESET)"
	kubectl apply -k deploy/k3s/base/ --dry-run=client
	kubectl apply -k deploy/k3s/overlays/production/ --dry-run=client
	kubectl apply -k deploy/k3s/overlays/dev/ --dry-run=client
	@echo "$(GREEN)✓ All manifests valid$(RESET)"

validate: ## Validate all Kustomize manifests
	@echo "$(CYAN)Validating manifests...$(RESET)"
	@kubectl kustomize deploy/k3s/base/ > /dev/null && echo "$(GREEN)✓ base/$(RESET)"
	@kubectl kustomize deploy/k3s/overlays/production/ > /dev/null && echo "$(GREEN)✓ overlays/production/$(RESET)"
	@kubectl kustomize deploy/k3s/overlays/dev/ > /dev/null && echo "$(GREEN)✓ overlays/dev/$(RESET)"

# ==========================================================================
# CI (vendor-agnostic)
# ==========================================================================

ci: ## Run CI locally (no GitHub Actions required)
	@echo "$(CYAN)Running vendor-agnostic CI pipeline...$(RESET)"
	./scripts/ci.sh
	@echo "$(GREEN)✓ CI checks completed$(RESET)"

clean-k8s: ## Clean up Kubernetes resources
	@echo "$(YELLOW)Deleting montage-ai namespace...$(RESET)"
	kubectl delete namespace $(NAMESPACE) 2>/dev/null || true

# ============================================================================
# RELEASE (for maintainers)
# ============================================================================

release: validate ## Create a release (builds, tags, pushes)
	@if [ -z "$(VERSION)" ]; then echo "Usage: make release VERSION=v1.0.0"; exit 1; fi
	@echo "$(CYAN)Creating release $(VERSION) (commit: $(GIT_COMMIT))...$(RESET)"
	docker buildx build --platform linux/amd64,linux/arm64 \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--cache-from type=registry,ref=$(IMAGE_NAME):build-cache \
		--cache-to type=registry,ref=$(IMAGE_NAME):build-cache,mode=max \
		-t $(IMAGE_NAME):$(VERSION) \
		-t $(IMAGE_NAME):latest \
		--push .
	@echo "$(GREEN)Release $(VERSION) published!$(RESET)"
