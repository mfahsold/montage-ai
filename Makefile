# Montage AI - Development Makefile
# Lightweight dev workflow for local and cluster deployments

.PHONY: help build push deploy test clean logs shell

# Configuration
IMAGE_NAME ?= ghcr.io/mfahsold/montage-ai
IMAGE_TAG ?= latest
REGISTRY ?= 192.168.1.12:5000
NAMESPACE ?= montage-ai
PLATFORM ?= linux/amd64

# Colors
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

help: ## Show this help
	@echo "$(CYAN)Montage AI - Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Web UI (Self-Hosted):$(RESET)"
	@echo "  make web            Start web UI (http://localhost:5000)"
	@echo "  make web-deploy     Deploy web UI to K8s cluster"
	@echo ""
	@echo "$(GREEN)Local Development:$(RESET)"
	@echo "  make build          Build Docker image (local arch)"
	@echo "  make run            Run montage locally"
	@echo "  make preview        Fast preview render"
	@echo "  make hq             High-quality render"
	@echo "  make shell          Interactive shell in container"
	@echo ""
	@echo "$(GREEN)Cluster Deployment:$(RESET)"
	@echo "  make build-amd64    Build for amd64 (cluster)"
	@echo "  make push           Push to local registry"
	@echo "  make push-ghcr      Push to GitHub Container Registry"
	@echo "  make deploy         Deploy to K8s cluster (base)"
	@echo "  make deploy-prod    Deploy to K8s cluster (production)"
	@echo "  make deploy-dev     Deploy to K8s cluster (dev)"
	@echo "  make job            Start render job in cluster"
	@echo "  make logs           View job logs"
	@echo "  make status         Show cluster status"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  make test           Run all tests"
	@echo "  make test-local     Test local Docker workflow"
	@echo "  make test-k8s       Test Kubernetes deployment"
	@echo ""
	@echo "$(GREEN)Maintenance:$(RESET)"
	@echo "  make clean          Clean up local resources"
	@echo "  make clean-k8s      Clean up Kubernetes resources"
	@echo "  make validate       Validate all manifests"

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

build: ## Build Docker image for local architecture
	@echo "$(CYAN)Building montage-ai (local arch, commit: $(GIT_COMMIT))...$(RESET)"
	docker build --build-arg GIT_COMMIT=$(GIT_COMMIT) -t $(IMAGE_NAME):$(IMAGE_TAG) .

build-multiarch: ## Build Docker image for amd64 and arm64
	@echo "$(CYAN)Building montage-ai for linux/amd64,linux/arm64 (commit: $(GIT_COMMIT))...$(RESET)"
	docker buildx build --platform linux/amd64,linux/arm64 \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--cache-from type=registry,ref=$(IMAGE_NAME):build-cache \
		--cache-to type=registry,ref=$(IMAGE_NAME):build-cache,mode=max \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		--push .

build-amd64: ## Build Docker image for amd64 (cluster deployment)
	@echo "$(CYAN)Building montage-ai for linux/amd64 (commit: $(GIT_COMMIT))...$(RESET)"
	docker buildx build --platform linux/amd64 \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		--load .

run: build ## Run montage with default settings
	@echo "$(CYAN)Running montage-ai...$(RESET)"
	./montage-ai.sh run

preview: build ## Fast preview render
	./montage-ai.sh preview

hq: build ## High-quality render
	./montage-ai.sh hq

shell: ## Interactive shell in container
	docker compose run --rm --entrypoint /bin/bash montage-ai

# ============================================================================
# CLUSTER DEPLOYMENT
# ============================================================================

push: build-amd64 ## Push image to local registry
	@echo "$(CYAN)Pushing to local registry $(REGISTRY)...$(RESET)"
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

# ============================================================================
# MAINTENANCE
# ============================================================================

clean: ## Clean up local Docker resources
	@echo "$(CYAN)Cleaning up local resources...$(RESET)"
	docker compose down -v 2>/dev/null || true
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true

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
