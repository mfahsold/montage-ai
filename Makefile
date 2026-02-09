code-health:
	@echo "Running vulture (dead-code detector)"
	vulture src --min-confidence 50 || true

test-docker-startup: ## Test that Docker container starts successfully
	@bash scripts/ci-docker-startup.sh

.PHONY: code-health test-docker-startup