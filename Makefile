code-health:
	@echo "Running vulture (dead-code detector)"
	vulture src --min-confidence 50 || true

.PHONY: code-health