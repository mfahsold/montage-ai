## 2026-02-09

### Fixes
- Fix: Dockerfile base image updated from Python 3.10 to 3.12 to match `pyproject.toml` `requires-python = ">=3.12,<3.13"` (#104). Also fixed `Dockerfile.qsv` (was 3.11).
- Fix: `deploy-cluster` now respects namespace override from `config-global.yaml` (#34). Removed hardcoded `namespace: montage-ai` from base kustomization; namespace is injected dynamically via `kustomize edit set namespace` at deploy time.
- Fix: CI installation on clean systems — added `cloud-private` optional dependency group and fixed `ci-local.sh` to handle it gracefully in both locked and non-locked sync paths (#35).
- Fix: Docker Web UI port mismatch — aligned all container ports from 5000 to 8080 across `docker-compose.yml`, `docker-compose.web.yml`, and `deploy/config.env` to match the Flask app default (#36).
- Fix: Kustomize ConfigMap hash suffix causing `CreateContainerConfigError` — added `generatorOptions.disableNameSuffixHash: true` so `envFrom.configMapRef` resolves correctly (#38).

### Documentation
- Added "Re-running Setup & Idempotency" section to `docs/getting-started.md` with idempotency guarantees, rebuild triggers, and full-reset instructions (#106).
- Updated `copilot-instructions.md` module table with correct module names (`audio_analysis.py`, `scene_analysis.py`, etc.) and added import examples (#105).
- Updated `docs/architecture.md` external dependencies: removed librosa (replaced by FFmpeg astats/tempo), corrected version pins, fixed GPU fallback chain.
- Expanded `docs/getting-started.md` from redirect stub to full standalone quickstart with Docker setup, prerequisites, and Kubernetes sections.
- Added "How to Set Configuration" intro and `WEB_PORT` to `docs/configuration.md`.
- Added port conflict diagnostics and `docker logs` hint to `docs/troubleshooting.md`.
- Removed duplicate environment variables in `docker-compose.yml`.

## Unreleased (prior)

### Fixes
- Fix: audio-analysis — handle very short/flat audio energy profiles without crashing in `detect_music_sections` (prevents IndexError on mismatched times/rms). Added unit + opt-in integration tests and updated documentation (recommended workaround: use WAV >= 3–4s for smoke runs). Contributed-by: mfahsold

### 🧠 SOTA 2025 Algorithmic Refactor

- **Reasoning Tree Selection**: Implemented "Tree-of-AdEditor" (ToAE) inspired branching decision logic in `ClipSelector`.
- **Human-Inspired Micro-Pacing**: Added "Breathing" logic to `PacingEngine` for non-robotic beat alignment (-40ms to +40ms jitter based on energy).
- **Beam Search Storytelling**: Upgraded `StorySolver` from greedy to Path Search for global narrative tension optimization.
- **Cinematic Heuristics**: Added shot progression (Wide -> Medium -> Close) and environmental continuity scoring.
- **DRY Refactoring**: Centralized all scoring weights and micro-timing constants in `EditingParameters`.

### Infrastructure & Cleanup

- Add `make code-health` target (vulture) and smoke test
- Add `docs/CODE_HEALTH.md` and small README note
- Remove obvious unused imports/params (api.py, web_ui/app.py, cli.py)
- Add `clean-deploy` ephemeral kustomize overlay and `dev-autoscale-smoke` CI workflow for safe dev/staging validation (KEDA/HPA); add docs and guarded dev-only endpoint for deterministic smoke tests
- Implement cluster-distributed deployment improvements: KEDA ScaledObjects for `montage-ai-worker` and `montage-ai-worker-heavy`, replaced host-based pod anti-affinity with zone-level spreading, and added a JobSet example for coordinated distributed renders (docs updated)
- Simplify deployments to two paths (local Docker + cluster overlay) and archive dev-specific overlays/flows

## 2026-01-15

### Highlights
- 3x performance improvements via lazy loading, lower startup time, and faster segment I/O.
- Default hardware acceleration paths for VAAPI (where available).
- Kubernetes job orchestration moved to the official Python API for reliability.
- Distributed scene detection sharding (file-based and time-based).
- Improved GPU scheduling with affinity rules and multi-arch image support.
