## Unreleased

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
