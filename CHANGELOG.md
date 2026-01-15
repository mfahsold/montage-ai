## Unreleased

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
