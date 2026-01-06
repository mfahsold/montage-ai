# Refactoring Plan: MontageBuilder Decomposition

## Objectives
1.  **Decouple**: Break `MontageBuilder` (God Class) into specialized engines.
2.  **Type Safety**: Replace `Dict` with Pydantic models.
3.  **Modernize**: Remove legacy attribute mapping in `MontageContext`.

## Status
**Last Updated:** January 2026
**Progress:** Completed

## ✅ Completed Phases

### Phase 1: Preparation (Audit)
- [x] Create `src/montage_ai/core/models.py` (Pydantic models)
- [x] Diagram existing architecture
- [x] Fix Config leakage in `app.py`

### Phase 2: Analysis Engine Extraction
- [x] Create `AssetAnalyzer` in `src/montage_ai/core/analysis_engine.py`
- [x] Move audio analysis logic (`_analyze_music`, `_apply_voice_isolation`, etc.)
- [x] Move scene detection logic (`_detect_scenes`, `analyze_scene_content`)
- [x] Delegate calls from `MontageBuilder`

### Phase 3: Render Engine Extraction
- [x] Create `RenderEngine` in `src/montage_ai/core/render_engine.py`
- [x] Move `render_output` logic
- [x] Delegate calls from `MontageBuilder`

### Phase 4: Integration
- [x] Instantiate engines in `MontageBuilder.__init__`
- [x] Ensure `MontageContext` can be shared

### Phase 5: Context Purification
- [x] Identify usages of `_LEGACY_ATTR_MAP`
- [x] Refactor explicit paths (e.g. `ctx.media.all_scenes`) in all core files
- [x] Delete `_LEGACY_ATTR_MAP` and magic methods from `MontageContext`
- [x] Verify import and initialization via `tests/test_engines_integration.py`

### Phase 6: Pacing Engine Extraction
- [x] Create `PacingEngine` in `src/montage_ai/core/pacing_engine.py`
- [x] Move rhythm and timing logic (`_get_energy_at_time`, `_calculate_cut_duration`)
- [x] Delegate calls from `MontageBuilder`
- [x] Create unit tests (`tests/test_pacing_engine.py`)

### Phase 7: Deployment Cleanup
- [x] Delete stale `deploy/k3s/overlays/distributed/montage_builder.py`
- [x] Update Kustomize config to rely on immutable container images

### Phase 8: Selection Engine Extraction
- [x] Create `SelectionEngine` in `src/montage_ai/core/selection_engine.py`
- [x] Move scoring logic (`_score_action_energy`, `_select_clip`, etc.)
- [x] Move Intelligent Selector integration validation
- [x] Delegate calls from `MontageBuilder`
- [x] Create unit tests (`tests/test_selection_engine.py`)

## Next Steps (Future)
- [ ] Refactor `MontageTimeline` class itself to be a Pydantic model.
- [ ] Add type hints to `StoryEngine` (dependent on `MontageBuilder`).
