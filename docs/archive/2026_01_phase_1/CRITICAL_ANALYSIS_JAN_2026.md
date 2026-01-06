# Critical Codebase Analysis (Jan 2026)

**Status:** Stable (Green)
**Tests:** 517 Passed, 0 Failed
**Focus:** Post-Refactor Stabilization & Roadmap Implementation

## 1. System Architecture Assessment

The recent refactor extracting `PacingEngine` and `SelectionEngine` from `MontageBuilder` has significantly improved modularity.

*   **PacingEngine**: Now fully owns timing logic. The new "Structure-Aware" pacing (Intro/Build/Drop/Outro) is implemented and state-independent where possible.
    *   *Risk*: The "Build" curve logic relies on `current_time` relative to section start. If the timeline is assembled non-linearly (rare in this app), curves might be discontinuous.
*   **SelectionEngine**: Now owns scoring logic. New signals (Faces, Visual Novelty) are integrated.
    *   *Implementation Note*: "Visual Novelty" currently relies on **Semantic Tag Overlap** rather than Color Histograms. This is a lightweight proxy that prevents semantic repetition (e.g., "dog", "dog") but may miss visual repetition (e.g., "red car", "red car" might both be tagged "car").
*   **MontageBuilder**: Reduced to an orchestrator. Still handles some state management (`ctx.timeline`) which is appropriate.

## 2. Roadmap Implementation Status

| Feature | Status | Implementation Details |
| :--- | :--- | :--- |
| **Advanced Pacing** | ✅ Done | `PacingEngine` detects `build`/`drop` labels from `AudioAnalysis` and adjusts beat counts (8 -> 4 -> 2 -> 1). |
| **Smart Selection** | ✅ Done | `SelectionEngine` scores `face_count` (+10) and penalizes tag overlap (-30). |
| **Templates** | ✅ Done | `vlog.json` and `sport.json` leverage new weighting system. |

## 3. Technical Debt & Future Risks

1.  **Face Detection Accuracy**:
    *   Current: Uses standard OpenCV detection (`detect_faces` in `scene_analysis.py`).
    *   Risk: High false positive rate on non-human patterns or low light.
    *   Recommendation: Unify with MediaPipe usage from `AutoReframeEngine`.

2.  **Visual Novelty Constraints**:
    *   Current: Tag-based.
    *   Risk: Scenes with no tags (if AI filter fails) get score 0.0 variance.
    *   Recommendation: Implement lightweight Color Histogram extraction in `SceneContentAnalyzer` for V3.

3.  **State Management**:
    *   `MontageTimeline` is becoming a "God Object" for state.
    *   Recommendation: Monitor size of `ctx.timeline`.

## 4. Conclusion

The codebase is in a robust state. The "Polish, Don't Generate" philosophy is maintained. The new features add significant creative intelligence without destabilizing the core pipeline.
