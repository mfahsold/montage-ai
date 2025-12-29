# editor.py Decomposition Plan (COMPLETED)

**Status:** ✅ Completed (December 2025)

## Summary

The monolithic `editor.py` (~3500 lines) has been successfully decomposed into a modular, testable architecture.

## New Module Structure

```
src/montage_ai/
├── editor.py                 # Slim orchestrator (CLI wrapper)
├── config.py                 # ✅ Centralized configuration
├── core/                     # ✅ New Core Module
│   ├── montage_builder.py    # Pipeline orchestrator
│   ├── audio_analysis.py     # Beat detection & energy analysis
│   ├── scene_analysis.py     # Scene detection & AI analysis
│   ├── video_metadata.py     # Video metadata utilities
│   └── clip_enhancement.py   # Stabilize/enhance/upscale
├── creative_director.py      # LLM integration
└── cgpu_jobs/                # Cloud offloading
```

## Completed Phases

### Phase 1: Centralize Configuration ✅
- Created `src/montage_ai/config.py`
- Migrated `api.py` and `editor.py` to use `Settings` object

### Phase 2: Core Extraction ✅
- **2.1 Audio:** Extracted `audio_analysis.py` (librosa encapsulation)
- **2.2 Scenes:** Extracted `scene_analysis.py` (scenedetect encapsulation)
- **2.3 Metadata:** Extracted `video_metadata.py` (ffprobe encapsulation)
- **2.4 Enhancement:** Extracted `clip_enhancement.py` (stabilize/upscale logic)
- **2.5 Builder:** Created `MontageBuilder` pipeline and `MontageContext`

### Phase 3: Cleanup & Verification ✅
- **3.1 Verification:** End-to-End test script `verify_pipeline.py` passed
- **3.2 Legacy Removal:** Deleted `create_montage_legacy` (~1400 lines)
- **3.3 Documentation:** Updated architecture docs

## Metrics (After Refactoring)

| Metric | Before | After |
|--------|--------|-------|
| editor.py lines | 3566 | ~800 |
| Test Coverage | Low | High (Unit + E2E) |
| Architecture | Monolithic | Modular Pipeline |

---

## Original Plan (Archived)

### Current State Analysis (Dec 2025)

| Metric | Value |
|--------|-------|
| Total Lines | 3566 |
| Functions | 38 |
| Largest Function | create_montage() - 1442 lines |
| Direct env reads remaining | 4 (in determine_output_profile for runtime overrides) |

### Function Size Distribution (Top 10)
```
create_montage:          1442 lines  ⚠️ God Function
determine_output_profile:  142 lines
color_match_clips:         131 lines
analyze_scene_content:     100 lines
_upscale_with_realesrgan:   98 lines
_upscale_with_ffmpeg:       98 lines
find_best_start_point:      93 lines
interpret_creative_prompt:  84 lines
enhance_clip:               84 lines
calculate_dynamic_cut_length: 81 lines
```

## Proposed Module Structure

```
src/montage_ai/
├── editor.py                 # Slim orchestrator (main entry point)
├── config.py                 # ✅ Centralized configuration
├── video_metadata.py         # NEW: Video metadata utilities
├── scene_analysis.py         # NEW: Scene detection & AI analysis
├── audio_analysis.py         # NEW: Beat detection & energy analysis
├── clip_enhancement.py       # NEW: Stabilize/enhance/upscale
├── montage_builder.py        # NEW: Core montage assembly
├── creative_integration.py   # NEW: Creative Director integration
└── cgpu_jobs/                # ✅ Already extracted
```

## Module Specifications

### 1. video_metadata.py (~200 lines)

**Functions to extract:**
- `get_video_rotation(video_path) -> int`
- `ffprobe_video_metadata(video_path) -> dict`
- `determine_output_profile(video_files) -> dict`
- `apply_output_profile(profile) -> None`
- `build_video_ffmpeg_params(crf) -> list`

**Clean Interface:**
```python
class VideoMetadata:
    width: int
    height: int
    fps: float
    codec: str
    duration: float
    rotation: int

def analyze_video(path: str) -> VideoMetadata: ...
def determine_output_format(videos: List[str]) -> OutputProfile: ...
```

### 2. scene_analysis.py (~350 lines)

**Functions to extract:**
- `detect_scenes(video_path, threshold) -> List[Scene]`
- `analyze_scene_content(video_path, time_point) -> SceneAnalysis`
- `calculate_visual_similarity(frame1, frame2) -> float`
- `detect_motion_blur(video_path, time) -> float`
- `find_best_start_point(video_path, start, end, duration) -> float`

**Clean Interface:**
```python
@dataclass
class Scene:
    start: float
    end: float
    path: str
    meta: dict  # action, shot type, etc.

class SceneAnalyzer:
    def detect_scenes(self, video: str) -> List[Scene]: ...
    def analyze_content(self, video: str, time: float) -> dict: ...
```

### 3. audio_analysis.py (~200 lines)

**Functions to extract:**
- `analyze_music_energy(audio_path) -> EnergyProfile`
- `get_beat_times(audio_path) -> BeatInfo`
- `calculate_dynamic_cut_length(energy, tempo, position) -> float`

**Clean Interface:**
```python
@dataclass
class BeatInfo:
    tempo: float
    beat_times: List[float]
    duration: float

@dataclass
class EnergyProfile:
    envelope: np.ndarray
    avg: float
    max: float
    high_energy_pct: float

def analyze_audio(path: str) -> Tuple[BeatInfo, EnergyProfile]: ...
```

### 4. clip_enhancement.py (~500 lines)

**Functions to extract:**
- `stabilize_clip(input, output) -> str`
- `enhance_clip(input, output) -> str`
- `upscale_clip(input, output) -> str`
- `color_match_clips(clips, reference) -> dict`
- `enhance_clips_parallel(jobs) -> dict`

**Clean Interface:**
```python
class ClipEnhancer:
    def stabilize(self, clip: str) -> str: ...
    def enhance(self, clip: str) -> str: ...
    def upscale(self, clip: str, scale: int = 2) -> str: ...
    def process(self, clip: str, options: EnhanceOptions) -> str: ...
```

### 5. montage_builder.py (~1200 lines)

**The core create_montage() broken into phases:**

```python
@dataclass
class MontageContext:
    """Shared state for montage creation."""
    variant_id: int
    job_id: str
    settings: Settings
    editing_instructions: Optional[dict]
    monitor: Optional[Monitor]

    # Audio
    music_path: str
    tempo: float
    beat_times: List[float]
    energy_profile: EnergyProfile
    target_duration: float

    # Footage
    footage_pool: FootagePool
    all_scenes: List[Scene]

    # Output
    output_path: str
    clips_metadata: List[dict]

class MontageBuilder:
    def __init__(self, ctx: MontageContext):
        self.ctx = ctx

    def prepare_footage(self) -> List[str]:
        """Phase 1: Load and validate footage."""
        ...

    def analyze_clips(self, video_files: List[str]) -> List[Scene]:
        """Phase 2: Scene detection and AI analysis."""
        ...

    def build_timeline(self, scenes: List[Scene]) -> Timeline:
        """Phase 3: Beat-synced clip selection."""
        ...

    def process_clips(self, timeline: Timeline) -> List[ProcessedClip]:
        """Phase 4: Enhancement pipeline."""
        ...

    def render_output(self, clips: List[ProcessedClip]) -> str:
        """Phase 5: Final composition."""
        ...

    def create(self) -> str:
        """Main entry point."""
        videos = self.prepare_footage()
        scenes = self.analyze_clips(videos)
        timeline = self.build_timeline(scenes)
        processed = self.process_clips(timeline)
        return self.render_output(processed)

# Convenience function (backward compatible)
def create_montage(variant_id: int = 1) -> str:
    ctx = MontageContext.from_settings(variant_id)
    builder = MontageBuilder(ctx)
    return builder.create()
```

### 6. creative_integration.py (~100 lines)

**Functions to extract:**
- `interpret_creative_prompt() -> dict`
- `apply_style_template(instructions, settings) -> None`

## Implementation Phases

### Phase 2.1: audio_analysis.py (Low Risk)
- Extract audio functions
- Pure functions, no shared state
- Easy to test in isolation
- **Time: ~2 hours**

### Phase 2.2: scene_analysis.py (Low Risk)
- Extract scene detection
- Abstract LLM calls
- **Time: ~3 hours**

### Phase 2.3: video_metadata.py (Low Risk)
- Extract metadata utilities
- Already fairly self-contained
- **Time: ~2 hours**

### Phase 2.4: clip_enhancement.py (Medium Risk)
- Consolidate local and cloud enhancement
- Unify with cgpu_jobs
- **Time: ~4 hours**

### Phase 2.5: montage_builder.py (High Risk, High Value)
- Break create_montage into phases
- Create MontageContext
- Maintain backward compatibility
- **Time: ~8 hours**

## Testing Strategy

1. **Before extraction**: Ensure existing tests pass
2. **During extraction**: Write unit tests for each extracted module
3. **After extraction**: Integration tests to verify behavior unchanged

## Backward Compatibility

All extracted modules will be re-exported from editor.py:

```python
# editor.py (after refactoring)
from .audio_analysis import analyze_music_energy, get_beat_times
from .scene_analysis import detect_scenes, analyze_scene_content
from .video_metadata import determine_output_profile
from .clip_enhancement import stabilize_clip, enhance_clip, upscale_clip
from .montage_builder import create_montage

# Maintain all existing exports
__all__ = [
    'create_montage',
    'detect_scenes',
    'analyze_music_energy',
    # ... all existing exports
]
```

## Success Criteria

- [ ] editor.py reduced to < 500 lines
- [ ] create_montage() reduced to < 100 lines
- [ ] All existing tests pass
- [ ] No behavioral changes
- [ ] Clear module boundaries
- [ ] Improved testability
