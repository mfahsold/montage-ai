# Strategic Backlog - Montage AI

## CLI DRY/KISS Refactorings

### P1: Critical - Single Source of Truth

#### 1. Centralize Style Definitions
**Current:** Duplicated in cli.py, montage-ai.sh, style_templates.py, web_ui
**Target:** Single `styles.yaml` loaded by all components

```yaml
# src/montage_ai/styles.yaml
styles:
  dynamic:
    name: dynamic
    description: "Position-aware pacing (intro→build→climax→outro)"
    pacing: {speed: medium, min_cut: 0.8, max_cut: 3.0}
```

**Tasks:**
- [ ] Create `src/montage_ai/styles.yaml`
- [ ] Create `spec_loader.py` to parse YAML
- [ ] Update `style_templates.py` to use YAML
- [ ] Update `cli.py` to generate help from YAML
- [ ] Update `montage-ai.sh` to source styles from Python export

---

#### 2. Centralize Quality Profiles
**Current:** Scattered in env_mapper.py, montage-ai.sh, implicit in cli.py
**Target:** `ProfileConfig` dataclass with YAML backing

```python
# src/montage_ai/profiles.py
@dataclass
class QualityProfile:
    name: str
    preset: str  # ffmpeg preset
    crf: int
    resolution: tuple[int, int]
    stabilize: bool
    upscale: bool
```

**Tasks:**
- [ ] Create `src/montage_ai/profiles.py`
- [ ] Migrate profiles from env_mapper.py
- [ ] Add CLI command: `montage-ai config --export-profiles`
- [ ] Update montage-ai.sh to source from export

---

#### 3. Generate CLI from Spec
**Current:** cli.py has 6 options, montage-ai.sh has 20+
**Target:** Both generated from single YAML spec

**Tasks:**
- [ ] Create `cli_spec.yaml` with all options
- [ ] Create `cli_builder.py` to generate Click commands
- [ ] Deprecate manual Click decorators
- [ ] Update documentation auto-generation

---

### P2: Significant - Reduce Duplication

#### 4. Docker Command Builder
**Current:** Command construction duplicated in cli.py and montage-ai.sh
**Target:** Single `docker_utils.py`

**Tasks:**
- [ ] Create `src/montage_ai/docker_utils.py`
- [ ] Implement `build_docker_cmd(options: dict) -> List[str]`
- [ ] Use env_mapper.py for env var generation
- [ ] Update cli.py to use docker_utils
- [ ] Update montage-ai.sh to call Python builder

---

#### 5. Consistent Error Handling
**Current:** Mixed strategies (swallow, log, raise)
**Target:** Unified error handling policy

**Tasks:**
- [ ] Define error handling policy document
- [ ] Create `errors.py` with custom exceptions
- [ ] Audit all try/except blocks
- [ ] Implement consistent retry logic for I/O

---

### P3: Code Quality

#### 6. Remove Dead Code
**Current:** Local run mode in editor.py is `pass`
**Tasks:**
- [ ] Implement or remove local run mode
- [ ] Audit for other dead code paths
- [ ] Add coverage reporting to CI

---

#### 7. Output Formatter Utility
**Current:** Manual logging formatting scattered
**Target:** `OutputFormatter` class

**Tasks:**
- [ ] Create `src/montage_ai/output.py`
- [ ] Implement startup banner generator
- [ ] Implement enhancement settings formatter
- [ ] Unify CLI and shell output styles

---

## Performance Optimizations

### Scene Detection
- [ ] GPU-accelerated scene detection (PySceneDetect + CUDA)
- [ ] Adaptive threshold based on content type
- [ ] Incremental detection for long videos

### Rendering
- [ ] Hardware encoder auto-selection (NVENC > VAAPI > QSV > CPU)
- [ ] Parallel clip processing with GPU scheduling
- [ ] Memory-mapped I/O for large files

### Audio Analysis
- [ ] Librosa GPU acceleration (cupy backend)
- [ ] Cached beat detection per audio file
- [ ] Streaming analysis for long tracks

---

## Testing

- [ ] Add CLI integration tests
- [ ] Add progress callback unit tests
- [ ] Add profile loading tests
- [ ] Performance regression tests

---

## Documentation

- [ ] Auto-generate CLI reference from spec
- [ ] Update architecture diagram with new modules
- [ ] Add migration guide for breaking changes

---

*Last updated: 2026-01-10*
