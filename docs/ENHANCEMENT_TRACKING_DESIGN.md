# NLE-Compatible Enhancement Tracking Design

## Problem Statement

Currently, all AI-driven enhancements in Montage AI are **baked into** the rendered output:
- Denoising, sharpening, color grading are applied directly to video files
- Parameters used are lost after rendering
- NLE import shows only the final result, not the decisions
- Professional editors cannot adjust or undo AI decisions
- Re-edits require re-running Montage AI from scratch

**Goal:** Make every AI edit traceable, exportable, and re-creatable in professional NLEs.

---

## Architecture: Enhancement Decision Layer

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Montage AI Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│  │  Analysis  │───▶│ EnhancementDecider│───▶│ EnhancementTracker│   │
│  │  Engine    │    │ (AI recommends)   │    │ (Stores decisions)│   │
│  └────────────┘    └──────────────────┘    └────────┬─────────┘    │
│                                                      │              │
│                                                      ▼              │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │              EnhancementDecisionList                      │      │
│  │  [                                                        │      │
│  │    ClipEnhancement(clip_id, denoise=..., color=...),     │      │
│  │    ClipEnhancement(clip_id, stabilize=..., sharpen=...), │      │
│  │    ...                                                    │      │
│  │  ]                                                        │      │
│  └──────────────────────────────────────────────────────────┘      │
│                           │                                         │
│           ┌───────────────┼───────────────┐                        │
│           ▼               ▼               ▼                        │
│  ┌────────────┐   ┌────────────┐   ┌─────────────┐                │
│  │  RENDER    │   │  EXPORT    │   │  EXPORT     │                │
│  │  (apply    │   │  TIMELINE  │   │  RECIPE     │                │
│  │  effects)  │   │  + metadata│   │  CARDS      │                │
│  └────────────┘   └────────────┘   └─────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### 1. EnhancementDecision (per-clip)

```python
@dataclass
class EnhancementDecision:
    """Tracks all enhancement decisions for a single clip."""
    clip_id: str  # Unique identifier (file path + timecode hash)
    source_path: str
    timeline_in: float
    timeline_out: float

    # Enhancement flags (what was applied)
    stabilized: bool = False
    upscaled: bool = False
    denoised: bool = False
    sharpened: bool = False
    color_graded: bool = False
    color_matched: bool = False
    film_grain_added: bool = False

    # Enhancement parameters (for recreation)
    stabilize_params: Optional[StabilizeParams] = None
    upscale_params: Optional[UpscaleParams] = None
    denoise_params: Optional[DenoiseConfig] = None
    sharpen_params: Optional[SharpenConfig] = None
    color_grade_params: Optional[ColorGradeParams] = None
    color_match_params: Optional[ColorMatchParams] = None
    film_grain_params: Optional[FilmGrainConfig] = None

    # Source analysis (why these decisions were made)
    analysis: Optional[ClipAnalysis] = None
    ai_reasoning: Optional[str] = None  # LLM explanation

    # Timestamps
    decided_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
```

### 2. Supporting Parameter Classes

```python
@dataclass
class StabilizeParams:
    method: str  # "vidstab", "deshake", "cgpu"
    smoothing: int = 30
    crop_mode: str = "black"  # "black", "crop", "fill"
    zoom: float = 0.0

    def to_resolve_node(self) -> str:
        """Generate DaVinci Resolve stabilizer node parameters."""
        return f"Stabilizer: Smoothing={self.smoothing}, Crop={self.crop_mode}"

    def to_premiere_effect(self) -> str:
        """Generate Premiere Pro Warp Stabilizer parameters."""
        return f"Warp Stabilizer: Smoothness={self.smoothing}%, Method=Position"

@dataclass
class UpscaleParams:
    method: str  # "realesrgan", "lanczos", "cgpu"
    scale_factor: int = 2
    model: str = "realesrgan-x4plus"

    def to_resolve_node(self) -> str:
        return f"Super Scale: {self.scale_factor}x (Enhanced)"

    def to_premiere_effect(self) -> str:
        return f"Scale: {self.scale_factor * 100}% with Bicubic Sharper"

@dataclass
class ColorGradeParams:
    preset: str  # "cinematic", "teal_orange", etc.
    intensity: float = 0.7
    lut_path: Optional[str] = None

    # Breakdown for NLE recreation
    lift: Tuple[float, float, float] = (0, 0, 0)
    gamma: Tuple[float, float, float] = (1, 1, 1)
    gain: Tuple[float, float, float] = (1, 1, 1)
    saturation: float = 1.0
    contrast: float = 1.0

    def to_resolve_node(self) -> str:
        """Generate DaVinci Resolve Color Wheels parameters."""
        return (
            f"Primary Wheels:\n"
            f"  Lift: R={self.lift[0]:.2f} G={self.lift[1]:.2f} B={self.lift[2]:.2f}\n"
            f"  Gamma: R={self.gamma[0]:.2f} G={self.gamma[1]:.2f} B={self.gamma[2]:.2f}\n"
            f"  Gain: R={self.gain[0]:.2f} G={self.gain[1]:.2f} B={self.gain[2]:.2f}\n"
            f"  Saturation: {self.saturation:.0%}\n"
            f"  Contrast: {self.contrast:.0%}"
        )

    def to_premiere_lumetri(self) -> str:
        """Generate Premiere Pro Lumetri Color parameters."""
        return (
            f"Lumetri Color:\n"
            f"  Creative > Look: {self.preset}\n"
            f"  Basic > Saturation: {self.saturation * 100:.0f}\n"
            f"  Basic > Contrast: {(self.contrast - 1) * 100:.0f}"
        )

@dataclass
class ColorMatchParams:
    reference_clip: str
    method: str = "mkl"  # "mkl", "hm", "mvgd"
    r_adjustment: float = 0.0
    g_adjustment: float = 0.0
    b_adjustment: float = 0.0
```

---

## Export Strategy

### 1. OTIO Metadata Extension

OpenTimelineIO supports custom metadata per clip. We extend this:

```python
def export_enhancement_to_otio(clip: otio.schema.Clip, decision: EnhancementDecision):
    """Embed enhancement decisions into OTIO clip metadata."""
    clip.metadata["montage_ai"] = {
        "version": "1.0",
        "enhancements": {
            "stabilized": decision.stabilized,
            "upscaled": decision.upscaled,
            "denoised": decision.denoised,
            "color_graded": decision.color_graded,
        },
        "params": {
            "denoise": asdict(decision.denoise_params) if decision.denoise_params else None,
            "sharpen": asdict(decision.sharpen_params) if decision.sharpen_params else None,
            "color_grade": asdict(decision.color_grade_params) if decision.color_grade_params else None,
            # ...
        },
        "ai_reasoning": decision.ai_reasoning,
    }
```

### 2. EDL Comments (CMX 3600)

EDL supports comments that most NLEs preserve:

```
001  AX       V     C        00:00:00:00 00:00:05:15 00:00:00:00 00:00:05:15
* FROM CLIP NAME: DJI_0042.MP4
* SOURCE FILE: /media/footage/DJI_0042.MP4
* MONTAGE_AI STABILIZE: vidstab smoothing=30 crop=black
* MONTAGE_AI DENOISE: hqdn3d spatial=0.3 temporal=0.5
* MONTAGE_AI COLOR_GRADE: cinematic intensity=0.7
* MONTAGE_AI REASONING: High motion detected, applied stabilization
```

### 3. FCP XML Effects (Native Re-Creation)

For Final Cut Pro / Resolve XML import, we can embed actual effect nodes:

```xml
<clipitem id="clip-1">
  <name>DJI_0042.MP4</name>
  <!-- ... timecodes ... -->

  <filter>
    <effect>
      <name>Warp Stabilizer</name>
      <effectid>AdjustmentLayer</effectid>
      <parameter>
        <parameterid>Smoothness</parameterid>
        <value>30</value>
      </parameter>
    </effect>
  </filter>

  <filter>
    <effect>
      <name>Lumetri Color</name>
      <effectid>Color Correction</effectid>
      <parameter>
        <parameterid>Saturation</parameterid>
        <value>115</value>
      </parameter>
    </effect>
  </filter>

  <!-- Montage AI metadata as labels -->
  <labels>
    <label2>AI_STABILIZED</label2>
  </labels>
  <comments>
    <mastercomment1>MONTAGE_AI: denoise=hqdn3d(0.3,0.5) sharpen=unsharp(0.4)</mastercomment1>
  </comments>
</clipitem>
```

### 4. Recipe Card Export (Human-Readable)

For parameters that can't map directly to NLE effects, generate a "recipe card":

```markdown
# Montage AI Enhancement Recipe - gallery_montage_v1

## Clip 1: DJI_0042.MP4 (00:00:00 - 00:05:15)

### Applied Enhancements:
- [x] Stabilization (vidstab 2-pass)
- [x] Denoising (hqdn3d)
- [x] Color Grading (cinematic)
- [ ] Upscaling

### DaVinci Resolve Recreation:
1. **Stabilizer** (Color Page > Tracker > Stabilizer)
   - Mode: Perspective
   - Smooth: 0.30
   - Crop Ratio: 1.0
   - Zoom: 0%

2. **Noise Reduction** (Color Page > Spatial NR)
   - Luma Threshold: 3.0
   - Luma Softness: 0.5
   - Chroma Threshold: 4.0

3. **Color Wheels** (Color Page > Primary Wheels)
   - Lift: R=0.00 G=0.00 B=+0.02
   - Gamma: R=0.98 G=1.00 B=1.02
   - Gain: R=1.02 G=0.98 B=0.95
   - Saturation: 115%

### Premiere Pro Recreation:
1. **Warp Stabilizer**
   - Result: Smooth Motion
   - Smoothness: 30%
   - Method: Position

2. **Lumetri Color > Creative**
   - Look: Cinematic
   - Intensity: 70%
   - Saturation: 115

### AI Reasoning:
> "Applied stabilization due to high camera motion (shake score: 0.72).
> Denoising added for low-light footage (ISO estimated: 3200).
> Cinematic color grade matches scene mood: dramatic."
```

---

## Implementation Phases

### Phase 1: Data Model (This PR)
- [ ] Create `EnhancementDecision` dataclass in `src/montage_ai/enhancement_tracking.py`
- [ ] Create parameter dataclasses (`StabilizeParams`, `ColorGradeParams`, etc.)
- [ ] Add `EnhancementTracker` class to collect decisions during pipeline

### Phase 2: Integration
- [ ] Modify `ClipEnhancer` to return `EnhancementDecision` objects
- [ ] Wire `EnhancementTracker` into `MontageBuilder` pipeline
- [ ] Store decisions per-clip during montage creation

### Phase 3: Export
- [ ] Extend `TimelineExporter._export_otio()` to include enhancement metadata
- [ ] Extend `TimelineExporter._export_edl()` to include comment annotations
- [ ] Extend `TimelineExporter._export_xml()` to include effect nodes where possible
- [ ] Create `RecipeCardExporter` for human-readable Markdown output

### Phase 4: NLE Mapping
- [ ] Create mapping tables for FFmpeg filters → DaVinci Resolve nodes
- [ ] Create mapping tables for FFmpeg filters → Premiere Pro effects
- [ ] Create mapping tables for FFmpeg filters → Final Cut Pro effects
- [ ] Test roundtrip: Montage AI → OTIO → Resolve → Re-export

---

## NLE Mapping Reference

### Denoising

| Montage AI (FFmpeg) | DaVinci Resolve | Premiere Pro | Final Cut Pro |
|---------------------|-----------------|--------------|---------------|
| `hqdn3d=4:3:4:3`    | Spatial NR: Luma=4, Chroma=3 | Denoise (Legacy) | Noise Reduction |
| `nlmeans=s=5`       | Temporal NR: Frame=3, Threshold=5 | Reduce Noise | Auto Noise Removal |

### Sharpening

| Montage AI (FFmpeg) | DaVinci Resolve | Premiere Pro | Final Cut Pro |
|---------------------|-----------------|--------------|---------------|
| `unsharp=5:5:0.8`   | Sharpen: Radius=0.5, Amount=0.8 | Unsharp Mask: Amount=80, Radius=2.5 | Sharpen |
| `cas=0.5`           | Sharpening: Amount=0.5 | — | — |

### Color Grading

| Montage AI Preset | DaVinci Node Chain | Premiere Lumetri |
|-------------------|-------------------|------------------|
| `cinematic`       | Contrast S-curve + Teal shadows + Orange highlights | Creative Look: Cinematic |
| `teal_orange`     | Color Wheels: Lift→Teal, Gain→Orange | HSL Secondary |
| `vintage`         | Faded blacks + Desaturation + Grain | Creative Look: Vintage |

### Stabilization

| Montage AI Method | DaVinci | Premiere |
|-------------------|---------|----------|
| vidstab 2-pass    | Stabilizer (Perspective) | Warp Stabilizer (Smooth Motion) |
| deshake           | Stabilizer (Translation Only) | Warp Stabilizer (No Motion) |
| cgpu              | — (manual recreation needed) | — |

---

## File Structure

```
src/montage_ai/
├── enhancement_tracking.py      # NEW: EnhancementDecision, EnhancementTracker
├── nle_mapping.py               # NEW: FFmpeg → NLE parameter mapping
├── recipe_exporter.py           # NEW: Markdown recipe card generation
├── clip_enhancement.py          # MODIFY: Return EnhancementDecision from methods
├── timeline_exporter.py         # MODIFY: Include enhancement metadata in exports
└── core/
    └── montage_builder.py       # MODIFY: Wire EnhancementTracker into pipeline
```

---

## Benefits

1. **Non-Destructive Workflow**: Editors can see what AI did and adjust it
2. **Professional Compliance**: Meets broadcast/film standards for edit documentation
3. **Reproducibility**: Recipe cards enable exact recreation in any NLE
4. **AI Transparency**: `ai_reasoning` field explains WHY decisions were made
5. **Iterative Refinement**: Editors can re-run Montage AI with adjusted parameters

---

*Design created: 2026-01-06*
*Status: Planning Phase*
