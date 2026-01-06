# Color Grading & Image Enhancement Strategy

> Strategic assessment with LUT/Color Grading properly positioned IN SCOPE.

**Date:** January 6, 2026
**Status:** Action Plan Ready

---

## Executive Summary

**CORRECTION:** The previous strategic analysis incorrectly marked `AI_LUT_GENERATION` as out of scope. **Color Grading and LUT support are CORE features** of the "Polish, Don't Generate" philosophy.

**Current Reality:** Color grading is DEFINED in style templates but **NOT IMPLEMENTED** in the rendering pipeline. This is a significant feature gap that undermines our value proposition.

---

## Part 1: Current State Analysis

### 1.1 What We Have (Implemented)

| Component | File | Status |
|-----------|------|--------|
| **ClipEnhancer** | `clip_enhancement.py` | Hardcoded "Teal & Orange" only |
| **ColorHarmonizer** | `color_harmonizer.py` | Working LUT application support |
| **HarmonizerConfig** | `color_harmonizer.py` | Presets: NATURAL, VIBRANT, CINEMATIC, DOCUMENTARY |
| **LUT Directory** | `data/luts/` | Documented but empty |
| **Style Templates** | `styles/*.json` | Define `color_grading` values |
| **LUT Generator** | `cgpu_jobs/lut_generator.py` | SKELETON only (NotImplementedError) |

### 1.2 The Gap: Style Templates Promise What Pipeline Ignores

**Style Template Example (cinema_trailer.json):**
```json
{
  "effects": {
    "color_grading": "teal_orange",  // <-- DEFINED
    "stabilization": true,
    "sharpness_boost": true
  }
}
```

**Pipeline Reality (montage_builder.py):**
```python
# Effects that ARE used:
self.ctx.stabilize = effects['stabilization']  # ✅ Works
self.ctx.upscale = effects['upscale']          # ✅ Works
self.ctx.enhance = effects['sharpness_boost']  # ✅ Works

# What's MISSING:
# color_grading value is NEVER read from style template!
# clip_enhancement.py always applies hardcoded "Teal & Orange"
```

### 1.3 Impact Assessment

| Issue | User Impact | Severity |
|-------|-------------|----------|
| `documentary` style still gets "Teal & Orange" | Wrong look for documentary content | **HIGH** |
| `wes_anderson` style gets same grading as `action` | No style differentiation | **HIGH** |
| Web UI has no color grading controls | Users can't customize | **MEDIUM** |
| LUT files in `data/luts/` are ignored | Manual setup useless | **MEDIUM** |
| LUT Generator skeleton unusable | Cloud colorist feature blocked | **LOW** |

---

## Part 2: Color Grading Scope Definition

### 2.1 What IS In Scope ("Polish, Don't Generate")

| Feature | Rationale |
|---------|-----------|
| **FFmpeg Filter-based Grading** | Fast, local, no AI needed |
| **LUT Application** | Industry standard, NLE-compatible |
| **Color Matching** | Shot-to-shot consistency |
| **Preset Profiles** | NATURAL, VIBRANT, CINEMATIC, etc. |
| **Style-Driven Grading** | Each style template gets appropriate look |

### 2.2 What Might Be Deferred

| Feature | Rationale | Decision |
|---------|-----------|----------|
| **AI LUT Generation** | Requires cloud GPU, complex | DEFER to Q2 |
| **Reference Image Matching** | Nice-to-have, not core workflow | DEFER |
| **Manual Color Wheels** | NLE job, not rough-cut scope | OUT |

---

## Part 3: Implementation Plan

### Phase 1: Connect Style Templates to Pipeline (1-2 days)

**Goal:** Make `color_grading` value from style templates actually work.

#### 3.1.1 Add Color Grading Presets (clip_enhancement.py)

```python
# New: Color grading filter presets
COLOR_GRADING_FILTERS = {
    "none": "",
    "neutral": "eq=saturation=1.0:contrast=1.0",
    "warm": "colortemperature=6500,eq=saturation=1.08",
    "cool": "colortemperature=8000,eq=saturation=0.95",
    "teal_orange": "colorbalance=rs=-0.1:gs=-0.05:bs=0.15...",  # Current hardcoded
    "vibrant": "eq=saturation=1.25:contrast=1.1",
    "desaturated": "eq=saturation=0.7:contrast=1.05",
    "high_contrast": "eq=contrast=1.3:brightness=0.02",
    "filmic_warm": "curves=m='0 0 0.25 0.22 0.5 0.5 0.75 0.78 1 1',eq=saturation=0.9",
    "cinematic_teal_orange": "colorbalance=...,curves=...",
}
```

#### 3.1.2 Modify enhance() to Accept Style Parameter

```python
def enhance(self, input_path: str, output_path: str, color_grade: str = "teal_orange") -> str:
    """Apply style-appropriate color grading."""
    grade_filter = COLOR_GRADING_FILTERS.get(color_grade, COLOR_GRADING_FILTERS["neutral"])
    # ... rest of enhance logic with dynamic filter
```

#### 3.1.3 Wire Up in montage_builder.py

```python
# In _parse_style_template() or similar:
if 'color_grading' in effects:
    self.ctx.color_grade = effects['color_grading']
else:
    self.ctx.color_grade = "neutral"

# In _process_clip_job():
result = enhancer.enhance(current_path, enhance_path, color_grade=ctx.color_grade)
```

### Phase 2: LUT File Support (1 day)

**Goal:** Allow users to drop `.cube` files and have them applied.

#### 3.2.1 Enhance ColorHarmonizer Integration

```python
# In color_harmonizer.py - already has lut3d support!
if cfg.lut_path and os.path.exists(cfg.lut_path):
    filters.append(f"lut3d={cfg.lut_path}")
```

#### 3.2.2 Add LUT Path Resolution

```python
# New utility function
def resolve_lut_path(style_name: str, lut_dir: Path) -> Optional[Path]:
    """Find LUT file matching style name."""
    for ext in [".cube", ".3dl", ".dat"]:
        lut_file = lut_dir / f"{style_name}{ext}"
        if lut_file.exists():
            return lut_file
    return None
```

#### 3.2.3 Style Template LUT Override

Allow styles to specify custom LUT:
```json
{
  "effects": {
    "color_grading": "custom",
    "lut_file": "kodak_2383.cube"  // Optional LUT override
  }
}
```

### Phase 3: Web UI Color Controls (1 day)

**Goal:** Expose color grading options in Web UI.

#### 3.3.1 Add to montage.html

```html
<div class="option-group">
  <div class="option-label">COLOR GRADING</div>
  <select name="color_grading" id="color_grading">
    <option value="auto">Auto (from style)</option>
    <option value="neutral">Neutral</option>
    <option value="warm">Warm</option>
    <option value="cool">Cool</option>
    <option value="teal_orange">Teal & Orange (Cinematic)</option>
    <option value="vibrant">Vibrant</option>
    <option value="desaturated">Desaturated (Documentary)</option>
    <option value="high_contrast">High Contrast</option>
  </select>
</div>
```

#### 3.3.2 Add to env_mapper.py

```python
# In map_options_to_env():
env["COLOR_GRADING"] = str(expanded.get("color_grading", "auto"))
```

#### 3.3.3 Add to job_options.py

```python
"color_grading": form.get("color_grading", "auto"),
```

### Phase 4: Bundle Free LUTs (Optional, 1 day)

**Goal:** Ship ready-to-use LUT presets.

#### 3.4.1 Download Open-Source LUTs

From [iwltbap/Free-Luts](https://github.com/iwltbap/Free-Luts):
- `cinematic.cube` - Classic Hollywood
- `vintage.cube` - Film emulation
- `teal_orange.cube` - Blockbuster look
- `documentary.cube` - Natural, neutral

#### 3.4.2 Add to data/luts/ with Attribution

```
data/luts/
├── README.md
├── cinematic.cube      # From iwltbap/Free-Luts, CC0
├── vintage.cube        # From iwltbap/Free-Luts, CC0
├── teal_orange.cube    # From iwltbap/Free-Luts, CC0
└── ATTRIBUTION.md
```

---

## Part 4: Quality Profiles Update

Current profiles need color grading integration:

| Profile | Color Grading | Enhancement |
|---------|--------------|-------------|
| **preview** | none | Off |
| **standard** | from style | On |
| **high** | from style | On + stabilize |
| **master** | from style + LUT | All features |

---

## Part 5: Updated SWOT (Color Grading Focus)

### Strengths (Now)
- ColorHarmonizer already supports LUT3D
- FFmpeg filter chain is modular
- Style templates already define grading values
- LUT directory structure exists

### Weaknesses (To Fix)
- **color_grading style value is IGNORED** - Priority fix
- No color grading in Web UI
- Hardcoded enhance() filter
- LUT generator is skeleton only

### Opportunities
- Differentiate from competitors with style-appropriate looks
- Ship pre-bundled open-source LUTs
- Enable "drop your LUT" workflow

### Threats
- Users expect style grading to work (it doesn't)
- Documentation promises features not implemented

---

## Part 6: Priority Actions

### IMMEDIATE (This Week)

1. **[P0] Wire color_grading from style to enhance()**
   - Files: `clip_enhancement.py`, `montage_builder.py`
   - Impact: All 16 styles finally get correct looks
   - Effort: ~4 hours

2. **[P1] Add COLOR_GRADING_FILTERS dictionary**
   - File: `clip_enhancement.py`
   - Impact: Replace hardcoded Teal & Orange
   - Effort: ~2 hours

3. **[P1] Add color grading dropdown to Web UI**
   - Files: `montage.html`, `env_mapper.py`, `job_options.py`
   - Impact: User control over look
   - Effort: ~2 hours

### SHORT-TERM (January)

4. **[P2] LUT file auto-detection**
   - File: New utility + integrate in pipeline
   - Impact: "Drop your LUT" workflow
   - Effort: ~4 hours

5. **[P2] Bundle 4 open-source LUTs**
   - Directory: `data/luts/`
   - Impact: Works out-of-box
   - Effort: ~1 hour

### MEDIUM-TERM (Q1)

6. **[P3] Color matching between clips**
   - ColorHarmonizer is already capable
   - Wire into pipeline as option
   - Effort: ~1 day

7. **[P3] LUT Generator implementation**
   - Complete cgpu_jobs/lut_generator.py skeleton
   - Requires AI model selection
   - Effort: ~3 days

---

## Part 7: Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Styles with working color grading | 0/16 | 16/16 |
| Web UI color options | 0 | 8+ presets |
| Bundled LUT files | 0 | 4+ |
| User satisfaction (color control) | N/A | >4/5 |

---

## Summary

**Color Grading is IN SCOPE.** The immediate priority is fixing the broken integration where style templates define `color_grading` values that are completely ignored by the pipeline.

This is a ~1-2 day fix that will:
1. Make all 16 style presets actually look different
2. Enable "documentary" to NOT look like a blockbuster
3. Deliver on the documented feature promise

**Recommended Next Step:** Implement Phase 1 (Connect Style Templates to Pipeline).

---

**Document Owner:** Product/Engineering
**Last Updated:** January 6, 2026
