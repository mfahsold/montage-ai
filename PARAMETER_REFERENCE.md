# Montage AI - Complete Parameter Reference

This document provides a comprehensive reference of all parameters that Montage AI controls and exports to professional NLEs (DaVinci Resolve, Premiere Pro, Final Cut Pro).

**Generated:** 2026-01-06
**Version:** 1.0

---

## Table of Contents

1. [Video Enhancement Parameters](#1-video-enhancement-parameters)
2. [Color Grading Parameters](#2-color-grading-parameters)
3. [Audio Analysis Parameters](#3-audio-analysis-parameters)
4. [Timeline & Editing Parameters](#4-timeline--editing-parameters)
5. [Style Template Parameters](#5-style-template-parameters)
6. [Clip Analysis Parameters](#6-clip-analysis-parameters)
7. [NLE Mapping Tables](#7-nle-mapping-tables)
8. [Preview Profile](#8-preview-profile)

---

## 1. Video Enhancement Parameters

### 1.1 Denoising (`DenoiseConfig`)

| Parameter | Type | Range | Default | Description | FFmpeg Filter |
|-----------|------|-------|---------|-------------|---------------|
| `enabled` | bool | - | `True` | Enable denoising | - |
| `temporal_strength` | float | 0.0-1.0 | 0.5 | Cross-frame noise reduction | `hqdn3d` luma_tmp |
| `spatial_strength` | float | 0.0-1.0 | 0.3 | In-frame noise reduction | `hqdn3d` luma_spatial |
| `chroma_strength` | float | 0.0-1.0 | 0.5 | Color channel noise reduction | `hqdn3d` chroma |
| `preserve_grain` | float | 0.0-1.0 | 0.2 | Preserve film grain texture | Reduces strength |
| `use_nlmeans` | bool | - | `False` | Use high-quality nlmeans | `nlmeans` |
| `nlmeans_strength` | int | 1-10 | 4 | NLMeans denoise strength | `nlmeans` s= |

**FFmpeg Filter Chain:**
```
# hqdn3d mode (default):
hqdn3d=<spatial*8>:<chroma*6>:<temporal*6>:<chroma*temporal*4>

# nlmeans mode (high quality):
nlmeans=s=<strength>:p=7:pc=5:r=3
```

### 1.2 Sharpening (`SharpenConfig`)

| Parameter | Type | Range | Default | Description | FFmpeg Filter |
|-----------|------|-------|---------|-------------|---------------|
| `enabled` | bool | - | `True` | Enable sharpening | - |
| `amount` | float | 0.0-1.0 | 0.4 | Sharpening intensity | `unsharp` luma_amount |
| `radius` | float | 0.5-11.5 | 1.5 | Edge detection radius (pixels) | `unsharp` size |
| `threshold` | int | 0-255 | 10 | Noise threshold | - |
| `protect_skin` | bool | - | `True` | Reduce on skin tones | Reduced chroma |

**FFmpeg Filter Chain:**
```
unsharp=<size>:<size>:<amount*2>:<size>:<size>:<amount>
# size = (radius * 2) * 2 + 1, clamped to 3-23
```

### 1.3 Film Grain (`FilmGrainConfig`)

| Parameter | Type | Range | Default | Description | FFmpeg Filter |
|-----------|------|-------|---------|-------------|---------------|
| `enabled` | bool | - | `False` | Enable film grain | - |
| `grain_type` | str | fine/medium/coarse/35mm/16mm/8mm | fine | Grain character | `noise` strength |
| `intensity` | float | 0.0-1.0 | 0.3 | Overall grain intensity | `noise` c0s= |
| `size` | float | 0.5-2.0 | 1.0 | Grain particle size | - |
| `shadow_boost` | float | 0.5-2.0 | 1.2 | More grain in shadows | - |
| `highlight_reduce` | float | 0.5-1.0 | 0.8 | Less grain in highlights | - |

**Grain Type Presets:**
| Type | Base Strength | Flags |
|------|--------------|-------|
| `fine` | 8 | t+u |
| `medium` | 12 | t+u |
| `coarse` | 18 | t+u |
| `35mm` | 10 | t+u+a |
| `16mm` | 15 | t+u+a |
| `8mm` | 22 | t+u+a |

### 1.4 Stabilization (`StabilizeParams`)

| Parameter | Type | Range | Default | Description | NLE Equivalent |
|-----------|------|-------|---------|-------------|----------------|
| `method` | str | vidstab/deshake/cgpu | vidstab | Stabilization algorithm | Resolve: Stabilizer |
| `smoothing` | int | 1-100 | 30 | Frame window for smoothing | Resolve: Smoothing |
| `crop_mode` | str | black/crop/fill | black | Border handling | Resolve: Crop Ratio |
| `zoom` | float | 0.0-0.5 | 0.0 | Additional zoom to hide borders | Resolve: Zoom |
| `shakiness` | int | 1-10 | 5 | Motion detection sensitivity | - |
| `accuracy` | int | 1-15 | 15 | Motion analysis accuracy | - |

**Method Comparison:**
| Method | Quality | Speed | Requirements |
|--------|---------|-------|--------------|
| `vidstab` | Excellent | Slow (2-pass) | FFmpeg with libvidstab |
| `deshake` | Good | Fast | FFmpeg builtin |
| `cgpu` | Excellent | Variable | Cloud GPU access |

### 1.5 Upscaling (`UpscaleParams`)

| Parameter | Type | Range | Default | Description | NLE Equivalent |
|-----------|------|-------|---------|-------------|----------------|
| `method` | str | realesrgan/lanczos/cgpu | lanczos | Upscaling algorithm | Resolve: Super Scale |
| `scale_factor` | int | 2/4 | 2 | Magnification factor | Resolve: Scale |
| `model` | str | - | realesrgan-x4plus | AI model name | - |
| `crf` | int | 0-51 | 18 | Output quality (lower=better) | - |

**Available Models:**
- `realesrgan-x4plus` - General purpose (default)
- `realesrgan-x4plus-anime` - Anime/illustration optimized
- `realesr-animevideov3` - Anime video

---

## 2. Color Grading Parameters

### 2.1 Color Grade Config (`ColorGradeConfig`)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `preset` | str | See presets | teal_orange | Color grading preset |
| `intensity` | float | 0.0-1.0 | 1.0 | Grade strength |
| `lut_path` | str | - | None | Custom LUT file path |
| `normalize_first` | bool | - | True | Apply broadcast safe levels first |

### 2.2 Available Color Presets

| Preset | Category | Description | FFmpeg Filter |
|--------|----------|-------------|---------------|
| `none` | Basic | No processing | - |
| `neutral` | Basic | Broadcast safe levels (16-235) | `colorlevels` |
| `natural` | Basic | Minimal processing | `eq=saturation=1.02:contrast=1.01` |
| `warm` | Temperature | Golden sunset warmth | `colortemperature=6500` + `colorbalance` |
| `cool` | Temperature | Blue/teal coolness | `colortemperature=8000` + `colorbalance` |
| `golden_hour` | Temperature | Sunset with lifted shadows | `colortemperature=5500` + `curves` |
| `blue_hour` | Temperature | Dawn/dusk cool tones | `colortemperature=9000` + `curves` |
| `teal_orange` | Cinematic | Hollywood blockbuster | `colorbalance` (shadows→teal, highlights→orange) |
| `cinematic` | Cinematic | Teal & Orange + S-curve | `colorbalance` + `curves` |
| `blockbuster` | Cinematic | High contrast action | `colorbalance` + `curves` + `eq=contrast=1.1` |
| `vibrant` | Stylized | Punchy, saturated | `eq=saturation=1.25:contrast=1.1` |
| `desaturated` | Stylized | Muted, filmic | `eq=saturation=0.75:contrast=1.05` |
| `high_contrast` | Stylized | Strong blacks/whites | `eq=contrast=1.3` + `curves` |
| `vintage` | Stylized | Faded film with lifted blacks | `curves` (lifted blacks) + `eq=saturation=0.8` |
| `filmic_warm` | Stylized | Classic warm film | `colorbalance` + `curves` + `eq=saturation=0.92` |
| `noir` | Stylized | Desaturated high contrast | `eq=saturation=0.3:contrast=1.4` |
| `documentary` | Documentary | Natural with sharpening | `eq` + `unsharp=3:3:0.3` |

### 2.3 Primary Color Wheels (`ColorGradeParams`)

| Parameter | Type | Range | Default | Description | NLE Equivalent |
|-----------|------|-------|---------|-------------|----------------|
| `lift` | (R,G,B) | -1.0 to 1.0 | (0,0,0) | Shadow color adjustment | Resolve: Lift |
| `gamma` | (R,G,B) | 0.0 to 2.0 | (1,1,1) | Midtone color adjustment | Resolve: Gamma |
| `gain` | (R,G,B) | 0.0 to 2.0 | (1,1,1) | Highlight color adjustment | Resolve: Gain |
| `offset` | (R,G,B) | -1.0 to 1.0 | (0,0,0) | Overall color offset | Resolve: Offset |

### 2.4 Basic Adjustments

| Parameter | Type | Range | Default | Description | NLE Equivalent |
|-----------|------|-------|---------|-------------|----------------|
| `saturation` | float | 0.0-2.0 | 1.0 | Color intensity | Resolve/Premiere: Saturation |
| `contrast` | float | 0.0-2.0 | 1.0 | Tonal contrast | Resolve/Premiere: Contrast |
| `brightness` | float | -1.0 to 1.0 | 0.0 | Overall brightness | Resolve: Brightness |
| `temperature` | float | -1.0 to 1.0 | 0.0 | Cool (-1) to Warm (+1) | Premiere: Temperature |
| `tint` | float | -1.0 to 1.0 | 0.0 | Green (-1) to Magenta (+1) | Premiere: Tint |

### 2.5 Color Matching (`ColorMatchParams`)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `reference_clip` | str | - | - | Path to reference clip |
| `method` | str | mkl/hm/mvgd | mkl | Color transfer algorithm |
| `r_adjustment` | float | -0.3 to 0.3 | 0.0 | Red channel adjustment |
| `g_adjustment` | float | -0.3 to 0.3 | 0.0 | Green channel adjustment |
| `b_adjustment` | float | -0.3 to 0.3 | 0.0 | Blue channel adjustment |

**Color Transfer Methods:**
| Method | Full Name | Speed | Quality |
|--------|-----------|-------|---------|
| `mkl` | Monge-Kantorovich Linear | Fast | Good |
| `hm` | Histogram Matching | Fastest | Basic |
| `mvgd` | Mean/Variance Gray Mapping | Medium | Best |

---

## 3. Audio Analysis Parameters

### 3.1 Beat Detection (`BeatInfo`)

| Parameter | Type | Description | Usage |
|-----------|------|-------------|-------|
| `tempo` | float | BPM (beats per minute) | Cut timing |
| `beat_times` | array | Timestamp of each beat (seconds) | Sync cuts to beats |
| `duration` | float | Total track duration (seconds) | Timeline length |
| `sample_rate` | int | Audio sample rate (Hz) | - |
| `beat_count` | int | Number of detected beats | - |
| `avg_beat_interval` | float | Average time between beats | Cut length calculation |
| `tempo_category` | str | slow (<80) / medium / fast (>140) | Pacing style |

### 3.2 Energy Profile (`EnergyProfile`)

| Parameter | Type | Description | Usage |
|-----------|------|-------------|-------|
| `times` | array | Time points (seconds) | Energy lookup |
| `rms` | array | Normalized RMS energy (0-1) | Dynamic cut length |
| `avg_energy` | float | Average energy level | Overall intensity |
| `max_energy` | float | Peak energy level | Climax detection |
| `min_energy` | float | Minimum energy level | Intro/outro detection |
| `high_energy_pct` | float | % of track with energy > 70% | Intensity rating |

### 3.3 Music Sections (`MusicSection`)

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `start_time` | float | seconds | Section start |
| `end_time` | float | seconds | Section end |
| `energy_level` | str | low/medium/high | Energy category |
| `avg_energy` | float | 0-1 | Average RMS in section |
| `label` | str | intro/build/drop/outro/verse/chorus | Story arc phase |

**Story Arc Mapping:**
| Label | Energy | Position | Pacing |
|-------|--------|----------|--------|
| `intro` | Low | 0-15% | Long cuts (8 beats) |
| `build` | Medium→High | 15-40% | Accelerating (8→2 beats) |
| `drop` | High | 40-75% | Rapid cuts (1-2 beats) |
| `outro` | Low | 85-100% | Decelerating (4→16 beats) |

### 3.4 Audio Quality (`AudioQuality`)

| Parameter | Type | Description | Threshold |
|-----------|------|-------------|-----------|
| `snr_db` | float | Signal-to-Noise Ratio (dB) | - |
| `mean_volume_db` | float | Average volume (dBFS) | - |
| `max_volume_db` | float | Peak volume (dBFS) | - |
| `is_usable` | bool | Audio is usable | SNR ≥ 8dB |
| `quality_tier` | str | Quality classification | See below |

**Quality Tiers:**
| Tier | SNR Range | Action |
|------|-----------|--------|
| `excellent` | ≥40 dB | No processing needed |
| `good` | 25-40 dB | Light denoising |
| `acceptable` | 15-25 dB | Moderate denoising |
| `poor` | 8-15 dB | Heavy denoising, warning |
| `unusable` | <8 dB | Reject or manual review |

---

## 8. Preview Profile

Centralized preview settings used by Transcript/Shorts previews and internal proxies. These improve iteration speed and can be tuned per environment.

| Parameter              | Type  | Default    | Description                                     |
|------------------------|-------|------------|-------------------------------------------------|
| `PREVIEW_WIDTH`        | int   | `640`      | Preview width (pixels)                          |
| `PREVIEW_HEIGHT`       | int   | `360`      | Preview height (pixels)                         |
| `PREVIEW_CRF`          | int   | `28`       | Quality factor for previews                     |
| `PREVIEW_PRESET`       | str   | `ultrafast`| Encoder preset for previews                     |
| `PREVIEW_MAX_DURATION` | float | `30.0`     | Max preview duration (seconds)                  |
| `PREVIEW_TIME_TARGET`  | int   | `180`      | KPI target for Time-to-First-Preview (seconds)  |

Implementation notes:
- `ffmpeg_config` exposes `PREVIEW_*` constants that mirror `settings.preview`.
- `scene_analysis` proxy generation and `preview_generator` now use `settings.preview`.
- Changing these values does not affect final export settings; they only affect previews and proxies.


## 4. Timeline & Editing Parameters

### 4.1 Clip (`Clip`)

| Parameter | Type | Description | Export |
|-----------|------|-------------|--------|
| `source_path` | str | Original video file path | OTIO/EDL/XML |
| `start_time` | float | In-point in source (seconds) | Source TC |
| `duration` | float | Clip duration (seconds) | Duration |
| `timeline_start` | float | Position in final timeline | Record TC |
| `proxy_path` | str | Path to proxy file (if generated) | Relinking |
| `metadata` | dict | Additional metadata | Comments |
| `enhancement_decision` | obj | Applied enhancements | Recipe card |

### 4.2 Timeline (`Timeline`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clips` | list | - | List of Clip objects |
| `audio_path` | str | - | Background music/audio path |
| `total_duration` | float | - | Total timeline duration |
| `fps` | float | 30.0 | Frame rate |
| `resolution` | tuple | (1080, 1920) | Width x Height |
| `project_name` | str | fluxibri_montage | Project identifier |

---

## 5. Style Template Parameters

Style templates are JSON files in `src/montage_ai/styles/` that define editing presets.

### 5.1 Pacing Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `pacing.speed` | str | slow/medium/fast | Overall cut frequency |
| `pacing.variation` | str | low/moderate/high | Cut length variety |
| `pacing.beat_sync` | bool | true/false | Sync cuts to music beats |

### 5.2 Transition Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `transitions.type` | str | cut/crossfade/dissolve/wipe | Default transition |
| `transitions.duration` | float | seconds | Transition duration |

### 5.3 Effects Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `effects.color_grading` | str | preset name | Color grading preset |
| `effects.stabilize` | bool | true/false | Enable stabilization |
| `effects.upscale` | bool | true/false | Enable AI upscaling |
| `effects.denoise` | bool | true/false | Enable denoising |

### 5.4 Available Style Templates

| Style | Pacing | Transitions | Color | Use Case |
|-------|--------|-------------|-------|----------|
| `hitchcock` | slow | dissolve | cinematic | Suspense, drama |
| `mtv` | fast | cut | vibrant | Music videos |
| `documentary` | medium | crossfade | documentary | Interviews |
| `action` | fast | cut | high_contrast | Sports, action |
| `wes_anderson` | slow | cut | warm | Symmetrical, quirky |
| `minimalist` | slow | crossfade | desaturated | Art, architecture |
| `viral` | fast | cut | vibrant | TikTok/Reels |
| `wedding` | slow | dissolve | filmic_warm | Weddings |
| `travel` | medium | crossfade | warm | Travel vlogs |
| `gaming` | fast | cut | vibrant | Gaming clips |

---

## 6. Clip Analysis Parameters

### 6.1 Motion Analysis

| Parameter | Type | Range | Description | Enhancement |
|-----------|------|-------|-------------|-------------|
| `shake_score` | float | 0-1 | Camera shake intensity | Stabilization |
| `motion_type` | str | static/pan/handheld/action | Motion classification | Cut selection |

### 6.2 Image Quality

| Parameter | Type | Range | Description | Enhancement |
|-----------|------|-------|-------------|-------------|
| `noise_level` | float | 0-1 | Estimated noise | Denoising |
| `sharpness_score` | float | 0-1 | Edge sharpness | Sharpening |
| `brightness_avg` | float | 0-255 | Average luma | Exposure correction |
| `is_dark` | bool | - | Under-exposed | Brightness boost |
| `is_bright` | bool | - | Over-exposed | Brightness reduce |

### 6.3 Content Analysis

| Parameter | Type | Values | Description | Usage |
|-----------|------|--------|-------------|-------|
| `scene_type` | str | establishing/action/detail/portrait/scenic | Shot classification | Pacing |
| `dominant_colors` | list | color names | Dominant colors | Color matching |
| `has_faces` | bool | - | Face detection | Skin protection |
| `has_text` | bool | - | Text/graphics detection | Sharpening |

---

## 7. NLE Mapping Tables

### 7.1 Denoising → NLE

| Montage AI | DaVinci Resolve | Premiere Pro | Final Cut Pro |
|------------|-----------------|--------------|---------------|
| `hqdn3d spatial=0.3` | Spatial NR: Luma=3.0 | Denoise (Legacy) | Noise Reduction: 0.3 |
| `hqdn3d temporal=0.5` | Temporal NR: Frame=3 | - | - |
| `nlmeans s=5` | Temporal NR: Threshold=5 | Reduce Noise | Auto Noise Removal |

### 7.2 Sharpening → NLE

| Montage AI | DaVinci Resolve | Premiere Pro | Final Cut Pro |
|------------|-----------------|--------------|---------------|
| `unsharp amount=0.4` | Blur/Sharpen: 0.4 | Unsharp Mask: 40% | Sharpen: 0.4 |
| `unsharp radius=1.5` | Radius: 1.5 | Radius: 1.5px | Radius: 1.5 |
| `cas=0.5` | Sharpening: 0.5 | - | - |

### 7.3 Stabilization → NLE

| Montage AI | DaVinci Resolve | Premiere Pro |
|------------|-----------------|--------------|
| `vidstab smoothing=30` | Stabilizer: Smoothing=0.30 | Warp Stabilizer: 30% |
| `vidstab crop=black` | Crop Ratio: 0.0 | Framing: Stabilize Only |
| `vidstab crop=crop` | Crop Ratio: 1.0 | Framing: Crop |
| `deshake` | Stabilizer: Translation | Warp Stabilizer: No Motion |

### 7.4 Color Grading → NLE

| Montage AI Preset | DaVinci Resolve | Premiere Pro Lumetri |
|-------------------|-----------------|----------------------|
| `cinematic` | Contrast S-curve + Teal/Orange | Creative: Cinematic |
| `teal_orange` | Color Wheels: Lift→Teal, Gain→Orange | HSL Secondary |
| `vintage` | Curves: Lifted blacks + Desat | Creative: Vintage |
| `warm` | Color Temp: 6500K | Basic: Temperature +20 |
| `cool` | Color Temp: 8000K | Basic: Temperature -15 |
| `vibrant` | Saturation: 125% | Basic: Saturation +25 |
| `desaturated` | Saturation: 75% | Basic: Saturation -25 |

---

## Export Formats

### OTIO (OpenTimelineIO)
- Full enhancement metadata in `clip.metadata["montage_ai"]`
- Parameters preserved for re-creation
- AI reasoning included

### EDL (CMX 3600)
- Comments with `* MONTAGE_AI` prefix
- Each enhancement on separate line
- Universal NLE compatibility

### FCP XML
- Effect nodes where possible
- Labels for quick identification
- Comments for parameters

### Recipe Card (Markdown)
- Human-readable instructions
- Step-by-step recreation for each NLE
- AI reasoning explanations

---

*Documentation generated by Montage AI v1.0*
*For questions: https://github.com/fluxibri/montage-ai*
