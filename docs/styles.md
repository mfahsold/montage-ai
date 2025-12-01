# Style Guide

How to use built-in styles and create custom editing presets.

---

## Built-in Styles

### hitchcock

**Inspiration:** Alfred Hitchcock's suspense techniques

**Characteristics:**
- Slow tension build in intro (16+ beats)
- Explosive rapid cuts at climax
- High contrast color grading
- Match cuts and invisible cuts enabled
- Hard cuts only (no crossfades)

**Best for:** Thrillers, dramatic reveals, tension building

```bash
./montage-ai.sh run hitchcock
```

---

### mtv

**Inspiration:** 1990s-2000s music videos

**Characteristics:**
- Rapid 1-2 beat cuts throughout
- Maximum energy (climax_intensity: 1.0)
- Vibrant color grading
- Hard cuts only
- Beat synchronization amplified (1.5x)

**Best for:** Music videos, high-energy content, dance footage

```bash
./montage-ai.sh run mtv
```

---

### action

**Inspiration:** Michael Bay, action cinema

**Characteristics:**
- Fast dynamic pacing
- Preference for high-action clips
- Energy-aware transitions
- High shot variation
- Sharpness boost enabled

**Best for:** Sports, action sequences, adventure footage

```bash
./montage-ai.sh run action
```

---

### documentary

**Inspiration:** Ken Burns, nature documentaries

**Characteristics:**
- Natural, unhurried pacing
- Preference for establishing and scenic shots
- Longer takes (4-8 beats)
- Neutral color grading
- Mixed transitions (cuts + crossfades)

**Best for:** Travel videos, interviews, nature footage

```bash
./montage-ai.sh run documentary
```

---

### minimalist

**Inspiration:** Art house cinema, contemplative films

**Characteristics:**
- Very slow pacing
- Long contemplative takes
- Minimal variation
- Calm mood throughout
- Subtle, desaturated colors

**Best for:** Artistic projects, meditation content, slow cinema

```bash
./montage-ai.sh run minimalist
```

---

### wes_anderson

**Inspiration:** Wes Anderson's visual style

**Characteristics:**
- Symmetrical, centered compositions preferred
- Warm, pastel color grading
- Playful, whimsical mood
- Moderate pacing with deliberate rhythm
- High shot variation for visual interest

**Best for:** Quirky content, lifestyle videos, aesthetic projects

```bash
./montage-ai.sh run wes_anderson
```

---

## Creating Custom Styles

### JSON Structure

Create a `.json` file with this structure:

```json
{
  "id": "my_style",
  "name": "My Custom Style",
  "description": "One-line description of the style",
  "params": {
    "style": {
      "name": "my_style",
      "mood": "energetic"
    },
    "pacing": {
      "speed": "fast",
      "variation": "moderate",
      "intro_duration_beats": 8,
      "climax_intensity": 0.8
    },
    "cinematography": {
      "prefer_wide_shots": false,
      "prefer_high_action": true,
      "match_cuts_enabled": true,
      "invisible_cuts_enabled": false,
      "shot_variation_priority": "high"
    },
    "transitions": {
      "type": "hard_cuts",
      "crossfade_duration_sec": 0.2
    },
    "energy_mapping": {
      "sync_to_beats": true,
      "energy_amplification": 1.0
    },
    "effects": {
      "color_grading": "neutral",
      "stabilization": false,
      "upscale": false,
      "sharpness_boost": false
    }
  }
}
```

### Parameter Reference

#### style

| Parameter | Values                                                                  | Description            |
| --------- | ----------------------------------------------------------------------- | ---------------------- |
| `name`    | string                                                                  | Unique identifier      |
| `mood`    | `suspenseful`, `playful`, `energetic`, `calm`, `dramatic`, `mysterious` | Overall emotional tone |

#### pacing

| Parameter              | Values                                                        | Description            |
| ---------------------- | ------------------------------------------------------------- | ---------------------- |
| `speed`                | `very_slow`, `slow`, `medium`, `fast`, `very_fast`, `dynamic` | Base cut frequency     |
| `variation`            | `minimal`, `moderate`, `high`, `fibonacci`                    | Cut timing variation   |
| `intro_duration_beats` | 2-32                                                          | Beats before first cut |
| `climax_intensity`     | 0.0-1.0                                                       | Peak energy multiplier |

#### cinematography

| Parameter                 | Values                  | Description                   |
| ------------------------- | ----------------------- | ----------------------------- |
| `prefer_wide_shots`       | boolean                 | Favor establishing shots      |
| `prefer_high_action`      | boolean                 | Favor motion-heavy clips      |
| `match_cuts_enabled`      | boolean                 | Enable visual continuity cuts |
| `invisible_cuts_enabled`  | boolean                 | Enable motion-masked cuts     |
| `shot_variation_priority` | `low`, `medium`, `high` | Clip variety importance       |

#### transitions

| Parameter                | Values                                            | Description                     |
| ------------------------ | ------------------------------------------------- | ------------------------------- |
| `type`                   | `hard_cuts`, `crossfade`, `mixed`, `energy_aware` | Transition style                |
| `crossfade_duration_sec` | 0.0-2.0                                           | Fade duration (when applicable) |

#### energy_mapping

| Parameter              | Values  | Description                |
| ---------------------- | ------- | -------------------------- |
| `sync_to_beats`        | boolean | Align cuts to beats        |
| `energy_amplification` | 0.5-2.0 | Energy response multiplier |

#### effects

| Parameter         | Values                                                                       | Description         |
| ----------------- | ---------------------------------------------------------------------------- | ------------------- |
| `color_grading`   | `none`, `neutral`, `warm`, `cool`, `high_contrast`, `desaturated`, `vibrant` | Color look          |
| `stabilization`   | boolean                                                                      | Apply stabilization |
| `upscale`         | boolean                                                                      | Apply AI upscaling  |
| `sharpness_boost` | boolean                                                                      | Enhance sharpness   |

---

## Using Custom Styles

### Single File

```bash
STYLE_PRESET_PATH=/path/to/my_style.json ./montage-ai.sh run my_style
```

### Directory of Styles

```bash
STYLE_PRESET_DIR=/path/to/styles/ ./montage-ai.sh run my_style
```

### Override Priority

Styles are loaded in this order (later overrides earlier):

1. Built-in styles (`src/montage_ai/styles/*.json`)
2. `STYLE_PRESET_DIR` files (sorted alphabetically)
3. `STYLE_PRESET_PATH` file

If two files define the same `id`, the later one wins.

---

## Tips for Great Styles

1. **Match mood to music:** Energetic music needs fast pacing
2. **Use intro_duration_beats:** Give viewers time to settle in
3. **Balance variation:** Too much randomness feels chaotic
4. **Test with different footage:** Styles behave differently with various content
5. **Start from existing:** Copy a built-in style and modify
