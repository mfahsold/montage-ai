# AI Director

LLM-based Creative Director component for natural language video editing control.

---

## Overview

The **Creative Director** translates natural language prompts into structured editing instructions.

```text
User Prompt → Keyword Matching → LLM (if needed) → JSON Instructions → Editing Engine
```

---

## LLM Backends (Priority Order)

| Priority | Backend                      | Protocol          | Use Case                    |
| -------- | ---------------------------- | ----------------- | --------------------------- |
| 1        | Google AI (Gemini 2.0 Flash) | REST API          | Fast, reliable, JSON-native |
| 2        | cgpu serve                   | OpenAI-compatible | Gemini via cgpu proxy       |
| 3        | Ollama                       | Local HTTP        | Offline fallback            |

**Fallback chain:** Google AI → cgpu → Ollama → Default instructions

---

## Keyword Matching (No LLM)

Fast pattern recognition for common styles. Zero latency, no API costs.

| Keywords                                  | Mapped Style   |
| ----------------------------------------- | -------------- |
| "hitchcock", "suspense", "thriller"       | `hitchcock`    |
| "action", "michael bay", "fast cuts"      | `action`       |
| "mtv", "music video", "energetic"         | `mtv`          |
| "documentary", "natural", "observational" | `documentary`  |
| "minimalist", "calm", "long takes"        | `minimalist`   |
| "wes anderson", "symmetry", "quirky"      | `wes_anderson` |

---

## JSON Output Schema

The LLM returns structured JSON with these categories:

```json
{
  "style": {
    "name": "hitchcock | action | mtv | documentary | minimalist | wes_anderson | custom",
    "mood": "suspenseful | playful | energetic | calm | dramatic | mysterious"
  },
  "pacing": {
    "speed": "very_slow | slow | medium | fast | very_fast | dynamic",
    "variation": "minimal | moderate | high | fibonacci",
    "intro_duration_beats": 4,
    "climax_intensity": 0.9
  },
  "cinematography": {
    "prefer_wide_shots": false,
    "prefer_high_action": true,
    "match_cuts_enabled": true,
    "invisible_cuts_enabled": true,
    "shot_variation_priority": "high"
  },
  "transitions": {
    "type": "hard_cuts | crossfade | mixed | energy_aware",
    "crossfade_duration_sec": 0.3
  },
  "energy_mapping": {
    "sync_to_beats": true,
    "energy_amplification": 1.0
  },
  "effects": {
    "color_grading": "none | neutral | warm | cool | high_contrast | desaturated | vibrant",
    "stabilization": false,
    "upscale": false,
    "sharpness_boost": true
  },
  "constraints": {
    "target_duration_sec": null,
    "min_clip_duration_sec": 0.5,
    "max_clip_duration_sec": 10.0
  }
}
```

---

## Examples

### Hitchcock Thriller

**Prompt:** `"Edit this like a Hitchcock thriller"`

```json
{
  "style": {"name": "hitchcock", "mood": "suspenseful"},
  "pacing": {"speed": "dynamic", "variation": "high", "intro_duration_beats": 16, "climax_intensity": 0.9},
  "cinematography": {"prefer_high_action": true, "match_cuts_enabled": true, "shot_variation_priority": "high"},
  "transitions": {"type": "hard_cuts"},
  "effects": {"color_grading": "high_contrast", "sharpness_boost": true}
}
```

**Result:** Slow tension build (16 beats), explosive climax, hard cuts, high contrast.

---

### Meditative / Calm

**Prompt:** `"Make it calm and meditative with long shots"`

```json
{
  "style": {"name": "minimalist", "mood": "calm"},
  "pacing": {"speed": "very_slow", "variation": "minimal", "intro_duration_beats": 32, "climax_intensity": 0.3},
  "cinematography": {"prefer_wide_shots": true, "shot_variation_priority": "low"},
  "transitions": {"type": "crossfade", "crossfade_duration_sec": 2.0},
  "effects": {"color_grading": "desaturated", "stabilization": true},
  "constraints": {"min_clip_duration_sec": 4.0, "max_clip_duration_sec": 60.0}
}
```

**Result:** Very slow pacing, 4-60s clips, soft crossfades, desaturated colors.

---

### MTV Music Video

**Prompt:** `"Fast-paced music video style"`

```json
{
  "style": {"name": "mtv", "mood": "energetic"},
  "pacing": {"speed": "very_fast", "variation": "high", "intro_duration_beats": 2, "climax_intensity": 1.0},
  "cinematography": {"prefer_high_action": true, "shot_variation_priority": "high"},
  "transitions": {"type": "hard_cuts"},
  "effects": {"color_grading": "vibrant", "sharpness_boost": true},
  "energy_mapping": {"energy_amplification": 1.5}
}
```

**Result:** 1-2 beat cuts, instant start, maximum intensity, vibrant colors.

---

## Integration

The JSON instructions are stored in `EDITING_INSTRUCTIONS` global and accessed throughout `editor.py`:

```python
# Example usage in editor.py
if EDITING_INSTRUCTIONS:
    transition_type = EDITING_INSTRUCTIONS.get('transitions', {}).get('type', 'energy_aware')
    if transition_type == "crossfade":
        clip = clip.crossfadein(fade_duration)
```

**Controlled parameters:**

- **Pacing:** Cut duration based on `speed` and `variation`
- **Transitions:** Crossfade vs hard cuts
- **Cinematography:** Shot selection (wide vs close, action vs calm)
- **Effects:** Color grading, stabilization, upscaling

---

## Performance

| Method        | Latency    | Cost         |
| ------------- | ---------- | ------------ |
| Keyword Match | ~1ms       | Free         |
| Google AI     | 500-2000ms | Per-request  |
| Ollama (70B)  | 5-15s      | Free (local) |

---

## Configuration

```bash
# Google AI (recommended)
GOOGLE_API_KEY="your-api-key"
GOOGLE_AI_MODEL="gemini-2.0-flash"

# cgpu serve (optional)
CGPU_ENABLED="true"
CGPU_HOST="127.0.0.1"
CGPU_PORT="8080"

# Ollama (fallback)
OLLAMA_HOST="http://localhost:11434"
DIRECTOR_MODEL="llama3.1:70b"
```

---

## Adding Custom Styles

1. Add template to `src/montage_ai/styles/your_style.json`
2. Add keywords to `creative_director.py` → `style_keywords` dict
3. System prompt updates automatically from templates

---

## Troubleshooting

**LLM returns invalid JSON:**

- Gemini sometimes wraps JSON in markdown code blocks → automatically stripped
- Falls back to Ollama on parse error

**Ollama connection error:**

```bash
ollama serve
curl http://localhost:11434/api/tags  # verify
```

**Google AI rate limit:**

- Automatic fallback to Ollama
- Alternative: use cgpu serve with your own API keys
