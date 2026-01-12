# AI Director Parameter Tuning - Low-Hanging Fruits

**Status:** ✅ Implemented (Phase 1)  
**Date:** January 9, 2026  
**Architecture:** Central LLM-based parameter suggestion system

## Overview

Montage AI jetzt hat ein **zentrales Parameter-Suggestion-System** für AI-gesteuerte Editing-Optimierung. Alle LLM-Anfragen gehen durch `CreativeDirector` (cgpu-robust mit Ollama-Fallback).

## Product Requirements (PRD-light, single source)

- **User Input**: Intent (Schnitt, Musik, Look, Qualitätsverbesserungen via Radiobutton/Dropdown/Switch). Kein Parameter-Baukasten.
- **AI Output**: Entscheidungen + Begründung + Export-Artefakte (OTIO/EDL/Premiere/Resolve-kompatibel). Nutzer sieht „was“ und „warum“, kann optional einzelne Entscheidungen übernehmen/ablehnen.
- **UI-Scope**: Leichtgewichtiges „AI Decisions“-Panel (Anzeige, Reasoning, Confidence, Apply/Decline), plus „Export“ Button. Keine tiefen Parameter-Slider.
- **Backend**: Alle LLM-Calls über `CreativeDirector._query_backend()` mit Fallback-Kette (OpenAI-API → cgpu → Google AI → Ollama). cgpu kann für Heavy Load genutzt werden.
- **Philosophie**: „Polish, don’t generate“ – wir optimieren Schnitt, Look, Stabilisierung; kein Pixel-Gen.

## Export-Pipeline (Design Draft)

- **Quelle**: EditingParameters + Timeline/Clip-Metadaten (shots, in/out, audio beats, style)
- **Zielartefakte**: Primär OTIO, davon abgeleitet EDL/AAF/Premiere XML/Resolve (wo sinnvoll)
- **Color Grading**: Als Note/LUT-Hinweis; wenn gerendert angewendet → Note „baked“. Sonst „recommended“ im Export-Note-Track
- **Stabilisierung**: Analog Color; Flag „applied“ vs „recommended“ je Clip
- **Audio/Musik**: Timeline-Events (start timecode, in/out, track id), Beat-Marker für Pacing
- **Pacing/Beats**: Marker je Schnitt („cut at beat n“, „section=intro/build/climax/outro“)
- **Qualitätsverbesserungen**: Clip-Notes („denoise applied“, „sharpen off“, „upscale disabled“)
- **Roundtrip**: Import EditingParameters-JSON + OTIO → rekonstruierbare Decisions ohne Feldverlust

## Changelog (kurz)

- 2026-01-09: Intent-in/Decisions-out verankert; Director-Systemprompt export-ready; Preset-Single-Source auf color_grading; Suggester nutzt dieselben Presets.

## ✅ Was ist jetzt implementiert


### 1. Unified Parameter Schema (`editing_parameters.py`)

**Zweck:** Zentrale Parameterdefinition für alle Editing-Domains  
**Impact:** 🟢 HIGH - Verhindert Parameter-Fragmentierung


```python
from montage_ai.editing_parameters import EditingParameters

# Alle tunable Parameter in einem Schema

params = EditingParameters()
params.stabilization.smoothing = 20
params.color_grading.preset = "teal_orange"
params.color_grading.intensity = 0.9
params.pacing.speed = PacingSpeed.DYNAMIC
params.validate()  # Validiert alle Ranges

```

**Parameter-Gruppen:**

- **Stabilization:** 8 Parameter (smoothing, shakiness, accuracy, stepsize, zoom, optzoom, crop, method)
- **Color Grading:** 9 Parameter (preset, intensity, LUT, temperature, tint, saturation, contrast, brightness, normalize)
- **Clip Selection:** 10 Parameter (bonus/penalty-Faktoren, weights, LLM-ranking)
- **Pacing:** 10+ Parameter (speed, pattern, chaos_factor, beat-syncing, Fibonacci sequences)

**Total:** 50+ tunable Parameter, zentral validiert und serialisierbar (JSON).

---

### 2. LLM-based Parameter Suggester (`parameter_suggester.py`)

**Zweck:** AI-gesteuerte intelligente Parameter-Optimierung  
**Impact:** 🟢 HIGH - Automatisiert manuelle Tuning-Decisions

#### Base Class: `ParameterSuggester`

Zentrale Abstraktion für alle LLM-basierten Suggester:

- Nutzt `CreativeDirector` für LLM-Backend (cgpu/Ollama/Gemini)
- Robust gegen cgpu-Ausfälle (automatischer Fallback)
- Typed responses mit Reasoning

#### Implementierte Suggester


##### a) `ColorGradingSuggester` (🔥 HIGH VALUE)

**Problem:** Color grading requires artistic expertise  
**Lösung:** LLM analysiert Scene + Intent → schlägt Preset + Parameter vor


```python
from montage_ai.parameter_suggester import ColorGradingSuggester

suggester = ColorGradingSuggester()  # Auto-detects cgpu/Ollama
context = {
    "scene_description": "sunset beach with warm orange sky",
    "user_intent": "cinematic blockbuster",
    "dominant_colors": ["orange", "blue"],
    "histogram": {"shadows": 0.25, "midtones": 0.50, "highlights": 0.25}
}

suggestion = suggester.suggest(context)

# suggestion.parameters = {

#   "preset": "golden_hour",

#   "intensity": 0.9,

#   "temperature": 0.3,  # Warmer

#   "saturation": 1.2,

#   ...

# }

# suggestion.reasoning = "Golden hour preset enhances warm sunset tones..."

# suggestion.confidence = 0.85

```

**Features:**

- 20+ Presets (teal_orange, cinematic, blockbuster, vintage, noir, etc.)
- Histogram-aware Adjustments
- Confidence Scores
- Explained Decisions (LLM reasoning)

##### b) `StabilizationTuner` (🟡 MEDIUM VALUE)

**Problem:** Stabilization parameter tuning requires shake analysis expertise  
**Lösung:** LLM analysiert shake_score + motion_type → optimiert vidstab-Parameter


```python
from montage_ai.parameter_suggester import StabilizationTuner

tuner = StabilizationTuner()
context = {
    "shake_score": 0.7,  # 0-1 scale (0=stable, 1=very shaky)
    "motion_type": "handheld",
    "resolution": "1080p",
    "user_intent": "smooth cinematic motion"
}

suggestion = tuner.suggest(context)

# suggestion.parameters = {

#   "smoothing": 20,  # Higher for shakier footage

#   "shakiness": 7,

#   "accuracy": 12,

#   "zoom": 5,  # Slight zoom to crop borders

#   ...

# }

```

**Integration mit cgpu:**

- `StabilizeJob` in `cgpu_jobs/stabilize.py` nutzt die vorgeschlagenen Parameter
- Parameter werden geclampt (1-30 für smoothing, etc.)
- Robust gegen invalide LLM-Ausgaben (Fallback zu safe defaults)

---

### 3. Zentrale LLM-Integration (`creative_director.py`)

**Neu hinzugefügt:** `_query_backend()` Methode


```python
class CreativeDirector:
    def _query_backend(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generic LLM query für non-editing tasks (parameter suggestion).
        Versucht Backends in Reihenfolge: OpenAI-API → cgpu → Google AI → Ollama
        """

```

**Warum zentral?**

- ✅ Ein Ort für Backend-Fallback-Logik
- ✅ cgpu-Robustheit garantiert (auto-fallback zu Ollama)
- ✅ Keine duplizierten LLM-Calls in verschiedenen Modulen
- ✅ Konsistente Fehlerbehandlung

---

## 📊 Implementierungs-Status

| Feature | Status | Impact | Effort | Datei |
| --- | --- | --- | --- | --- |
| **Unified Parameter Schema** | ✅ Done | 🟢 HIGH | 2h | `editing_parameters.py` |
| **ColorGradingSuggester** | ✅ Done | 🟢 HIGH | 5h | `parameter_suggester.py` |
| **StabilizationTuner** | ✅ Done | 🟡 MEDIUM | 3h | `parameter_suggester.py` |
| **Central LLM Integration** | ✅ Done | 🟢 HIGH | 1h | `creative_director.py` |
| **Convenience Functions** | ✅ Done | 🟢 HIGH | 1h | `parameter_suggester.py` |
| **Test Suite** | ✅ Done | 🟡 MEDIUM | 2h | `test_parameter_suggester.py` |
| **PacingAdvisor** | ⏳ Planned | 🟡 MEDIUM | 4h | - |
| **Web UI Integration** | ⏳ Planned | 🟢 HIGH | 6h | - |

**Total Effort (Phase 1):** ~14 hours  
**Total Effort (Phase 2):** ~10 hours

---

## 🚀 Usage Examples

### Quick Start (Convenience Functions)


```python

# 1. Quick color grading

from montage_ai.parameter_suggester import suggest_color_grading

params = suggest_color_grading(
    scene_description="night city with neon lights",
    user_intent="cyberpunk atmosphere"
)

# Returns: ColorGradingParameters with preset="cool", intensity=0.9, etc.

# 2. Quick stabilization

from montage_ai.parameter_suggester import suggest_stabilization

params = suggest_stabilization(
    shake_score=0.6,
    motion_type="walking"
)

# Returns: StabilizationParameters with smoothing=18, shakiness=6, etc.

```

### Integration in MontageBuilder


```python
from montage_ai.parameter_suggester import ColorGradingSuggester
from montage_ai.core.montage_builder import MontageBuilder

class EnhancedMontageBuilder(MontageBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_suggester = ColorGradingSuggester()
    
    def apply_intelligent_color_grading(self, clip_metadata):
        """Auto-tune color grading based on clip analysis."""
        context = {
            "scene_description": clip_metadata.get("description"),
            "user_intent": self.style_template.get("color_intent", "cinematic"),
            "dominant_colors": clip_metadata.get("color_palette", [])
        }
        
        suggestion = self.color_suggester.suggest(context)
        logger.info(f"AI suggested: {suggestion.parameters['preset']} "
                   f"(confidence: {suggestion.confidence:.2f})")
        logger.info(f"Reasoning: {suggestion.reasoning}")
        
        # Apply suggested parameters
        self.color_grade_config.preset = suggestion.parameters["preset"]
        self.color_grade_config.intensity = suggestion.parameters["intensity"]

```

---

## 🔧 cgpu-Robustness Design

### Problem

cgpu kann ausfallen (Netzwerk, Rate-Limits, API-Änderungen). System muss resilient sein.

### Lösung: Central Fallback-Chain

Alle LLM-Calls gehen durch `CreativeDirector._query_backend()`:

```text
User Request
    ↓
CreativeDirector._query_backend()
    ↓
1. OpenAI-API (KubeAI/vLLM)  [if configured]
    ↓ (fallback on error)
2. cgpu/Gemini             [if CGPU_ENABLED=true]
    ↓ (fallback on error)
3. Google AI Direct        [if GOOGLE_API_KEY set]
    ↓ (fallback on error)
4. Ollama (Local)          [always available]
    ↓
Response or RuntimeError

```

**Test Coverage:**

```bash

# Test cgpu failure scenario

export CGPU_ENABLED=false
python test_parameter_suggester.py

# → Should fallback to Ollama without errors

```

---

## 📈 Impact Analysis

### Before (Manual Tuning)


```python

# Hard-coded magic numbers

stabilization_smoothing = 15  # Why 15? Unknown.
color_preset = "cinematic"    # Always the same
intensity = 0.8               # Fixed value

```

**Problems:**

- ❌ No scene-specific optimization
- ❌ No reasoning/explanation
- ❌ Requires expert knowledge
- ❌ Not adaptable to user intent

### After (AI-Driven Tuning)


```python

# Context-aware intelligent tuning

suggestion = color_suggester.suggest({
    "scene_description": "dark moody interior",
    "user_intent": "film noir atmosphere"
})

# LLM returns:

# {

#   "preset": "noir",

#   "intensity": 0.95,

#   "contrast": 1.4,

#   "reasoning": "High contrast noir preset enhances shadows..."

# }

```

**Benefits:**

- ✅ Scene-specific optimization
- ✅ Explainable decisions (LLM reasoning)
- ✅ Learns from context (histogram, colors, intent)
- ✅ Adaptable to creative direction

---

## 🎯 Next Steps (Phase 2)

### Priority 1: Web UI Integration (6h)

**Goal:** Expose parameter suggestions in web interface


```html
<!-- montage.html -->
<div class="ai-suggestions-panel">
  <h3>🤖 AI Parameter Suggestions</h3>
  
  <div class="suggestion color-grading">
    <label>Color Grading</label>
    <p class="reasoning">{{ suggestion.reasoning }}</p>
    <button onclick="applyColorGrading()">Apply Suggestion</button>
  </div>
  
  <div class="suggestion stabilization">
    <label>Stabilization</label>
    <p class="reasoning">{{ suggestion.reasoning }}</p>
    <button onclick="applyStabilization()">Apply Suggestion</button>
  </div>
</div>

```

**API Endpoint:**

```python

# web_ui/app.py

@app.route('/api/suggest-parameters', methods=['POST'])
def suggest_parameters():
    """Return AI-suggested editing parameters."""
    scene_desc = request.json.get('scene_description')
    user_intent = request.json.get('user_intent')
    
    suggester = ColorGradingSuggester()
    suggestion = suggester.suggest({
        "scene_description": scene_desc,
        "user_intent": user_intent
    })
    
    return jsonify({
        "parameters": suggestion.parameters,
        "reasoning": suggestion.reasoning,
        "confidence": suggestion.confidence
    })

```

### Priority 2: PacingAdvisor (4h)

**Goal:** LLM suggests beats_per_cut overrides for specific sections


```python
class PacingAdvisor(ParameterSuggester):
    """
    Suggests pacing adjustments for different video sections.
    
    Example:
        advisor = PacingAdvisor()
        context = {
            "section": "intro",  # intro/build/climax/outro
            "music_energy": 0.3,
            "user_intent": "slow cinematic build"
        }
        suggestion = advisor.suggest(context)
        # suggestion.parameters = {"beats_per_cut": 8, "pattern": "fibonacci"}
    """

```

### Priority 3: Integration Tests (2h)

**Goal:** End-to-end tests mit echten Clips


```python
def test_e2e_color_grading_suggestion():
    """Test full pipeline: clip analysis → LLM suggestion → FFmpeg application."""
    clip_path = "test_data/input/sunset_beach.mp4"
    
    # 1. Analyze clip
    analyzer = ClipAnalyzer()
    metadata = analyzer.analyze(clip_path)
    
    # 2. Get LLM suggestion
    suggester = ColorGradingSuggester()
    suggestion = suggester.suggest({
        "scene_description": metadata["description"],
        "dominant_colors": metadata["color_palette"]
    })
    
    # 3. Apply color grading
    enhancer = ClipEnhancer()
    output = enhancer.apply_color_grade(
        clip_path,
        preset=suggestion.parameters["preset"],
        intensity=suggestion.parameters["intensity"]
    )
    
    assert output.exists()

```

---

## 📝 Design Decisions

### 1. Warum zentrale Mechanismen?

**Problem:** Wenn jeder Modul eigene LLM-Calls macht:

- 10 Orte mit Backend-Selection-Logic
- 10 Orte mit Fehlerbehandlung
- 10 Orte für cgpu-Fallback-Code
- Inkonsistente Timeouts, Retry-Logic

**Lösung:** `CreativeDirector` als Single Source of Truth

- ✅ Backend-Selection: 1 Ort (`__init__`)
- ✅ Fallback-Logic: 1 Ort (`_query_backend()`)
- ✅ Timeout-Config: 1 Ort (`LLMConfig`)
- ✅ cgpu-Robustheit: Automatisch für alle Suggester

### 2. Warum EditingParameters Schema?

**Problem:** Parameter scattered across modules

- `color_grading.py`: ColorGradeConfig
- `cgpu_jobs/stabilize.py`: Constructor params
- `core/pacing_engine.py`: Various dicts
- Keine zentrale Validierung

**Lösung:** Unified typed schema

- ✅ Single source of truth für Parameter-Ranges
- ✅ JSON-serialisierbar (für API, Storage)
- ✅ Type-safe (Enums, dataclasses)
- ✅ Self-documenting (docstrings)

### 3. Warum LLM statt Heuristiken?

**Heuristik-Ansatz:**

```python
if shake_score > 0.7:
    smoothing = 20
elif shake_score > 0.4:
    smoothing = 15
else:
    smoothing = 10

```

**Probleme:**

- ❌ Rigid rules
- ❌ No context awareness
- ❌ No reasoning
- ❌ Schwer zu maintainen

**LLM-Ansatz:**

```python
suggestion = tuner.suggest({
    "shake_score": 0.7,
    "motion_type": "handheld",  # Context!
    "user_intent": "smooth"      # Intent!
})

```

**Vorteile:**

- ✅ Context-aware (motion_type, resolution, intent)
- ✅ Explainable (reasoning field)
- ✅ Adaptiv (lernt von Prompt)
- ✅ Erweiterbar (neue Faktoren → Prompt-Update)

---

## 🧪 Testing

### Run Tests


```bash
cd /home/codeai/montage-ai
python test_parameter_suggester.py

```

### Expected Output

```text
================================================================================
LLM-BASED PARAMETER SUGGESTION SYSTEM TESTS
Testing cgpu-robust AI director parameter tuning
================================================================================

================================================================================
TEST 1: Color Grading Suggestion
================================================================================

--- Scene 1: Sunset Beach ---
Suggested Preset: golden_hour
Intensity: 0.90
Temperature: 0.30
Saturation: 1.20
Confidence: 0.85
Reasoning: Golden hour preset enhances warm sunset tones with increased
saturation to emphasize orange/yellow hues. Positive temperature shift adds
warmth. High confidence due to clear scene characteristics.

--- Scene 2: Night City ---
Suggested Preset: cool
Intensity: 0.85
Temperature: -0.40
Confidence: 0.80
Reasoning: Cool preset with negative temperature shift creates cyberpunk
atmosphere. Desaturated look enhances neon lights contrast.

[... weitere Tests ...]

================================================================================
ALL TESTS COMPLETED
================================================================================

```

---

## 📚 References

### Related Files

- `src/montage_ai/editing_parameters.py` - Parameter schema
- `src/montage_ai/parameter_suggester.py` - LLM suggester system
- `src/montage_ai/creative_director.py` - LLM backend (updated)
- `src/montage_ai/color_grading.py` - Color grading implementation
- `src/montage_ai/cgpu_jobs/stabilize.py` - Stabilization job
- `test_parameter_suggester.py` - Test suite

### External Research

- DirectorLLM: LLM-based cinematography orchestration
- Descript Underlord: Conversational video editing
- LAVE: Structured JSON for video editing agents

---

## ✅ Success Criteria

**Phase 1 (Done):**

- [x] Zentrale Parameter-Schema erstellt
- [x] ColorGradingSuggester implementiert
- [x] StabilizationTuner implementiert
- [x] CreativeDirector._query_backend() hinzugefügt
- [x] cgpu-Robustheit getestet
- [x] Convenience functions für quick usage
- [x] Test suite erstellt

**Phase 2 (Planned):**

- [ ] Web UI integration (Suggestions Panel)
- [ ] PacingAdvisor implementiert
- [ ] End-to-end tests mit echten Clips
- [ ] Performance benchmarks (LLM latency)
- [ ] Dokumentation für User (README update)

---

## 📤 Export to NLE (CLI + API)

**Status:** ✅ Implemented (Phase 1.5)

### Overview

Montage AI kann Timelines jetzt zu professionellen NLE-Formaten exportieren:

- **OTIO** (OpenTimelineIO) - Canonical format mit vollem Metadata
- **EDL** (CMX 3600) - Kompatibilität mit allen Legacy-Systemen
- **Premiere XML** - Adobe Premiere Pro
- **AAF** - Avid Media Composer
- **JSON Parameters** - Roundtrip: Export → NLE-Edit → Re-import

### CLI Usage

```bash
# Export zu OTIO (Standard)
./montage-ai.sh export-to-nle --manifest /data/output/manifest.json

# Export zu mehreren Formaten
./montage-ai.sh export-to-nle --manifest /data/output/manifest.json \
  --formats otio edl premiere aaf \
  --project-name "My Project" \
  --output-dir /data/output

# Mit EditingParameters JSON
./montage-ai.sh export-to-nle \
  --manifest /data/output/manifest.json \
  --params /data/output/parameters.json
```

### Python API

```python
from montage_ai.export import export_to_nle, create_export_summary
from montage_ai.export.otio_builder import TimelineClipInfo
from montage_ai.editing_parameters import EditingParameters
from pathlib import Path

# Prepare clips
clips = [
    TimelineClipInfo(
        source_path="/data/input/clip1.mp4",
        in_time=0.0,
        out_time=5.0,
        duration=5.0,
        sequence_number=1,
        applied_effects={
            "color_grading": {"preset": "teal_orange", "intensity": 0.9},
            "stabilization": {"smoothing": 20}
        },
        confidence_scores={"color_grading": 0.85}
    ),
    # ... more clips
]

# Export
params = EditingParameters()
results = export_to_nle(
    timeline_clips=clips,
    editing_params=params,
    output_dir=Path("/data/output"),
    formats=["otio", "edl", "premiere"],
    project_name="My Montage"
)

# Summary
print(create_export_summary(results))
```

### Manifest Format

Timeline manifest JSON (from MontageBuilder):

```json
{
  "clips": [
    {
      "source_path": "/data/input/clip1.mp4",
      "in_time": 0.0,
      "out_time": 5.0,
      "duration": 5.0,
      "sequence_number": 1,
      "applied_effects": {
        "color_grading": {"preset": "teal_orange"},
        "stabilization": {"smoothing": 20}
      },
      "recommended_effects": {...},
      "confidence_scores": {"color_grading": 0.85}
    }
  ],
  "beat_timecodes": [[1.0, "beat_1"], [2.0, "beat_2"]],
  "section_markers": [[0.0, "intro"], [2.0, "build"], [4.0, "climax"]]
}
```

### Metadata Attachment

Alle Effects werden als Clip-Metadaten exportiert:

```
Clip Metadata (OTIO):
├── montage_ai.applied_effects        # Effects already applied (baked)
├── montage_ai.recommended_effects    # Suggestions for NLE user
├── montage_ai.confidence_scores      # LLM confidence per effect
├── montage_ai.beat_markers           # Pacing markers
└── notes                             # Human-readable descriptions
```

**Color Grading Example:**
- Applied: "Color Grading preset=teal_orange, intensity=0.9"
- Recommended: "Consider desaturation for dramatic effect"
- Confidence: 0.85

### Roundtrip Workflow

1. **Render:** `./montage-ai.sh run --export` → generates manifest.json + parameters.json
2. **Export:** `./montage-ai.sh export-to-nle` → OTIO/EDL/Premiere/AAF
3. **Import:** Load OTIO in Premiere/Resolve → metadata preserved
4. **Edit:** Adjust parameters, re-export JSON
5. **Re-import:** Load parameters.json back to Montage AI for re-render

---

## 🧪 Tests & CLI

**Status:** ✅ Implemented (Phase 1.5)

### Test Coverage

```bash
# Run OTIO export tests
pytest tests/test_otio_export.py -v

# Test JSON robustness
pytest tests/test_otio_export.py::TestJSONParsingRobustness -v
```

**Test scenarios:**
- ✅ Timeline creation
- ✅ Multi-clip handling
- ✅ Metadata attachment
- ✅ Beat + section markers
- ✅ Multi-format export (OTIO, EDL, Premiere, AAF)
- ✅ JSON parameter serialization
- ✅ Malformed JSON recovery

### CLI Examples

```bash
# Help
./montage-ai.sh export-to-nle --help

# Quick export (default OTIO + EDL)
./montage-ai.sh export-to-nle --manifest /data/output/manifest.json

# All formats with verbose logging
./montage-ai.sh export-to-nle \
  --manifest /data/output/manifest.json \
  --formats otio edl premiere aaf params_json \
  --project-name "Feature Film" \
  --verbose

# Custom FPS/resolution
./montage-ai.sh export-to-nle \
  --manifest /data/output/manifest.json \
  --fps 25.0 \
  --width 3840 --height 2160 \
  --formats premiere
```

---

## 📋 Updated Changelog

- **2026-01-09:** 
  - ✅ Export to NLE: OTIO Builder + CLI (`export-to-nle`)
  - ✅ Parser robustness: retry logic + JSON fallback + safe defaults
  - ✅ Tests for OTIO export, JSON parsing
  - ✅ Markdown lint fixes (all docs)

---

## 🎬 Conclusion

Das System ist jetzt **produktionsbereit** für AI-gesteuerte Parameter-Optimierung. Alle Suggester nutzen **zentrale Mechanismen** (CreativeDirector) und sind **cgpu-robust** (auto-fallback zu Ollama).

**Key Achievements:**

1. ✅ **50+ Parameter** zentral definiert und validierbar
2. ✅ **2 Suggester** implementiert (Color Grading, Stabilization)
3. ✅ **cgpu-Integration** robust mit automatischem Fallback
4. ✅ **Zero Fragmentation** - Alle LLM-Calls durch CreativeDirector
5. ✅ **Explainable AI** - LLM liefert Reasoning für Decisions
6. ✅ **Export to NLE** - OTIO/EDL/Premiere/AAF mit vollem Metadata
7. ✅ **CLI Ready** - `./montage-ai.sh export-to-nle` command
8. ✅ **Robust Parser** - Retry logic, JSON extraction fallback, safe defaults

**Next:** Web UI Integration + Cluster validation.

