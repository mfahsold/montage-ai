# ML Enhancement Roadmap

Iterative Verbesserungen für professionellere AI-gestützte Schnitte.

---

## 🎯 Ziel

Von "heuristischer Clip-Selection" zu "intelligenter, kontextbewusster Regie mit ML/LLM"

---

## 📊 Aktuelle Situation

**Clip Selection (editor.py:1200-1350):**
```python
# Heuristischer Score
for scene in candidates:
    score = 0
    score += energy_match * 50
    score += action_level * 30
    score += random_variation * 10

    best_scene = max(scenes, key=lambda s: s['score'])
```

**Probleme:**
- Keine semantische Analyse (was ist im Clip?)
- Keine Continuity-Checks (passt der Übergang?)
- Kein "Creative Reasoning" (warum dieser Clip?)
- Keine Shot-Composition-Analyse

---

## 🚀 Iterative Verbesserungen (Klein → Groß)

### ✅ Phase 0: Baseline (Aktuell)
- LLM für Style-Interpretation (1x beim Start)
- Heuristische Clip-Selection
- Beat-synced Cutting
- Energy-matching

**Metrics:** Funktioniert, aber mechanisch

---

### 🎯 Phase 1: Intelligent Clip Selection mit LLM Reasoning (QUICK WIN)

**Dauer:** 1-2 Stunden | **Complexity:** Low | **Impact:** High

**Konzept:**
```python
# Statt:
best_scene = max(scenes, key=lambda s: heuristic_score(s))

# Neu:
candidates = get_top_3_by_heuristic(scenes)
llm_ranking = llm.rank_clips(
    candidates=candidates,
    context={
        "style": "hitchcock",
        "previous_clips": last_3_clips,
        "current_energy": 0.72,
        "beat_position": "climax"
    }
)
best_scene = llm_ranking[0]  # LLM's favorite
```

**LLM Prompt:**
```
You are a professional film editor working on a ${style} montage.

Context:
- Style: ${style} (suspenseful, slow builds)
- Previous clips: ${describe_last_3_clips}
- Current energy: ${energy} (high/medium/low)
- Position: ${position} (intro/build/climax/outro)

Clip candidates:
1. Scene A: ${describe_scene_A}
2. Scene B: ${describe_scene_B}
3. Scene C: ${describe_scene_C}

Rank these clips (1=best, 3=worst) and explain why.
Consider:
- Continuity with previous clips
- Match with current energy/mood
- Visual variety
- ${style}-specific aesthetics

Response format:
{
  "ranking": [
    {"clip": "A", "score": 95, "reason": "High-action close-up creates tension after wide establishing shot"},
    {"clip": "C", "score": 78, "reason": "Good energy match but too similar to previous clip"},
    {"clip": "B", "score": 45, "reason": "Low energy doesn't match climax position"}
  ]
}
```

**Benefits:**
- ✅ Bessere Clip-Auswahl durch Kontext-Verständnis
- ✅ Nachvollziehbare Entscheidungen (Reasoning)
- ✅ Kann in Logs/UI angezeigt werden
- ✅ Vorbereitung für komplexere LLM-Integration

**Implementation:**
- Neues Modul: `clip_selector.py`
- Integration: `editor.py` ruft `select_best_clip()` auf
- Fallback: Bei LLM-Fehler → heuristischer Score

**Latency:** ~500-1000ms pro Clip-Selection
**Mitigation:** Nur für wichtige Cuts (z.B. alle 5. Clip) oder async

---

### 🎯 Phase 2: Scene Understanding mit Vision Models

**Dauer:** 2-3 Stunden | **Complexity:** Medium | **Impact:** High

**Konzept:** CLIP/BLIP für semantische Szenen-Analyse

```python
from transformers import CLIPProcessor, CLIPModel

# Extract frame from clip
frame = extract_middle_frame(clip_path)

# Analyze with CLIP
features = clip_model.encode_image(frame)
tags = clip_model.classify(frame, labels=[
    "person", "landscape", "object", "close-up",
    "wide shot", "action", "calm", "indoor", "outdoor"
])

# Use in scoring
scene['semantic_tags'] = tags
scene['visual_features'] = features

# Better matching
if style == "wes_anderson" and "symmetry" in tags:
    score += 50
```

**Benefits:**
- ✅ Versteht WAS im Clip ist (nicht nur Energy)
- ✅ Shot-Type-Erkennung (Close-up, Wide, Medium)
- ✅ Bessere Style-Matching (z.B. Symmetrie für Wes Anderson)
- ✅ Semantische Ähnlichkeit zwischen Clips

**Models:**
- OpenAI CLIP (für Zero-Shot Classification)
- BLIP-2 (für Image Captioning)
- YOLOv8 (für Object Detection, optional)

**Latency:** ~100-300ms pro Frame (GPU) oder ~1-2s (CPU)

---

### 🎯 Phase 3: Shot Composition Analysis

**Dauer:** 3-4 Stunden | **Complexity:** Medium | **Impact:** Medium

**Konzept:** Analyze framing quality

```python
# Rule of Thirds
def analyze_composition(frame):
    thirds_grid = detect_rule_of_thirds_alignment(frame)
    subject_position = detect_main_subject(frame)
    visual_balance = calculate_visual_weight_distribution(frame)

    return {
        "thirds_score": 0.85,  # How well does it follow rule of thirds?
        "balance_score": 0.92,  # Visual balance
        "subject_clarity": 0.78  # Is subject clear?
    }

# Use in selection
if scene['composition']['thirds_score'] > 0.8:
    score += 20  # Reward good composition
```

**Benefits:**
- ✅ Bevorzugt gut komponierte Shots
- ✅ Professionelleres Aussehen
- ✅ Kann in UI als "Quality Score" gezeigt werden

**Techniques:**
- Rule of Thirds detection
- Leading Lines detection
- Symmetry analysis
- Subject isolation (Saliency Maps)

---

### 🎯 Phase 4: Continuity & Flow Optimization (LLM-gestützt)

**Dauer:** 4-6 Stunden | **Complexity:** High | **Impact:** Very High

**Konzept:** LLM prüft Story-Flow über mehrere Clips

```python
# Nach jedem Cut
continuity_check = llm.check_continuity(
    previous_clips=[clip1, clip2, clip3],
    current_clip=candidate,
    style="hitchcock"
)

if continuity_check['score'] < 0.7:
    # Schlechter Übergang - probiere anderen Clip
    candidate = get_next_best_candidate()
```

**LLM Prompt:**
```
You are reviewing a sequence of clips for a ${style} montage.

Previous clips:
1. Wide shot of person walking (3s) - Energy: 0.5
2. Close-up of face looking worried (2s) - Energy: 0.6
3. Medium shot of door opening (2.5s) - Energy: 0.7

Proposed next clip:
4. Wide shot of empty room (2s) - Energy: 0.4

Evaluate:
1. Visual continuity (does the transition make sense?)
2. Emotional flow (does mood progression work?)
3. Pacing (is timing appropriate?)
4. ${style} adherence (matches style guidelines?)

Response:
{
  "continuity_score": 0.65,
  "issues": ["Energy drop too sudden", "Empty room breaks tension"],
  "suggestions": ["Use tighter shot", "Increase clip duration"],
  "approved": false
}
```

**Benefits:**
- ✅ Story-Flow-Optimierung
- ✅ Vermeidung von Jump Cuts
- ✅ Bessere emotionale Progression
- ✅ Style-Adherence über Zeit

---

### 🎯 Phase 5: Advanced Beat-Syncing mit ML

**Dauer:** 3-5 Stunden | **Complexity:** Medium | **Impact:** Medium

**Konzept:** Präzisere Beat-Detection + Transient-Analysis

```python
import librosa
import madmom  # State-of-the-art beat tracking

# Madmom RNN-based beat tracker (bessere Genauigkeit)
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

# Analyze
signal, sr = librosa.load(audio_path)
beat_processor = DBNBeatTrackingProcessor(fps=100)
beats = beat_processor(signal)

# Transient detection (für snappy cuts)
onsets = librosa.onset.onset_detect(
    y=signal, sr=sr, units='time',
    backtrack=True  # Snap to nearest zero-crossing
)

# Use for ultra-precise cuts
cut_time = find_nearest_zero_crossing(beat_time, signal)
```

**Benefits:**
- ✅ Genauere Beat-Detection
- ✅ Snap to transients (snappier cuts)
- ✅ Besser für komplexe Rhythmen

---

### 🎯 Phase 6: Color Harmony Analysis

**Dauer:** 2-3 Stunden | **Complexity:** Low | **Impact:** Medium

**Konzept:** Match color palettes between clips

```python
from sklearn.cluster import KMeans

def extract_color_palette(frame):
    pixels = frame.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_  # Dominant colors

# Compare palettes
def color_similarity(palette1, palette2):
    # Use Delta-E color distance
    distances = [delta_e(c1, c2) for c1, c2 in zip(palette1, palette2)]
    return 1.0 - (sum(distances) / len(distances) / 100)

# Use in scoring
color_sim = color_similarity(previous_clip.palette, candidate.palette)
if color_sim > 0.7:
    score += 15  # Reward harmonious transitions
```

**Benefits:**
- ✅ Smooth color transitions
- ✅ Professioneller Look
- ✅ Style-Consistency

---

### 🎯 Phase 7: Multi-Modal Analysis (Vision + Audio + LLM)

**Dauer:** 1-2 Wochen | **Complexity:** Very High | **Impact:** Revolutionary

**Konzept:** Unified model versteht Video + Audio + Context

```python
# Multimodal embedding
embedding = multimodal_model.encode(
    video=clip_frames,
    audio=clip_audio,
    context={"style": "hitchcock", "position": "climax"}
)

# Semantic similarity in latent space
similarity = cosine_similarity(embedding_prev, embedding_candidate)

# LLM gets multimodal context
decision = llm.decide_cut(
    visual_features=clip.visual_embedding,
    audio_features=clip.audio_embedding,
    style_embedding=style.embedding,
    previous_context=history_embedding
)
```

**Benefits:**
- ✅ Holistisches Verständnis von Video+Audio
- ✅ Kann komplexe Patterns erkennen
- ✅ "Intuitive" Entscheidungen wie menschlicher Editor

---

## 📈 Priorisierung nach ROI

| Phase | Aufwand | Impact | ROI | Priority |
|-------|---------|--------|-----|----------|
| Phase 1: LLM Reasoning | 1-2h | High | ⭐⭐⭐⭐⭐ | 1 |
| Phase 6: Color Harmony | 2-3h | Medium | ⭐⭐⭐⭐ | 2 |
| Phase 2: Scene Understanding | 2-3h | High | ⭐⭐⭐⭐ | 3 |
| Phase 5: Beat-Syncing | 3-5h | Medium | ⭐⭐⭐ | 4 |
| Phase 3: Composition | 3-4h | Medium | ⭐⭐⭐ | 5 |
| Phase 4: Continuity | 4-6h | Very High | ⭐⭐⭐⭐ | 6 |
| Phase 7: Multimodal | 1-2w | Revolutionary | ⭐⭐⭐⭐⭐ | 7 |

---

## 🎬 Empfohlener Start: Phase 1

**Nächste Schritte:**
1. Erstelle `clip_selector.py` mit LLM-Reasoning
2. Integriere in `editor.py` (optional toggle)
3. Teste mit einem Job
4. Zeige Reasoning in Logs/UI
5. Iteriere basierend auf Ergebnissen

**Estimated Time:** 1-2 Stunden
**Risk:** Low (Fallback zu heuristischem Score)
**Benefit:** Sofort sichtbare Verbesserung + bessere Nachvollziehbarkeit
