# LLM Workflow - Konzeptionelle Übersicht

Wie Textmodelle die Video-Regie steuern.

---

## 🎯 Gesamtflow (High-Level)

```
User Input → Keyword Match → LLM Call → JSON Response → Validation →
Editing Engine → Video Assembly → Final Output
```

---

## 📍 1. Einstiegspunkt: User Input

**Wo:** Web UI oder Environment Variable

```javascript
// In Web UI (index.html)
<input id="prompt" placeholder="e.g., 'Edit like a thriller with slow builds'">

// Wird gesendet als:
POST /api/jobs
{
  "style": "dynamic",
  "prompt": "Edit like a thriller with slow builds",
  "options": {...}
}
```

**Im Backend:**
```python
# In app.py:260-290
def api_create_job():
    data = request.json
    options = {
        "prompt": data.get('prompt', ''),  # User's creative prompt
        "stabilize": data.get('stabilize', False),
        ...
    }

    # Set environment for montage process
    env["CREATIVE_PROMPT"] = options.get("prompt", "")
    env["CUT_STYLE"] = data['style']
```

---

## 📍 2. Startup: Prompt Interpretation

**Wo:** `editor.py:330-420` - `interpret_creative_prompt()`

**Ablauf:**

```python
# editor.py:359-361
if CREATIVE_PROMPT and CREATIVE_DIRECTOR_AVAILABLE:
    print(f"\n🎯 Creative Prompt: '{CREATIVE_PROMPT}'")
    EDITING_INSTRUCTIONS = interpret_natural_language(CREATIVE_PROMPT)
```

**Was passiert:**
1. Prüft ob Creative Director verfügbar ist
2. Ruft `interpret_natural_language()` auf
3. Speichert Ergebnis in globale Variable `EDITING_INSTRUCTIONS`

---

## 📍 3. LLM Call Flow

**Wo:** `creative_director.py:223-305` - `interpret_prompt()`

### Phase 1: Keyword Matching (Schnellster Weg - Kein LLM!)

```python
# creative_director.py:243-285
user_lower = user_prompt.lower()

# Direct style template match
for style_name in list_available_styles():
    if style_name in user_lower:
        print(f"   🎯 Detected style template: {style_name}")
        template = get_style_template(style_name)
        return template['params']  # ✅ DONE! No LLM needed

# Extended keyword matching
style_keywords = {
    'hitchcock': ['hitchcock', 'suspense', 'thriller', 'tension'],
    'action': ['action', 'blockbuster', 'explosive', 'michael bay'],
    'mtv': ['mtv', 'music video', 'fast-paced'],
    ...
}

for style_name, keywords in style_keywords.items():
    for keyword in keywords:
        if keyword in user_lower:
            return get_style_template(style_name)  # ✅ DONE!
```

**Ergebnis:** ~80% der Prompts werden hier bereits resolved (1ms Latency)

### Phase 2: LLM Call (Fallback für komplexe Prompts)

**Nur wenn kein Keyword-Match gefunden wurde:**

```python
# creative_director.py:288-305
response = self._query_llm(user_prompt)  # 🤖 LLM CALL HERE
instructions = self._parse_and_validate(response)
```

---

## 📍 4. LLM Backend Selection

**Wo:** `creative_director.py:307-324` - `_query_llm()`

**Priority Chain:**
```
1. Google AI (Gemini 2.0 Flash) → 500-2000ms
2. cgpu serve (Gemini proxy) → 1000-3000ms
3. Ollama (Local LLM) → 5000-15000ms
4. Default Instructions → 0ms (Fallback)
```

### Google AI Call (Bevorzugt)

**Wo:** `creative_director.py:325-397` - `_query_google_ai()`

```python
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

payload = {
    "contents": [{
        "parts": [{
            "text": f"{self.system_prompt}\n\nUser request: {user_prompt}"
        }]
    }],
    "generationConfig": {
        "temperature": 0.3,           # Low = consistent
        "topP": 0.9,
        "maxOutputTokens": 1024,
        "responseMimeType": "application/json"  # ⭐ Force JSON!
    }
}

headers = {
    "Content-Type": "application/json",
    "X-goog-api-key": GOOGLE_API_KEY
}

response = requests.post(url, json=payload, headers=headers, timeout=60)
```

**Wichtig:** `responseMimeType: "application/json"` zwingt Gemini, nur valides JSON zurückzugeben!

---

## 📍 5. System Prompt (Was das LLM sieht)

**Wo:** `creative_director.py:52-143` - `DIRECTOR_SYSTEM_PROMPT`

**Struktur:**

```python
DIRECTOR_SYSTEM_PROMPT = """You are the Creative Director for the Fluxibri video editing system.

Your role: Translate natural language editing requests into structured JSON editing instructions.

Available cinematic styles:
- dynamic: Position-aware pacing (default)
- hitchcock: Suspense - slow build, fast climax
- mtv: Rapid 1-2 beat cuts
- action: Michael Bay fast cuts
- documentary: Natural, observational
- minimalist: Contemplative long takes
- wes_anderson: Symmetric, stylized

You MUST respond with ONLY valid JSON matching this structure:
{
  "style": {
    "name": "hitchcock" | "mtv" | "action" | ...,
    "mood": "suspenseful" | "playful" | "energetic" | ...
  },
  "pacing": {
    "speed": "very_slow" | "slow" | "medium" | "fast" | "very_fast" | "dynamic",
    "variation": "minimal" | "moderate" | "high" | "fibonacci",
    "intro_duration_beats": 4-32,
    "climax_intensity": 0.0-1.0
  },
  "cinematography": {
    "prefer_wide_shots": true | false,
    "prefer_high_action": true | false,
    "match_cuts_enabled": true | false,
    "invisible_cuts_enabled": true | false,
    "shot_variation_priority": "low" | "medium" | "high"
  },
  "transitions": {
    "type": "hard_cuts" | "crossfade" | "mixed" | "energy_aware",
    "crossfade_duration_sec": 0.1-2.0
  },
  "energy_mapping": {
    "sync_to_beats": true | false,
    "energy_amplification": 0.5-2.0
  },
  "effects": {
    "color_grading": "none" | "neutral" | "warm" | "cool" | "high_contrast" | "desaturated" | "vibrant",
    "stabilization": true | false,
    "upscale": true | false,
    "sharpness_boost": true | false
  },
  "constraints": {
    "target_duration_sec": null | number,
    "min_clip_duration_sec": 0.5-10.0,
    "max_clip_duration_sec": 2.0-60.0
  }
}

Examples:

User: "Edit this like a Hitchcock thriller"
Response:
{
  "style": {"name": "hitchcock", "mood": "suspenseful"},
  "pacing": {"speed": "dynamic", "variation": "high", "intro_duration_beats": 16, "climax_intensity": 0.9},
  "cinematography": {"prefer_wide_shots": false, "prefer_high_action": true, "match_cuts_enabled": true},
  "transitions": {"type": "hard_cuts"}
}

CRITICAL RULES:
1. Return ONLY valid JSON - no markdown, no explanations
2. Use predefined styles when possible
3. Be conservative with effects (stabilization/upscale are slow!)
4. Match the user's creative intent while staying technically feasible
"""
```

**User Prompt wird hinzugefügt:**
```
System Prompt + "\n\nUser request: Edit like a thriller with slow builds"
```

---

## 📍 6. LLM Response (Was zurückkommt)

**Gemini 2.0 Flash Response:**

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "{\"style\":{\"name\":\"hitchcock\",\"mood\":\"suspenseful\"},\"pacing\":{\"speed\":\"dynamic\",\"variation\":\"high\",\"intro_duration_beats\":16,\"climax_intensity\":0.9},\"cinematography\":{\"prefer_wide_shots\":false,\"prefer_high_action\":true,\"match_cuts_enabled\":true,\"invisible_cuts_enabled\":true,\"shot_variation_priority\":\"high\"},\"transitions\":{\"type\":\"hard_cuts\",\"crossfade_duration_sec\":0.3},\"effects\":{\"color_grading\":\"high_contrast\",\"sharpness_boost\":true}}"
          }
        ]
      }
    }
  ]
}
```

**Extraktion:**
```python
# creative_director.py:364-380
result = response.json()
candidates = result.get("candidates", [])
content = candidates[0].get("content", {})
parts = content.get("parts", [])
text = parts[0].get("text", "")

# Clean up (manchmal wraps Gemini in ```json...```)
if text.startswith("```json"):
    text = text[7:]
if text.endswith("```"):
    text = text[:-3]

return text.strip()  # Pure JSON string
```

---

## 📍 7. JSON Parsing & Validation

**Wo:** `creative_director.py:477-514` - `_parse_and_validate()`

```python
def _parse_and_validate(self, llm_response: str) -> Optional[Dict[str, Any]]:
    try:
        # Parse JSON
        instructions = json.loads(llm_response)

        # Required fields validation
        if "style" not in instructions or "pacing" not in instructions:
            print(f"   ⚠️ Missing required fields (style/pacing)")
            return None

        # Validate style name
        style_name = instructions["style"].get("name")
        valid_styles = ["hitchcock", "mtv", "action", "documentary", ...]
        if style_name not in valid_styles:
            print(f"   ⚠️ Invalid style name: {style_name}")
            return None

        # ✅ Valid JSON
        return instructions

    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON from LLM: {e}")
        return None
```

**Bei Fehler:** Fallback zu Default Instructions

---

## 📍 8. Storage: Global Variable

**Wo:** `editor.py:127` & `editor.py:361`

```python
# Global variable (module-level)
EDITING_INSTRUCTIONS = None  # Will be populated at startup

# Startup (called once)
def interpret_creative_prompt():
    global EDITING_INSTRUCTIONS

    if CREATIVE_PROMPT:
        EDITING_INSTRUCTIONS = interpret_natural_language(CREATIVE_PROMPT)

        if EDITING_INSTRUCTIONS:
            print(f"   ✅ Style Applied: {EDITING_INSTRUCTIONS['style']['name']}")

            # Verbose mode: Show full details
            if VERBOSE:
                print(f"\n📋 STYLE TEMPLATE DETAILS:")
                print(f"   Style Name:       {EDITING_INSTRUCTIONS['style']['name']}")
                print(f"   Pacing Speed:     {EDITING_INSTRUCTIONS['pacing']['speed']}")
                print(f"   Transitions:      {EDITING_INSTRUCTIONS['transitions']['type']}")
                ...
```

**Ergebnis:** `EDITING_INSTRUCTIONS` ist jetzt ein Dict wie:

```python
{
  "style": {"name": "hitchcock", "mood": "suspenseful"},
  "pacing": {
    "speed": "dynamic",
    "variation": "high",
    "intro_duration_beats": 16,
    "climax_intensity": 0.9
  },
  "cinematography": {
    "prefer_wide_shots": False,
    "prefer_high_action": True,
    "match_cuts_enabled": True,
    "invisible_cuts_enabled": True,
    "shot_variation_priority": "high"
  },
  "transitions": {
    "type": "hard_cuts",
    "crossfade_duration_sec": 0.3
  },
  "effects": {
    "color_grading": "high_contrast",
    "sharpness_boost": True
  }
}
```

---

## 📍 9. Verwendung im Editing Engine

**Wo:** Überall in `editor.py` während Video-Assembly

### 9.1 Pacing Control

**Wo:** `editor.py:1100-1180` - Cut Duration Berechnung

```python
# Check if we have Creative Director instructions
if EDITING_INSTRUCTIONS is not None:
    pacing = EDITING_INSTRUCTIONS.get('pacing', {})
    speed = pacing.get('speed', 'dynamic')

    if speed == 'very_fast':
        beats_per_cut = 1  # 1 beat per cut
    elif speed == 'fast':
        beats_per_cut = 2
    elif speed == 'medium':
        beats_per_cut = 4
    elif speed == 'slow':
        beats_per_cut = 8
    elif speed == 'very_slow':
        beats_per_cut = 16
    elif speed == 'dynamic':
        # Position-aware: slow intro → fast middle → medium outro
        position = beat_idx / len(beat_times)
        if position < 0.2:      # Intro (20%)
            beats_per_cut = 8
        elif position < 0.7:    # Main (50%)
            beats_per_cut = 2
        else:                   # Outro (30%)
            beats_per_cut = 4

    # Calculate cut duration
    cut_duration = (beat_times[beat_idx + beats_per_cut] - beat_times[beat_idx])
```

### 9.2 Transition Control

**Wo:** `editor.py:1491-1526` - Crossfade vs Hard Cuts

```python
# 🎬 CREATIVE DIRECTOR INTEGRATION: Transitions control
if EDITING_INSTRUCTIONS is not None:
    transitions = EDITING_INSTRUCTIONS.get('transitions', {})
    transition_type = transitions.get('type', 'energy_aware')
    crossfade_duration_sec = transitions.get('crossfade_duration_sec', 0.5)

    if transition_type == "crossfade":
        # Always crossfade
        if len(clips) > 0:
            fade_duration = min(crossfade_duration_sec, cut_duration * 0.3)
            v_clip = v_clip.crossfadein(fade_duration)
            clips[-1] = clips[-1].crossfadeout(fade_duration)

    elif transition_type == "mixed":
        # Random crossfade (50% chance)
        if len(clips) > 0 and random.random() > 0.5:
            fade_duration = min(crossfade_duration_sec, cut_duration * 0.3)
            v_clip = v_clip.crossfadein(fade_duration)
            clips[-1] = clips[-1].crossfadeout(fade_duration)

    elif transition_type == "energy_aware":
        # Crossfade on low energy scenes
        if len(clips) > 0 and current_energy < 0.3:
            fade_duration = min(crossfade_duration_sec, cut_duration * 0.2)
            v_clip = v_clip.crossfadein(fade_duration)
            clips[-1] = clips[-1].crossfadeout(fade_duration)

    # "hard_cuts" = no crossfade (skip all above)
```

### 9.3 Cinematography (Clip Selection)

**Wo:** `editor.py:1200-1350` - Scene Selection Logic

```python
# Score each candidate scene
for scene in candidate_scenes:
    score = 0

    # 🎬 CREATIVE DIRECTOR: Prefer wide shots?
    if EDITING_INSTRUCTIONS:
        cinematography = EDITING_INSTRUCTIONS.get('cinematography', {})

        if cinematography.get('prefer_wide_shots', False):
            # Boost score for wide shots (based on resolution ratio)
            if scene['meta']['shot'] == 'wide':
                score += 20

        if cinematography.get('prefer_high_action', False):
            # Boost score for high-action scenes
            if scene['meta']['action'] == 'high':
                score += 30

    # Energy matching
    energy_diff = abs(scene['meta']['energy'] - current_energy)
    score += (1.0 - energy_diff) * 50  # Lower diff = higher score

    # ...more scoring logic...

    # Select scene with highest score
    best_scene = max(candidate_scenes, key=lambda s: s['score'])
```

### 9.4 Effects Control

**Wo:** `editor.py:1380-1450` - Enhancement Pipeline

```python
# 🎬 CREATIVE DIRECTOR: Effects control
if EDITING_INSTRUCTIONS:
    effects = EDITING_INSTRUCTIONS.get('effects', {})

    # Override enhancement settings based on style
    if effects.get('stabilization', False):
        stabilize_applied = True
        v_clip = stabilize_clip(temp_clip_path, stabilized_path)

    if effects.get('upscale', False):
        upscale_applied = True
        v_clip = upscale_clip(temp_clip_path, upscaled_path)

    if effects.get('sharpness_boost', True):
        enhance_applied = True
        v_clip = enhance_clip(temp_clip_path, enhanced_path)
```

---

## 📍 10. Monitoring & Logging

**Wo:** Console Output während Rendering

```bash
🎬 Montage AI v0.3.0
================================================================
🎯 Creative Prompt: 'Edit like a thriller with slow builds'
   🎯 Keyword match 'thriller' → hitchcock
   ✅ Style Applied: hitchcock

📋 STYLE TEMPLATE DETAILS:
   Style Name:       hitchcock
   Pacing Speed:     dynamic
   Cut Duration:     0.5-3.0s
   Transitions:      hard_cuts
   Color Grading:    high_contrast

────────────────────────────────────────────────────────────
📍 PHASE: ASSEMBLING
────────────────────────────────────────────────────────────
   ┌─ Cut #1 [█░░░░░░░░░░░░░░] 5%
   │  📹 VID_001.mp4
   │  ⏱️  2.50s → 4.80s (2.30s)
   │  🎵 Beat 4 + 2 beats
   │  ⚡ Energy: 0.72 | Score: 85
   └─ 💡 High-action scene matches energy peak (hitchcock style)

   ┌─ Cut #2 [██░░░░░░░░░░░░░] 10%
   │  📹 VID_003.mp4
   │  ⏱️  1.20s → 3.60s (2.40s)
   │  🎵 Beat 6 + 2 beats
   │  ⚡ Energy: 0.68 | Score: 82
   └─ 💡 Tension build-up (dynamic pacing)
   │  🔄 Transition: hard_cuts - high-energy scene
```

---

## 📍 11. Verfügbarkeit in Web UI

**Neue API Endpoints:**

```javascript
// Get Creative Instructions for a job
GET /api/jobs/20240315_120000/creative-instructions

Response:
{
  "job_id": "20240315_120000",
  "creative_prompt": "Edit like a thriller with slow builds",
  "style": "hitchcock",
  "options": {
    "prompt": "Edit like a thriller with slow builds",
    "enhance": true,
    "stabilize": false,
    "upscale": false
  }
}
```

**Im Job Details Modal:**

```
🎬 Creative Director Instructions
┌─────────────────────────────────────────┐
│ Prompt:   "Edit like a thriller..."    │
│ Style:    hitchcock                     │
│ Options:  enhance                       │
└─────────────────────────────────────────┘
```

---

## 🎯 Zusammenfassung: Datenfluss

```
1. User Input (Web UI)
   ↓
   "Edit like a thriller with slow builds"

2. Keyword Matching (creative_director.py)
   ↓
   🎯 Match 'thriller' → hitchcock template
   ⏱️ Latency: ~1ms

3. (Alternative: LLM Call)
   ↓
   🤖 Gemini API Call
   📤 System Prompt + User Prompt
   📥 JSON Response
   ⏱️ Latency: ~500-2000ms

4. Validation (creative_director.py)
   ↓
   ✅ JSON parsed & validated

5. Storage (editor.py)
   ↓
   EDITING_INSTRUCTIONS = {
     "style": {"name": "hitchcock", ...},
     "pacing": {"speed": "dynamic", ...},
     "transitions": {"type": "hard_cuts", ...},
     ...
   }

6. Video Assembly (editor.py)
   ↓
   ├─ Pacing: Calculate cut durations based on speed
   ├─ Transitions: Apply hard_cuts (no crossfade)
   ├─ Cinematography: Prefer high-action scenes
   └─ Effects: Apply high_contrast color grading

7. Output
   ↓
   gallery_montage_20240315_120000_v1_hitchcock.mp4
   + OTIO/EDL timeline files
```

---

## 🔍 Debug: Wo sehe ich was?

### Console/Logs:
```bash
# Creative Director Output
🎯 Creative Prompt: '...'
   ✅ Style Applied: hitchcock

# Every Cut Decision
┌─ Cut #5
│  💡 High-action scene matches energy peak (hitchcock style)
```

### Web UI:
```
Click "View Details & Logs" on any job →
   → Section "🎬 Creative Director Instructions"
   → Section "📋 Processing Logs" (live updates)
```

### JSON Export (optional):
```bash
# Set in monitoring.py
monitor.export_json("/data/output/decisions.json")

# Contains:
{
  "decisions": [
    {
      "type": "clip_selection",
      "choice": "VID_001.mp4 @ 2.5s",
      "reason": "highest_energy_match",
      "scores": {...}
    }
  ]
}
```

---

## 💡 Optimierungen

**Warum Keyword Matching zuerst?**
- 80% der Prompts sind einfach ("Edit like Hitchcock")
- LLM Call kostet Zeit (500-2000ms) und Geld
- Keyword Match ist instant (1ms) und kostenlos

**Warum JSON-Force Mode?**
- Gemini 2.0 unterstützt `responseMimeType: "application/json"`
- Garantiert valides JSON (kein Markdown wrapping)
- Keine manuelle Parsing-Hacks nötig

**Warum temperature=0.3?**
- Niedrige Temperature = konsistente Ausgabe
- Wir wollen keine Kreativität vom LLM, sondern zuverlässige Übersetzung
- User's kreative Intention ist bereits im Prompt kodiert

---

## 🚀 Erweiterungsmöglichkeiten

**Multi-Turn Conversation:**
```python
# Aktuell: Single-shot
User: "Edit like Hitchcock"
LLM: {JSON response}

# Möglich: Iterative refinement
User: "Edit like Hitchcock"
LLM: "I'll create suspenseful pacing. Any specific preferences?"
User: "More long takes in the intro"
LLM: {refined JSON with intro_duration_beats=32}
```

**Scene-Level Analysis:**
```python
# Aktuell: Style-level only
# Möglich: Per-scene decisions
for scene in scenes:
    decision = llm.analyze_scene(scene.image, context=previous_scenes)
    # "This scene has people arguing → cut faster"
```

**Real-time Feedback:**
```python
# Aktuell: One-time at startup
# Möglich: Live adjustments during render
if user_feedback == "too_fast":
    EDITING_INSTRUCTIONS['pacing']['speed'] = 'medium'
    re_render_from_cut(current_cut)
```
