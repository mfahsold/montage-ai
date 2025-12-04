# Algorithmen und Heuristiken

Technische Dokumentation der Analyse-Algorithmen in Montage AI.

---

## Musik-Analyse

### Beat-Detection

**Bibliothek:** librosa
**Algorithmus:** Onset Strength Envelope + Dynamic Programming

```python
y, sr = librosa.load(audio_path)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
```

**Funktionsweise:**
1. Audio-Signal wird in Spektrogramm transformiert (STFT)
2. Onset Strength Envelope berechnet Energie-Änderungen
3. Dynamic Programming findet rhythmische Struktur
4. Beat-Frames werden in Zeitpunkte konvertiert

**Parameter:**
- Sample Rate: 22050 Hz (default)
- Hop Length: 512 samples (~23ms bei 22050 Hz)
- Tempo-Range: 30-300 BPM

**Anwendung:**
- Synchronisation von Schnitten mit Musik-Beats
- Dynamische Tempoanpassung des Schnitts
- Beat-Synced Transitions

---

### Energy-Level-Analyse

**Bibliothek:** librosa
**Algorithmus:** Root Mean Square (RMS) Energy

```python
rms = librosa.feature.rms(y=y, hop_length=512)[0]
times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
rms_normalized = (rms - min(rms)) / (max(rms) - min(rms) + 1e-6)
```

**Funktionsweise:**
1. Signal wird in Frames unterteilt (Hop Length = 512)
2. RMS-Energie pro Frame berechnet: `sqrt(mean(frame^2))`
3. Normalisierung auf Bereich 0-1
4. Zeitachse aus Frame-Indizes generiert

**Anwendung:**
- Identifikation energetischer Momente für Action-Schnitte
- Pacing-Anpassung (schnelle Schnitte bei hoher Energie)
- Drop/Build-Up Detection in elektronischer Musik

**Heuristiken:**
- Energie > 0.7: "High Energy" → schnelle Schnitte (0.5-1s)
- Energie 0.3-0.7: "Medium" → normale Schnitte (1-2s)
- Energie < 0.3: "Low Energy" → lange Einstellungen (2-4s)

---

### Tempo-Extraktion

**Bibliothek:** librosa
**Methode:** Autocorrelation of Onset Strength

**Funktionsweise:**
1. Onset Strength Envelope berechnen
2. Autocorrelation über verschiedene Lag-Perioden
3. Peak-Detection für dominante Periode
4. Konvertierung zu BPM: `60 / period_in_seconds`

**Robustheit:**
- Multi-Tempo-Support (neuere librosa Versionen geben Array zurück)
- Fallback auf Median-Tempo bei Ambiguität

---

## Video-Analyse

### Szenen-Detection

**Bibliothek:** PySceneDetect
**Algorithmus:** Content-Aware Scene Detection

```python
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=30.0))
scene_manager.detect_scenes(video)
```

**Funktionsweise:**
1. Frame-by-Frame Differenz-Berechnung
2. HSV-Histogram-Differenz für Farbwechsel
3. Threshold-basierte Schnitt-Erkennung
4. Adaptive Threshold-Anpassung bei Varianz

**Parameter:**
- `threshold`: 30.0 (default) - niedrigere Werte = empfindlicher
- Vergleicht aufeinanderfolgende Frames
- Cut-Detection bei signifikanter visueller Änderung

**Optimierungen:**
- Hardware-Decoding über FFmpeg (wenn verfügbar)
- Downscaling auf 720p für Geschwindigkeit
- Multi-Threading für große Videos

**Anwendung:**
- Automatische Clip-Segmentierung
- Identifikation natürlicher Schnittpunkte
- Pre-Filtering für LLM-basierte Clip-Auswahl

---

### Motion Blur Detection

**Bibliothek:** OpenCV
**Algorithmus:** Laplacian Variance

```python
frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
is_blurry = laplacian_var < threshold  # threshold ~100
```

**Funktionsweise:**
1. Frame zu Grayscale konvertieren
2. Laplacian-Filter anwenden (2. Ableitung)
3. Varianz berechnet Schärfe-Niveau
4. Niedrige Varianz = Motion Blur / Out-of-Focus

**Threshold:**
- `< 100`: Starker Blur (reject)
- `100-500`: Moderate Schärfe (accept)
- `> 500`: Hohe Schärfe (bevorzugt)

**Anwendung:**
- Clip-Quality-Filtering
- Autofokus-Validation
- Action-Szenen-Priorisierung (Blur akzeptabel)

---

### Visual Similarity (Match Cuts)

**Bibliothek:** OpenCV
**Algorithmus:** Color Histogram Comparison

```python
hist1 = cv2.calcHist([frame1], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
hist2 = cv2.calcHist([frame2], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
```

**Funktionsweise:**
1. 3D-RGB-Histogram (8x8x8 bins = 512 bins total)
2. Histogramm-Normalisierung
3. Correlation-basierter Vergleich (Range: -1 bis 1)
4. High Correlation (>0.7) = visuell ähnlich

**Match-Cut-Kriterien:**
- Correlation > 0.7: Strong Match
- Farbpaletten-Übereinstimmung
- Kompositions-Ähnlichkeit (Center-Weighted)

**Anwendung:**
- Match Cuts (Hitchcock-Style)
- Visual Continuity
- Color-Matched Transitions

---

### Brightness Analysis

**Bibliothek:** OpenCV
**Algorithmus:** Mean Luminance Calculation

```python
frame = cv2.imread(frame_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mean_brightness = np.mean(gray)  # 0-255 scale
```

**Kategorisierung:**
- `< 50`: Very Dark
- `50-100`: Dark
- `100-150`: Normal
- `150-200`: Bright
- `> 200`: Very Bright / Overexposed

**Anwendung:**
- Exposure-Validation
- Tag/Nacht-Klassifizierung
- Color-Grading-Hints

---

## Heuristische Schnitt-Strategien

### Fibonacci-Pacing

**Konzept:** Schnittlängen folgen Fibonacci-Sequenz für natürlichen Rhythmus

```python
fib_sequence = [1, 1, 2, 3, 5, 8, 13]  # seconds
clip_durations = [fib_sequence[i % len(fib_sequence)] for i in range(num_clips)]
```

**Anwendung:**
- Organic Flow in Documentary-Style
- Prevents Repetitive Pacing
- Balanced zwischen Varianz und Struktur

**Stilrichtungen:**
- Documentary: Lange Sequenzen (3, 5, 8s)
- Minimalist: Fibonacci-Mittelwerte
- Slow-Cinema: Verdoppelte Fibonacci

---

### Energy-Adaptive Pacing

**Algorithmus:** Clip-Länge inversely proportional zu Musik-Energie

```python
if energy > 0.7:
    duration = 0.5 + random.uniform(0, 0.5)  # 0.5-1s
elif energy > 0.3:
    duration = 1.0 + random.uniform(0, 1.0)  # 1-2s
else:
    duration = 2.0 + random.uniform(0, 2.0)  # 2-4s
```

**Dynamische Anpassung:**
- Beat-Sync bei hoher Energie
- Längere Einstellungen bei ruhigen Passagen
- Smooth Transitions zwischen Energie-Levels

---

### Invisible Cuts (2024 Technique)

**Konzept:** Cuts verstecken durch visuelle Ablenkung

**Trigger:**
1. **Motion Blur:** Schnitt während Kamerabewegung
2. **Whip Pan:** Schnitt während schneller Kameraschwenk
3. **Occlusion:** Schnitt wenn Frame verdeckt ist (Person geht vorbei)
4. **Match Action:** Schnitt auf identischer Bewegung

**Detection:**
- Laplacian Variance < 150 → Motion Blur vorhanden
- Frame Difference > 50% → Schneller Schwenk
- Object Detection → Occlusion Candidate

---

## LLM-Integration

### Scene Content Analysis

**Model:** Gemini 2.0 Flash / Ollama Llava
**Input:** JPEG-Frame @ Midpoint von Szene
**Output:** JSON mit:
```json
{
  "quality": "YES/NO/MAYBE",
  "description": "Brief scene description",
  "action": "high/medium/low",
  "shot": "close/medium/wide",
  "composition": "good/acceptable/poor"
}
```

**Prompting-Strategie:**
- Structured JSON Output
- Binary Quality Gates
- Style-Aware Scoring

**Fallback:**
- Bei Timeout: Accept Clip (Fail-Open)
- Bei Parse-Error: Heuristic Scoring

---

## Performance-Optimierungen

### Metadata Caching

**System:** SQLite-basiertes Caching von Szenen-Analysen

**Cache-Key:** SHA256(video_path + file_mtime + analysis_version)

**Cached Data:**
- Scene Boundaries
- LLM Descriptions
- Quality Scores
- Beat Times
- Energy Curve

**Hit Rate:** ~90% bei wiederholten Runs

---

### Progressive Rendering

**Strategie:** Clips sofort nach Erstellung zu Disk schreiben

**Memory-Benefit:**
- Jeder Clip wird gerendert und freigegeben
- Vermeidet RAM-Akkumulation bei langen Montagen
- Ermöglicht 100+ Clip-Projekte auf 16GB RAM

**Concat-Methode:**
- FFmpeg concat demuxer (fastest)
- Fallback: MoviePy compose (compatibility)

---

## Literatur & Referenzen

- **Librosa:** McFee et al. "librosa: Audio and Music Signal Analysis in Python"
- **PySceneDetect:** Castellano "Content-Aware Scene Detection"
- **Match Cuts:** Hitchcock "Cutting on Action" Technique
- **Fibonacci Editing:** Dan Olson "The Fibonacci Sequence in Film"

---

*Dokumentation Stand: 2025-12-04*
*Montage AI Version: 0.4.0*
