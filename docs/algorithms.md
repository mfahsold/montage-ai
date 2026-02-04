# Algorithms and Heuristics

Technical documentation of analysis algorithms in Montage AI.

---

## Music Analysis

> Note: As of 2026-01, Montage AI uses an FFmpeg-first beat/tempo path (`astats`/tempo) for portability and startup speed. Librosa remains optional.

### Beat Detection (Librosa - optional)

**Library:** librosa
**Algorithm:** Onset Strength Envelope + Dynamic Programming

```python
y, sr = librosa.load(audio_path)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
```

**How it works:**
1. Audio signal is transformed into spectrogram (STFT)
2. Onset Strength Envelope computes energy changes
3. Dynamic Programming finds rhythmic structure
4. Beat frames are converted to time points

**Parameters:**
- Sample Rate: 22050 Hz (default)
- Hop Length: 512 samples (~23ms at 22050 Hz)
- Tempo Range: 30-300 BPM

**Application:**
- Synchronize cuts with music beats
- Dynamic tempo adaptation of cuts
- Beat-synced transitions

---

### Energy Level Analysis

**Library:** librosa
**Algorithm:** Root Mean Square (RMS) Energy

```python
rms = librosa.feature.rms(y=y, hop_length=512)[0]
times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
rms_normalized = (rms - min(rms)) / (max(rms) - min(rms) + 1e-6)
```

**How it works:**
1. Signal is divided into frames (Hop Length = 512)
2. RMS energy per frame calculated: `sqrt(mean(frame^2))`
3. Normalization to range 0-1
4. Time axis generated from frame indices

**Application:**
- Identify energetic moments for action cuts
- Pacing adaptation (fast cuts at high energy)
- Drop/build-up detection in electronic music

**Heuristics:**
- Energy > 0.7: "High Energy" → fast cuts (0.5-1s)
- Energy 0.3-0.7: "Medium" → normal cuts (1-2s)
- Energy < 0.3: "Low Energy" → long shots (2-4s)

---

### Tempo Extraction

**Library:** librosa
**Method:** Autocorrelation of Onset Strength

**How it works:**
1. Calculate Onset Strength Envelope
2. Autocorrelation over different lag periods
3. Peak detection for dominant period
4. Convert to BPM: `60 / period_in_seconds`

**Robustness:**
- Multi-tempo support (newer librosa versions return array)
- Fallback to median tempo on ambiguity

---

## Video Analysis

### Scene Detection

**Library:** PySceneDetect
**Algorithm:** Content-Aware Scene Detection

```python
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=30.0))
scene_manager.detect_scenes(video)
```

**How it works:**
1. Frame-by-frame difference calculation
2. HSV histogram difference for color changes
3. Threshold-based cut detection
4. Adaptive threshold adjustment on variance

**Parameters:**
- `threshold`: 30.0 (default) - lower values = more sensitive
- Compares consecutive frames
- Cut detection on significant visual change

**Optimizations:**
- Hardware decoding via FFmpeg (when available)
- Downscaling to 720p for speed
- Multi-threading for large videos

**Application:**
- Automatic clip segmentation
- Identify natural cut points
- Pre-filtering for LLM-based clip selection

---

### Motion Blur Detection

**Library:** OpenCV
**Algorithm:** Laplacian Variance

```python
frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
is_blurry = laplacian_var < threshold  # threshold ~1000 (configurable)
```

**How it works:**
1. Convert frame to grayscale
2. Apply Laplacian filter (2nd derivative)
3. Variance measures sharpness level
4. Low variance = motion blur / out-of-focus

**Threshold (Laplacian variance):**
- Default is controlled by `BLUR_DETECTION_VARIANCE_THRESHOLD` (default: 1000.0)
- Typical range: blurry ~100, sharp ~1000
- `< 100`: Strong blur (reject)
- `100-500`: Moderate sharpness (accept)
- `> 500`: High sharpness (preferred)

**Application:**
- Clip quality filtering
- Autofocus validation
- Action scene prioritization (blur acceptable)

---

### Visual Similarity (Match Cuts)

**Library:** OpenCV
**Algorithm:** Color Histogram Comparison

```python
hist1 = cv2.calcHist([frame1], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
hist2 = cv2.calcHist([frame2], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
```

**How it works:**
1. 3D RGB histogram (8x8x8 bins = 512 bins total)
2. Histogram normalization
3. Correlation-based comparison (Range: -1 to 1)
4. High correlation (>0.7) = visually similar

**Match Cut Criteria:**
- Correlation > 0.7: Strong match
- Color palette agreement
- Composition similarity (center-weighted)

**Application:**
- Match cuts (Hitchcock style)
- Visual continuity
- Color-matched transitions

---

### Brightness Analysis

**Library:** OpenCV
**Algorithm:** Mean Luminance Calculation

```python
frame = cv2.imread(frame_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mean_brightness = np.mean(gray)  # 0-255 scale
```

**Categorization:**
- `< 50`: Very Dark
- `50-100`: Dark
- `100-150`: Normal
- `150-200`: Bright
- `> 200`: Very Bright / Overexposed

**Application:**
- Exposure validation
- Day/night classification
- Color grading hints

---

## Heuristic Cutting Strategies

### Fibonacci Pacing

**Concept:** Cut lengths follow Fibonacci sequence for natural rhythm

```python
fib_sequence = [1, 1, 2, 3, 5, 8, 13]  # seconds
clip_durations = [fib_sequence[i % len(fib_sequence)] for i in range(num_clips)]
```

**Application:**
- Organic flow in documentary style
- Prevents repetitive pacing
- Balance between variance and structure

**Style Variants:**
- Documentary: Long sequences (3, 5, 8s)
- Minimalist: Fibonacci averages
- Slow Cinema: Doubled Fibonacci

---

### Energy-Adaptive Pacing

**Algorithm:** Clip length inversely proportional to music energy

```python
if energy > 0.7:
    duration = 0.5 + random.uniform(0, 0.5)  # 0.5-1s
elif energy > 0.3:
    duration = 1.0 + random.uniform(0, 1.0)  # 1-2s
else:
    duration = 2.0 + random.uniform(0, 2.0)  # 2-4s
```

**Dynamic Adaptation:**
- Beat sync at high energy
- Longer shots during quiet passages
- Smooth transitions between energy levels

---

### Invisible Cuts (2024 Technique)

**Concept:** Hide cuts through visual distraction

**Triggers:**
1. **Motion Blur:** Cut during camera movement
2. **Whip Pan:** Cut during fast camera pan
3. **Occlusion:** Cut when frame is obscured (person passes by)
4. **Match Action:** Cut on identical movement

**Detection:**
- Laplacian Variance < 150 → Motion blur present
- Frame Difference > 50% → Fast pan
- Object Detection → Occlusion candidate

---

## LLM Integration

### Scene Content Analysis

**Model:** Gemini 2.0 Flash / moondream2 / Ollama Llava
**Input:** JPEG frame @ midpoint of scene
**Output:** JSON with:

```json
{
  "quality": "YES/NO/MAYBE",
  "description": "Brief scene description",
  "action": "high/medium/low",
  "shot": "close/medium/wide",
  "composition": "good/acceptable/poor"
}
```

**Prompting Strategy:**
- Structured JSON output
- Binary quality gates
- Style-aware scoring

**Fallback:**
- On timeout: Accept clip (fail-open)
- On parse error: Heuristic scoring

**Backend Priority:**
- OpenAI-compatible (LiteLLM / llama-box tiered models) - preferred
- Google AI (Gemini)
- Ollama (local fallback)

---

## Performance Optimizations

### Metadata Caching

**System:** SQLite-based caching of scene analyses

**Cache Key:** SHA256(video_path + file_mtime + analysis_version)

**Cached Data:**
- Scene boundaries
- LLM descriptions
- Quality scores
- Beat times
- Energy curve

**Hit Rate:** ~90% on repeated runs

---

### Progressive Rendering

**Strategy:** Write clips to disk immediately after creation

**Memory Benefit:**
- Each clip is rendered and freed
- Avoids RAM accumulation in long montages
- Enables 100+ clip projects on 16GB RAM

**Concat Method:**
- FFmpeg concat demuxer (fastest)
- Fallback: MoviePy compose (compatibility)

---

## References

- **Librosa:** McFee et al. "librosa: Audio and Music Signal Analysis in Python"
- **PySceneDetect:** Castellano "Content-Aware Scene Detection"
- **Match Cuts:** Hitchcock "Cutting on Action" Technique
- **Fibonacci Editing:** Dan Olson "The Fibonacci Sequence in Film"

---

*Last Updated: 2025-12-04*
*Montage AI Version: 0.4.0*
