# Phase 3 Performance Optimization - Research & Implementation Plan

## 🔬 Latest Research Findings (Januar 2026)

### 1. ProcessPoolExecutor für CPU-Bound Tasks
**Quelle:** Python 3.14 concurrent.futures documentation

**Key Insights:**
- **ProcessPoolExecutor** umgeht den GIL (Global Interpreter Lock)
- **InterpreterPoolExecutor** (Python 3.14+) nutzt Sub-Interpreter für echte Parallelisierung
- Ideal für CPU-intensive Aufgaben wie Audio/Video-Analyse
- **Best Practice:** `max_workers=os.process_cpu_count()` für CPU-Arbeit

**Implementierung:**
```python
with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    futures = [executor.submit(cpu_intensive_task, data) for data in dataset]
    results = [f.result() for f in as_completed(futures)]
```

### 2. Codebase Analyse - CPU-Intensive Bottlenecks

| Modul | Operation | Current | GIL-Bound | Parallelisierbar |
|-------|-----------|---------|-----------|------------------|
| `audio_analysis.py` | Beat detection (librosa) | ThreadPool | ✅ Ja | ✅ ProcessPool |
| `audio_analysis.py` | Energy profiling (NumPy) | Single-threaded | ✅ Ja | ✅ ProcessPool |
| `scene_analysis.py` | Scene detection (PySceneDetect) | ThreadPool | ✅ Ja | ✅ ProcessPool |
| `scene_analysis.py` | Histogram extraction (cv2) | ThreadPool | ✅ Ja | ✅ ProcessPool |
| `clip_selector.py` | Clip scoring | Single-threaded | ✅ Ja | ✅ Vectorized |

### 3. Aktuelle ThreadPool-Nutzung

**Probleme:**
- `ThreadPoolExecutor` wird für CPU-bound tasks verwendet
- Python GIL verhindert echte Parallelität
- Nur I/O-bound tasks profitieren von Threads

**Gefunden in:**
- `analysis_engine.py:313` - Scene detection (4 workers)
- `analysis_engine.py:361` - AI scene analysis (4 workers)
- `clip_enhancement.py:345` - Enhancement (multi workers)
- `montage_builder.py:287` - General executor (max_workers)

### 4. Binary Serialization (msgpack)

**Vorteile:**
- 40-60% schneller als JSON
- Kleinere Dateigröße (30-50% Reduktion)
- Native NumPy-Support

**Use Cases:**
- Cache-Dateien (scenes, beats, energy)
- Inter-process communication
- Metadata-Speicherung

### 5. Content-Addressable Caching

**Strategie:**
```python
cache_key = sha256(file_path + str(mtime) + str(size) + config_hash)
```

**Vorteile:**
- Eliminiert redundante Berechnungen
- Validiert Cache-Freshness
- Config-aware (verschiedene Parameter = verschiedene Keys)

---

## 📊 Phase 3 Optimierungen (Priorität)

### 🔥 High Priority (2-4x Speedup)

#### 1. ProcessPoolExecutor für Scene Detection
**Datei:** `src/montage_ai/core/analysis_engine.py`

**Problem:** ThreadPool für CPU-intensive PySceneDetect

**Lösung:**
```python
# ALT:
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(detect_video_scenes, v) for v in videos]

# NEU:
with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    futures = [executor.submit(detect_video_scenes, v) for v in videos]
```

**Expected Impact:** 2-4x schneller bei 4+ Cores

#### 2. ProcessPoolExecutor für Audio Analysis
**Datei:** `src/montage_ai/audio_analysis.py`

**Problem:** Beat detection und Energy profiling im GIL

**Lösung:** Parallel processing für mehrere Audio-Files

**Expected Impact:** 2-3x schneller bei Multi-Track-Projekten

#### 3. msgpack für Cache Serialization
**Dateien:** `src/montage_ai/core/analysis_cache.py`

**Problem:** JSON serialization langsam für große Datasets

**Lösung:**
```python
import msgpack

# Speichern
with open(cache_file, 'wb') as f:
    msgpack.pack(data, f)

# Laden
with open(cache_file, 'rb') as f:
    data = msgpack.unpack(f)
```

**Expected Impact:** 40-60% schneller Caching

### ⚡ Medium Priority (1.5-2x Speedup)

#### 4. Content-Addressable Cache
**Datei:** `src/montage_ai/core/analysis_cache.py`

**Lösung:**
```python
def _compute_cache_key(file_path: str, config: dict) -> str:
    stat = os.stat(file_path)
    config_str = json.dumps(config, sort_keys=True)
    data = f"{file_path}|{stat.st_mtime}|{stat.st_size}|{config_str}"
    return hashlib.sha256(data.encode()).hexdigest()
```

**Expected Impact:** Eliminiert falsche Cache-Hits

#### 5. Batch Processing für Histogram Extraction
**Datei:** `src/montage_ai/scene_analysis.py`

**Problem:** Frame-by-frame extraction ineffizient

**Lösung:** OpenCV batch frame reader

**Expected Impact:** 1.5-2x schneller

### 💡 Low Priority (Ergänzungen)

#### 6. Explicit GC Collection
**Überall wo große Objekte freigegeben werden**

```python
import gc
# Nach großen Operationen
del large_object
gc.collect()
```

#### 7. Memory-Mapped Files für große Cache-Files
**Für Histogram-Cache**

```python
import mmap
# Für sehr große Dateien
with open(file, 'r+b') as f:
    mmapped_file = mmap.mmap(f.fileno(), 0)
```

---

## 🎯 Implementation Strategy

### Phase 3.1: ProcessPoolExecutor (CPU Parallelization)
1. ✅ Scene detection → ProcessPool
2. ✅ Audio beat detection → ProcessPool
3. ✅ Energy profiling → ProcessPool

### Phase 3.2: Serialization Optimization
4. ✅ msgpack integration
5. ✅ Content-addressable caching

### Phase 3.3: Memory & Batch Optimization
6. ✅ Batch histogram extraction
7. ✅ Explicit GC management

---

## 📈 Expected Combined Impact

| Phase | Optimizations | Expected Speedup | Cumulative |
|-------|---------------|------------------|------------|
| Phase 1 | FFmpeg astats, LRU cache | 1.5-2x | 1.5-2x |
| Phase 2 | Keyframes, RAM disk, vectorization, K-D tree | 2-3x | 3-6x |
| **Phase 3** | **ProcessPool, msgpack, content cache** | **2-4x** | **6-24x** |

---

## 🔧 Technical Details

### ProcessPool vs ThreadPool

**ThreadPoolExecutor:**
- ✅ Low overhead
- ✅ Shared memory
- ❌ GIL-bound (kein echtes Parallelism für CPU-Tasks)

**ProcessPoolExecutor:**
- ✅ Echtes Parallelism (umgeht GIL)
- ✅ Nutzt alle CPU-Kerne
- ❌ Pickle-Overhead für Datenübertragung
- ❌ Mehr Memory (separate Prozesse)

**Faustregel:**
- CPU-bound → ProcessPoolExecutor
- I/O-bound → ThreadPoolExecutor
- Mixed → ProcessPool für CPU-Teil, ThreadPool für I/O

### Pickle-Kompatibilität

**Für ProcessPool müssen Funktionen picklable sein:**
- ✅ Top-level Funktionen
- ✅ Lambdas (in Python 3.8+)
- ✅ Klassen-Methoden (mit `__reduce__`)
- ❌ Verschachtelte Funktionen
- ❌ Lokale Closures mit komplexem State

**Lösung:** Extract zu Modul-Level Funktionen

---

## 🧪 Benchmark Targets

### Before Phase 3
```
Audio Analysis:    369ms (Phase 1 optimized)
Scene Detection:   ~2000ms (keyframes)
Clip Selection:    ~50ms (vectorized)
Cache Operations:  ~100ms (JSON)
```

### After Phase 3 (Target)
```
Audio Analysis:    120-180ms (2-3x via ProcessPool)
Scene Detection:   500-1000ms (2-4x via ProcessPool)
Clip Selection:    ~50ms (bereits optimiert)
Cache Operations:  40-60ms (2.5x via msgpack)
```

### Total Expected Improvement
**Baseline (Phase 0):** ~3000ms  
**Phase 1+2:** ~1000ms (3x)  
**Phase 3 Target:** ~300-500ms (6-10x vs Baseline)

---

## ✅ Implementation Checklist

- [ ] ProcessPool für scene detection
- [ ] ProcessPool für audio beat detection
- [ ] ProcessPool für energy profiling
- [ ] msgpack integration in analysis_cache
- [ ] Content-addressable cache keys
- [ ] Batch histogram extraction
- [ ] Explicit GC in critical sections
- [ ] Benchmark Phase 3 improvements
- [ ] Update documentation

---

**Research Date:** 7. Januar 2026  
**Python Version:** 3.12.3 (3.14 features noted for future)  
**Target:** < 500ms critical path latency
