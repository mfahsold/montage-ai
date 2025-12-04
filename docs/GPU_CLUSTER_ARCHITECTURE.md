# GPU-Offloading & Cluster-Distribution Architektur

> **Status:** Planungsphase (Dezember 2024)  
> **Ziel:** Rechenintensive Workloads auf GPU verlagern und nativ im K8s-Cluster verteilen

---

## Übersicht

Dieses Dokument beschreibt die Architektur für:
1. **GPU-Offloading** - Verlagerung von CPU-bound Prozessen auf lokale/Cloud GPUs
2. **Cluster-Distribution** - Native Verteilung von Workloads im Kubernetes-Cluster
3. **cgpu Integration** - Nutzung kostenloser Cloud-GPUs via cgpu

---

## 1. Workload-Analyse: CPU vs GPU

### Aktuelle CPU-intensive Operationen

| Operation            | Modul                        | CPU-Last  | GPU-geeignet            | Priorität |
| -------------------- | ---------------------------- | --------- | ----------------------- | --------- |
| **Beat Detection**   | `editor.py` (librosa)        | Hoch      | ✅ Ja (cupy/torch-audio) | P1        |
| **Scene Detection**  | `editor.py` (scenedetect)    | Hoch      | ✅ Ja (CUDA backend)     | P1        |
| **Video Encoding**   | `segment_writer.py` (FFmpeg) | Sehr hoch | ✅ Ja (NVENC/VAAPI)      | P0        |
| **AI Upscaling**     | `cgpu_upscaler.py`           | Sehr hoch | ✅ Bereits via cgpu      | ✓         |
| **Frame Extraction** | `ffmpeg_tools.py`            | Mittel    | ✅ Ja (CUVID decode)     | P2        |
| **Color Grading**    | `editor.py` (cv2/LUTs)       | Mittel    | ✅ Ja (OpenCV CUDA)      | P2        |
| **Image Similarity** | `clip_selector.py`           | Mittel    | ✅ Ja (torch)            | P2        |
| **Stabilization**    | `editor.py` (vidstab)        | Hoch      | ⚠️ Begrenzt              | P3        |

### Workload-Kategorien für Verteilung

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Workload Distribution Matrix                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐ │
│  │   LOKAL (CPU)    │     │  LOKAL (GPU)     │     │   CLOUD (cgpu)   │ │
│  ├──────────────────┤     ├──────────────────┤     ├──────────────────┤ │
│  │ • File I/O       │     │ • Video Decode   │     │ • AI Upscaling   │ │
│  │ • Timeline Logic │     │ • Video Encode   │     │ • LLM Inference  │ │
│  │ • Clip Selection │     │ • Scene Detect   │     │ • Batch Transcode│ │
│  │ • Story Arc      │     │ • Beat Detect    │     │ • Training Jobs  │ │
│  │ • Metadata       │     │ • Color Grade    │     │                  │ │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘ │
│           │                       │                        │            │
│           └───────────────────────┼────────────────────────┘            │
│                                   │                                      │
│                      ┌────────────▼────────────┐                        │
│                      │   CLUSTER (K8s Jobs)    │                        │
│                      ├─────────────────────────┤                        │
│                      │ • Parallel Clip Process │                        │
│                      │ • Multi-Video Batches   │                        │
│                      │ • Distributed Encoding  │                        │
│                      └─────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. cgpu-Erweiterung: Neue Workloads

### 2.1 Beat Detection auf Cloud GPU

**Aktuell:** librosa auf CPU (langsam bei langen Tracks)

**Neu:** Torch-Audio auf cgpu T4/A100

```python
# Geplante Implementierung: cgpu_audio.py
class CGPUBeatDetector:
    """Beat detection via cgpu cloud GPU."""
    
    def detect_beats_remote(self, audio_path: str) -> Dict[str, Any]:
        """
        Upload audio → Run torchaudio on Colab → Download results
        
        Returns: {
            'beats': [0.5, 1.0, 1.5, ...],  # Beat timestamps
            'tempo': 120.0,                   # BPM
            'energy': [...],                  # Energy curve
            'onset_strength': [...]           # For beat alignment
        }
        """
        script = '''
import torchaudio
import torch

# Load audio
waveform, sr = torchaudio.load("/content/audio.wav")

# Beat detection with torchaudio
# ... GPU-accelerated processing ...
'''
        return run_cgpu_python(script)
```

**Vorteile:**
- 5-10x schneller für Tracks >3 Minuten
- Konsistente Performance unabhängig vom lokalen System
- Moderne Torch-Audio Features verfügbar

### 2.2 Scene Detection auf Cloud GPU

**Aktuell:** scenedetect mit OpenCV auf CPU

**Neu:** GPU-beschleunigte Scene Detection via cgpu

```python
# Geplante Implementierung: cgpu_scene_detect.py
class CGPUSceneDetector:
    """Scene detection via cgpu with CUDA acceleration."""
    
    def detect_scenes_remote(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Upload video → Run scene detection with CUDA → Download scene list
        
        Uses:
        - OpenCV CUDA backend for frame differencing
        - Optional: PySceneDetect with GPU acceleration
        - Optional: TransNetV2 neural scene detector
        """
        script = '''
import cv2
cv2.cuda.setDevice(0)  # Use GPU

# TransNetV2 for neural scene detection
from transnetv2 import TransNetV2
model = TransNetV2()

# Process video on GPU
scenes = model.predict_video("/content/video.mp4")
'''
        return run_cgpu_python(script)
```

**Vorteile:**
- TransNetV2 liefert präzisere Schnitte als Content-Detector
- Batch-Processing mehrerer Videos parallel
- Unabhängig von lokaler GPU-Verfügbarkeit

### 2.3 Video Encoding auf Cloud GPU

**Aktuell:** FFmpeg libx264 auf CPU (bottleneck!)

**Neu:** NVENC Encoding auf cgpu für Batch-Jobs

```python
# Geplante Implementierung: cgpu_encoder.py
class CGPUEncoder:
    """Hardware-accelerated encoding via cgpu."""
    
    def encode_remote(
        self, 
        input_path: str,
        output_path: str,
        codec: str = "h264_nvenc"
    ) -> str:
        """
        Upload raw/intermediate → NVENC encode on Colab → Download encoded
        
        Besonders nützlich für:
        - Finale 4K Renders
        - Batch-Transcoding vieler Clips
        - HEVC/AV1 Encoding (CPU-intensiv)
        """
        cmd = f'''
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
  -i /content/input.mp4 \
  -c:v h264_nvenc -preset p7 -cq 20 \
  -c:a copy \
  /content/output.mp4
'''
        return run_cgpu_command(cmd)
```

---

## 3. Cluster-Distribution Architektur

### 3.1 Job-basierte Verteilung

```yaml
# Neue K8s Job-Architektur
apiVersion: batch/v1
kind: Job
metadata:
  name: montage-ai-worker-{{ .task_id }}
spec:
  parallelism: 4  # 4 parallele Worker
  completions: 4
  template:
    spec:
      containers:
      - name: worker
        image: montage-ai:latest
        env:
        - name: WORKER_MODE
          value: "distributed"
        - name: TASK_TYPE
          value: "{{ .task_type }}"  # encode|analyze|upscale
        - name: TASK_ID
          value: "{{ .task_id }}"
        resources:
          requests:
            nvidia.com/gpu: 1
```

### 3.2 Task Queue Architektur

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Distributed Processing Flow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐                                                       │
│  │ Web UI /     │                                                       │
│  │ CLI Request  │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐     │
│  │ Coordinator  │────▶│            Redis Task Queue              │     │
│  │ (Orchestrator)│     │  ┌────────┐ ┌────────┐ ┌────────┐       │     │
│  └──────────────┘     │  │analyze │ │encode  │ │upscale │       │     │
│                       │  │ queue  │ │ queue  │ │ queue  │       │     │
│                       │  └───┬────┘ └───┬────┘ └───┬────┘       │     │
│                       └──────┼──────────┼──────────┼─────────────┘     │
│                              │          │          │                    │
│         ┌────────────────────┼──────────┼──────────┼────────────────┐  │
│         │                    │          │          │                │  │
│         ▼                    ▼          ▼          ▼                │  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │  │
│  │ Worker Pod 1 │     │ Worker Pod 2 │     │ Worker Pod 3 │        │  │
│  │ (CPU Node)   │     │ (GPU Node)   │     │ (cgpu Cloud) │        │  │
│  │              │     │              │     │              │        │  │
│  │ • Metadata   │     │ • Encoding   │     │ • Upscaling  │        │  │
│  │ • Timeline   │     │ • Scene Det. │     │ • LLM        │        │  │
│  │ • Selection  │     │ • Beat Det.  │     │ • Heavy AI   │        │  │
│  └──────────────┘     └──────────────┘     └──────────────┘        │  │
│         │                    │                    │                 │  │
│         └────────────────────┼────────────────────┘                 │  │
│                              │                                      │  │
│                              ▼                                      │  │
│                       ┌──────────────┐                              │  │
│                       │ Shared PVC   │                              │  │
│                       │ (NFS/Ceph)   │                              │  │
│                       │ /data/input  │                              │  │
│                       │ /data/output │                              │  │
│                       └──────────────┘                              │  │
│                                                                      │  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Neue Module für Cluster-Betrieb

```
src/montage_ai/
├── distributed/                    # NEU: Cluster-Module
│   ├── __init__.py
│   ├── coordinator.py             # Job-Orchestrierung
│   ├── worker.py                  # Worker-Loop für K8s Pods
│   ├── task_queue.py              # Redis-basierte Task Queue
│   └── shared_storage.py          # PVC/NFS Abstraction
│
├── cgpu_audio.py                  # NEU: Beat Detection via cgpu
├── cgpu_scene_detect.py           # NEU: Scene Detection via cgpu
├── cgpu_encoder.py                # NEU: Video Encoding via cgpu
└── ...
```

---

## 4. Implementierungsplan

### Phase 1: cgpu Workload-Erweiterung (2-3 Wochen)

| Task | Beschreibung                 | Dateien                                 |
| ---- | ---------------------------- | --------------------------------------- |
| 1.1  | cgpu Beat Detection Modul    | `cgpu_audio.py`                         |
| 1.2  | cgpu Scene Detection Modul   | `cgpu_scene_detect.py`                  |
| 1.3  | cgpu Encoder Modul           | `cgpu_encoder.py`                       |
| 1.4  | Feature Flags in `editor.py` | `CGPU_BEAT_DETECT`, `CGPU_SCENE_DETECT` |
| 1.5  | Fallback-Logik lokal ↔ cgpu  | `cgpu_utils.py`                         |

### Phase 2: Lokale GPU-Beschleunigung (1-2 Wochen)

| Task | Beschreibung                   | Dateien                 |
| ---- | ------------------------------ | ----------------------- |
| 2.1  | OpenCV CUDA Backend aktivieren | `editor.py`, Dockerfile |
| 2.2  | NVENC/VAAPI für lokale GPUs    | `ffmpeg_config.py`      |
| 2.3  | GPU Memory Management          | `memory_monitor.py`     |

### Phase 3: Cluster-Distribution (3-4 Wochen)

| Task | Beschreibung               | Dateien                         |
| ---- | -------------------------- | ------------------------------- |
| 3.1  | Redis Task Queue Setup     | `distributed/task_queue.py`     |
| 3.2  | Worker Pod Implementation  | `distributed/worker.py`         |
| 3.3  | Coordinator Service        | `distributed/coordinator.py`    |
| 3.4  | K8s Manifests für Workers  | `deploy/k3s/workers/`           |
| 3.5  | Shared Storage Integration | `distributed/shared_storage.py` |

### Phase 4: Integration & Testing (2 Wochen)

| Task | Beschreibung           | Dateien                          |
| ---- | ---------------------- | -------------------------------- |
| 4.1  | End-to-End Tests       | `tests/test_distributed.py`      |
| 4.2  | Performance Benchmarks | `docs/benchmarks.md`             |
| 4.3  | Dokumentation          | `docs/DISTRIBUTED_PROCESSING.md` |

---

## 5. Konfiguration

### Neue Umgebungsvariablen

```bash
# cgpu Workload-Erweiterung
CGPU_BEAT_DETECT=true          # Beat detection auf Cloud GPU
CGPU_SCENE_DETECT=true         # Scene detection auf Cloud GPU
CGPU_ENCODE=false              # Encoding auf Cloud GPU (für große Jobs)
CGPU_ENCODE_THRESHOLD=300      # Nur Videos >5min auf Cloud GPU encodieren

# Lokale GPU-Beschleunigung
LOCAL_GPU_ENCODE=auto          # NVENC/VAAPI wenn verfügbar
LOCAL_GPU_OPENCV=auto          # OpenCV CUDA wenn verfügbar
GPU_MEMORY_LIMIT=4096          # Max GPU Memory in MB

# Cluster-Distribution
DISTRIBUTED_MODE=false         # Aktiviert Cluster-Betrieb
REDIS_URL=redis://redis:6379   # Task Queue
WORKER_CONCURRENCY=4           # Tasks pro Worker
COORDINATOR_URL=http://coordinator:8080
```

### docker-compose.distributed.yml (Konzept)

```yaml
version: '3.8'

services:
  coordinator:
    image: montage-ai:latest
    command: python -m montage_ai.distributed.coordinator
    environment:
      - DISTRIBUTED_MODE=true
      - REDIS_URL=redis://redis:6379
    ports:
      - "8080:8080"
    depends_on:
      - redis

  worker-cpu:
    image: montage-ai:latest
    command: python -m montage_ai.distributed.worker
    environment:
      - WORKER_TYPE=cpu
      - REDIS_URL=redis://redis:6379
    deploy:
      replicas: 2

  worker-gpu:
    image: montage-ai:latest
    command: python -m montage_ai.distributed.worker
    environment:
      - WORKER_TYPE=gpu
      - REDIS_URL=redis://redis:6379
    deploy:
      replicas: 1
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  worker-cgpu:
    image: montage-ai:latest
    command: python -m montage_ai.distributed.worker
    environment:
      - WORKER_TYPE=cgpu
      - CGPU_GPU_ENABLED=true
      - REDIS_URL=redis://redis:6379
    deploy:
      replicas: 1

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

---

## 6. Performance-Erwartungen

### Geschätzte Speedups durch GPU-Offloading

| Operation                        | CPU (aktuell) | GPU (lokal)       | cgpu (Cloud)  |
| -------------------------------- | ------------- | ----------------- | ------------- |
| Beat Detection (3min Track)      | 15s           | 3s                | 5s + Upload   |
| Scene Detection (10min Video)    | 45s           | 8s                | 12s + Upload  |
| Encoding 1080p 1min              | 60s           | 5s (NVENC)        | 8s + Transfer |
| AI Upscaling 2x (1min)           | 20min (CPU)   | 2min (lokale GPU) | 3min (T4)     |
| **Gesamt-Pipeline (5min Video)** | **~8min**     | **~2min**         | **~3min**     |

### Break-Even für cgpu vs Lokal

- **cgpu lohnt sich bei:**
  - Keine lokale GPU verfügbar
  - Batch-Jobs mit vielen Videos
  - AI-Workloads (Upscaling, LLM)
  - 4K/8K Content

- **Lokal ist besser bei:**
  - Schnelle lokale GPU vorhanden
  - Kleine Videos (<1min)
  - Niedrige Latenz wichtiger als Throughput
  - Offline-Betrieb nötig

---

## 7. Nächste Schritte

1. **Sofort:** `cgpu_audio.py` für Beat Detection implementieren
2. **Woche 1-2:** Scene Detection und Encoder Module
3. **Woche 3-4:** Lokale GPU-Optimierungen
4. **Woche 5-8:** Cluster-Distribution mit Redis Queue

---

## Referenzen

- [cgpu Repository](https://github.com/RohanAdwankar/cgpu)
- [TransNetV2 Scene Detection](https://github.com/soCzech/TransNetV2)
- [torchaudio Beat Tracking](https://pytorch.org/audio/stable/tutorials/beat_detection_tutorial.html)
- [FFmpeg NVENC Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/nvenc-video-encoder-api-prog-guide/)
