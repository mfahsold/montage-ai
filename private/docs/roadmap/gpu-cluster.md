# GPU Offloading & Cluster Distribution Architecture

> **Status:** Planning phase (December 2024)
> **Goal:** Offload compute-intensive workloads to GPU and natively distribute across K8s cluster

---

## Overview

This document describes the architecture for:
1. **GPU Offloading** - Moving CPU-bound processes to local/cloud GPUs
2. **Cluster Distribution** - Native workload distribution in Kubernetes cluster
3. **cgpu Integration** - Using free cloud GPUs via cgpu

---

## 1. Workload Analysis: CPU vs GPU

### Current CPU-Intensive Operations

| Operation            | Module                        | CPU Load  | GPU-Suitable            | Status              |
| -------------------- | ----------------------------- | --------- | ----------------------- | ------------------- |
| **Video Encoding**   | `segment_writer.py` (FFmpeg)  | Very High | NVENC/VAAPI/QSV         | **Implemented**     |
| **AI Upscaling**     | `cgpu_upscaler.py`            | Very High | CUDA via cgpu           | **Implemented**     |
| **Beat Detection**   | `editor.py` (librosa)         | High      | Yes (cupy/torch-audio)  | Planned             |
| **Scene Detection**  | `editor.py` (scenedetect)     | High      | Yes (CUDA backend)      | Planned             |
| **Frame Extraction** | `ffmpeg_tools.py`             | Medium    | Yes (CUVID decode)      | Planned             |
| **Color Grading**    | `editor.py` (cv2/LUTs)        | Medium    | Yes (OpenCV CUDA)       | Planned             |
| **Image Similarity** | `clip_selector.py`            | Medium    | Yes (torch)             | Planned             |
| **Stabilization**    | `editor.py` (vidstab)         | High      | Limited                 | CPU only            |

### GPU Encoder Support

```
FFMPEG_HWACCEL=auto  # Auto-detection
FFMPEG_HWACCEL=nvenc # NVIDIA (h264_nvenc, hevc_nvenc)
FFMPEG_HWACCEL=vaapi # AMD/Intel Linux (h264_vaapi, hevc_vaapi)
FFMPEG_HWACCEL=qsv   # Intel QuickSync (h264_qsv, hevc_qsv)
FFMPEG_HWACCEL=none  # CPU (libx264, libx265)
```

### Workload Categories for Distribution

```
+-------------------------------------------------------------------------+
|                    Workload Distribution Matrix                          |
+-------------------------------------------------------------------------+
|                                                                          |
|  +------------------+     +------------------+     +------------------+  |
|  |   LOCAL (CPU)    |     |   LOCAL (GPU)    |     |   CLOUD (cgpu)   |  |
|  +------------------+     +------------------+     +------------------+  |
|  | - File I/O       |     | - Video Decode   |     | - AI Upscaling   |  |
|  | - Timeline Logic |     | - Video Encode   |     | - LLM Inference  |  |
|  | - Clip Selection |     | - Scene Detect   |     | - Batch Transcode|  |
|  | - Story Arc      |     | - Beat Detect    |     | - Training Jobs  |  |
|  | - Metadata       |     | - Color Grade    |     |                  |  |
|  +------------------+     +------------------+     +------------------+  |
|           |                       |                        |             |
|           +-----------------------+------------------------+             |
|                                   |                                      |
|                      +------------v------------+                         |
|                      |   CLUSTER (K8s Jobs)    |                         |
|                      +-------------------------+                         |
|                      | - Parallel Clip Process |                         |
|                      | - Multi-Video Batches   |                         |
|                      | - Distributed Encoding  |                         |
|                      +-------------------------+                         |
+-------------------------------------------------------------------------+
```

---

## 2. cgpu Extension: New Workloads

### 2.1 Beat Detection on Cloud GPU

**Current:** librosa on CPU (slow for long tracks)

**New:** Torch-Audio on cgpu T4/A100

```python
# Planned implementation: cgpu_audio.py
class CGPUBeatDetector:
    """Beat detection via cgpu cloud GPU."""

    def detect_beats_remote(self, audio_path: str) -> Dict[str, Any]:
        """
        Upload audio -> Run torchaudio on Colab -> Download results

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

**Benefits:**
- 5-10x faster for tracks >3 minutes
- Consistent performance regardless of local system
- Modern Torch-Audio features available

### 2.2 Scene Detection on Cloud GPU

**Current:** scenedetect with OpenCV on CPU

**New:** GPU-accelerated scene detection via cgpu

```python
# Planned implementation: cgpu_scene_detect.py
class CGPUSceneDetector:
    """Scene detection via cgpu with CUDA acceleration."""

    def detect_scenes_remote(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Upload video -> Run scene detection with CUDA -> Download scene list

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

**Benefits:**
- TransNetV2 provides more precise cuts than content detector
- Batch processing of multiple videos in parallel
- Independent of local GPU availability

### 2.3 Video Encoding on Cloud GPU

**Current:** FFmpeg libx264 on CPU (bottleneck!)

**New:** NVENC encoding on cgpu for batch jobs

```python
# Planned implementation: cgpu_encoder.py
class CGPUEncoder:
    """Hardware-accelerated encoding via cgpu."""

    def encode_remote(
        self,
        input_path: str,
        output_path: str,
        codec: str = "h264_nvenc"
    ) -> str:
        """
        Upload raw/intermediate -> NVENC encode on Colab -> Download encoded

        Particularly useful for:
        - Final 4K renders
        - Batch transcoding of many clips
        - HEVC/AV1 encoding (CPU-intensive)
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

## 3. Cluster Distribution Architecture

### 3.1 Job-Based Distribution

```yaml
# New K8s job architecture
apiVersion: batch/v1
kind: Job
metadata:
  name: montage-ai-worker-{{ .task_id }}
spec:
  parallelism: 4  # 4 parallel workers
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

### 3.2 Task Queue Architecture

```
+-------------------------------------------------------------------------+
|                     Distributed Processing Flow                          |
+-------------------------------------------------------------------------+
|                                                                          |
|  +--------------+                                                        |
|  | Web UI /     |                                                        |
|  | CLI Request  |                                                        |
|  +------+-------+                                                        |
|         |                                                                |
|         v                                                                |
|  +--------------+     +------------------------------------------+       |
|  | Coordinator  |---->|            Redis Task Queue              |       |
|  | (Orchestrator)|    |  +--------+ +--------+ +--------+       |       |
|  +--------------+     |  |analyze | |encode  | |upscale |       |       |
|                       |  | queue  | | queue  | | queue  |       |       |
|                       |  +---+----+ +---+----+ +---+----+       |       |
|                       +------+----------+----------+-------------+       |
|                              |          |          |                     |
|         +--------------------+----------+----------+----------------+    |
|         |                    |          |          |                |    |
|         v                    v          v          v                |    |
|  +--------------+     +--------------+     +--------------+        |    |
|  | Worker Pod 1 |     | Worker Pod 2 |     | Worker Pod 3 |        |    |
|  | (CPU Node)   |     | (GPU Node)   |     | (cgpu Cloud) |        |    |
|  |              |     |              |     |              |        |    |
|  | - Metadata   |     | - Encoding   |     | - Upscaling  |        |    |
|  | - Timeline   |     | - Scene Det. |     | - LLM        |        |    |
|  | - Selection  |     | - Beat Det.  |     | - Heavy AI   |        |    |
|  +--------------+     +--------------+     +--------------+        |    |
|         |                    |                    |                 |    |
|         +--------------------+--------------------+                 |    |
|                              |                                      |    |
|                              v                                      |    |
|                       +--------------+                              |    |
|                       | Shared PVC   |                              |    |
|                       | (NFS/Ceph)   |                              |    |
|                       | /data/input  |                              |    |
|                       | /data/output |                              |    |
|                       +--------------+                              |    |
|                                                                      |    |
+-------------------------------------------------------------------------+
```

### 3.3 New Modules for Cluster Operation

```
src/montage_ai/
├── distributed/                    # NEW: Cluster modules
│   ├── __init__.py
│   ├── coordinator.py             # Job orchestration
│   ├── worker.py                  # Worker loop for K8s pods
│   ├── task_queue.py              # Redis-based task queue
│   └── shared_storage.py          # PVC/NFS abstraction
│
├── cgpu_audio.py                  # NEW: Beat detection via cgpu
├── cgpu_scene_detect.py           # NEW: Scene detection via cgpu
├── cgpu_encoder.py                # NEW: Video encoding via cgpu
└── ...
```

---

## 4. Implementation Plan

### Phase 1: cgpu Workload Extension (2-3 weeks)

| Task | Description                  | Files                                   |
| ---- | ---------------------------- | --------------------------------------- |
| 1.1  | cgpu beat detection module   | `cgpu_audio.py`                         |
| 1.2  | cgpu scene detection module  | `cgpu_scene_detect.py`                  |
| 1.3  | cgpu encoder module          | `cgpu_encoder.py`                       |
| 1.4  | Feature flags in `editor.py` | `CGPU_BEAT_DETECT`, `CGPU_SCENE_DETECT` |
| 1.5  | Fallback logic local <-> cgpu| `cgpu_utils.py`                         |

### Phase 2: Local GPU Acceleration (1-2 weeks)

| Task | Description                    | Files                 |
| ---- | ------------------------------ | --------------------- |
| 2.1  | Enable OpenCV CUDA backend     | `editor.py`, Dockerfile |
| 2.2  | NVENC/VAAPI for local GPUs     | `ffmpeg_config.py`    |
| 2.3  | GPU memory management          | `memory_monitor.py`   |

### Phase 3: Cluster Distribution (3-4 weeks)

| Task | Description                | Files                           |
| ---- | -------------------------- | ------------------------------- |
| 3.1  | Redis task queue setup     | `distributed/task_queue.py`     |
| 3.2  | Worker pod implementation  | `distributed/worker.py`         |
| 3.3  | Coordinator service        | `distributed/coordinator.py`    |
| 3.4  | K8s manifests for workers  | `deploy/k3s/workers/`           |
| 3.5  | Shared storage integration | `distributed/shared_storage.py` |

### Phase 4: Integration & Testing (2 weeks)

| Task | Description            | Files                            |
| ---- | ---------------------- | -------------------------------- |
| 4.1  | End-to-end tests       | `tests/test_distributed.py`      |
| 4.2  | Performance benchmarks | `docs/benchmarks.md`             |
| 4.3  | Documentation          | `docs/DISTRIBUTED_PROCESSING.md` |

---

## 5. Configuration

### New Environment Variables

```bash
# cgpu workload extension
CGPU_BEAT_DETECT=true          # Beat detection on cloud GPU
CGPU_SCENE_DETECT=true         # Scene detection on cloud GPU
CGPU_ENCODE=false              # Encoding on cloud GPU (for large jobs)
CGPU_ENCODE_THRESHOLD=300      # Only encode videos >5min on cloud GPU

# Local GPU acceleration
LOCAL_GPU_ENCODE=auto          # NVENC/VAAPI if available
LOCAL_GPU_OPENCV=auto          # OpenCV CUDA if available
GPU_MEMORY_LIMIT=4096          # Max GPU memory in MB

# Cluster distribution
DISTRIBUTED_MODE=false         # Enables cluster operation
REDIS_URL=redis://redis:6379   # Task queue
WORKER_CONCURRENCY=4           # Tasks per worker
COORDINATOR_URL=http://coordinator:8080
```

### docker-compose.distributed.yml (Concept)

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

## 6. Performance Expectations

### Estimated Speedups from GPU Offloading

| Operation                        | CPU (current) | GPU (local)       | cgpu (Cloud)  |
| -------------------------------- | ------------- | ----------------- | ------------- |
| Beat Detection (3min track)      | 15s           | 3s                | 5s + upload   |
| Scene Detection (10min video)    | 45s           | 8s                | 12s + upload  |
| Encoding 1080p 1min              | 60s           | 5s (NVENC)        | 8s + transfer |
| AI Upscaling 2x (1min)           | 20min (CPU)   | 2min (local GPU)  | 3min (T4)     |
| **Total Pipeline (5min video)**  | **~8min**     | **~2min**         | **~3min**     |

### Break-Even for cgpu vs Local

- **cgpu is worthwhile when:**
  - No local GPU available
  - Batch jobs with many videos
  - AI workloads (upscaling, LLM)
  - 4K/8K content

- **Local is better when:**
  - Fast local GPU available
  - Small videos (<1min)
  - Low latency more important than throughput
  - Offline operation required

---

## 7. Next Steps

1. **Immediate:** Implement `cgpu_audio.py` for beat detection
2. **Week 1-2:** Scene detection and encoder modules
3. **Week 3-4:** Local GPU optimizations
4. **Week 5-8:** Cluster distribution with Redis queue

---

## References

- [cgpu Repository](https://github.com/RohanAdwankar/cgpu)
- [TransNetV2 Scene Detection](https://github.com/soCzech/TransNetV2)
- [torchaudio Beat Tracking](https://pytorch.org/audio/stable/tutorials/beat_detection_tutorial.html)
- [FFmpeg NVENC Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/nvenc-video-encoder-api-prog-guide/)
