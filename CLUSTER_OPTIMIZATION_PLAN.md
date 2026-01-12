# Cluster-Optimierungsplan für Montage-AI

## Executive Summary

Ziel: Nutzung aller 8 Cluster-Nodes für parallele Video-Verarbeitung.
Geschätzte Verbesserung: **5-10x schnellere Render-Zeiten** für große Jobs.

---

## Phase 1: Distributed Scene Detection (Effort: 2h)

### Problem
Aktuell: Scene Detection läuft nur auf einem Node (CPU-bound, ~20min für 65min 4K).

### Lösung
Kubernetes Job-basierte parallele Analyse über mehrere Nodes.

### Implementierung

```yaml
# deploy/k3s/overlays/distributed/scene-detection-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: scene-detect-${VIDEO_HASH}
spec:
  parallelism: 4  # 4 Nodes parallel
  completions: 4
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                job-name: scene-detect-${VIDEO_HASH}
            topologyKey: kubernetes.io/hostname
      containers:
      - name: scene-detect
        image: 192.168.1.12:30500/montage-ai:latest
        env:
        - name: SHARD_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: SHARD_COUNT
          value: "4"
        command: ["python", "-m", "montage_ai.scene_detection", "--shard"]
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
```

### Node-Zuweisung

| Node | Shard | Video-Segment |
|------|-------|---------------|
| codeai-thinkpad-t14s-gen-6 | 0 | 0-25% |
| codeai-esprimo-q958 | 1 | 25-50% |
| codeai | 2 | 50-75% |
| codeai-worker-amd64 | 3 | 75-100% |

---

## Phase 2: GPU-Accelerated Encoding (Effort: 3h)

### Problem
Aktuell: Software-Encoding (libx264) ist langsam.

### Lösung
GPU-basiertes Encoding auf dedizierten Nodes.

### Hardware-Mapping

| Node | Encoder | Codec | Speed |
|------|---------|-------|-------|
| codeai-fluxibriserver | **ROCm/AMF** | h264_amf, hevc_amf | 10-20x |
| codeaijetson-desktop | **NVENC** | h264_nvenc | 5-10x |
| codeai | VAAPI | h264_vaapi | 3-5x |

### Implementierung

```yaml
# deploy/k3s/overlays/distributed/gpu-encoder-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: montage-ai-encoder-amd
spec:
  replicas: 1
  selector:
    matchLabels:
      app: montage-ai-encoder
      gpu: amd
  template:
    spec:
      nodeSelector:
        amd.com/gpu: "present"
      containers:
      - name: encoder
        image: 192.168.1.12:30500/montage-ai:latest
        env:
        - name: FFMPEG_HWACCEL
          value: "amf"
        - name: OUTPUT_CODEC
          value: "h264_amf"
        resources:
          limits:
            amd.com/gpu: 1
          requests:
            memory: "8Gi"
            cpu: "4"
```

---

## Phase 3: Distributed Segment Rendering (Effort: 4h)

### Problem
Aktuell: Segment-Verarbeitung ist seriell auf einem Node.

### Lösung
Jeder Segment wird auf einem separaten Node gerendert.

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Redis Job Queue                          │
│  [Seg1] [Seg2] [Seg3] [Seg4] [Seg5] [Seg6] [Seg7] [Seg8]   │
└─────────────────────────────────────────────────────────────┘
           │       │       │       │       │
           ▼       ▼       ▼       ▼       ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
     │ Worker1 │ │ Worker2 │ │ Worker3 │ │ Worker4 │
     │ (AMD)   │ │(Jetson) │ │(Esprimo)│ │(Thinkpad│
     └─────────┘ └─────────┘ └─────────┘ └─────────┘
           │       │       │       │
           ▼       ▼       ▼       ▼
     ┌─────────────────────────────────────────────┐
     │         NFS: /data/output/segments/         │
     └─────────────────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Final Concat   │
              │  (AMD GPU Node) │
              └─────────────────┘
```

### Worker Deployment

```yaml
# deploy/k3s/overlays/distributed/worker-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: montage-ai-worker
spec:
  selector:
    matchLabels:
      app: montage-ai-worker
  template:
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node-role.kubernetes.io/control-plane
                operator: DoesNotExist
      containers:
      - name: worker
        image: 192.168.1.12:30500/montage-ai:latest
        env:
        - name: WORKER_MODE
          value: "true"
        - name: REDIS_URL
          value: "redis://redis.montage-ai:6379"
        volumeMounts:
        - name: nfs-data
          mountPath: /data
      volumes:
      - name: nfs-data
        nfs:
          server: 192.168.1.12
          path: /data/montage-ai
```

---

## Phase 4: Intelligent Job Scheduling (Effort: 2h)

### Node-Capability Matrix

```python
# src/montage_ai/cluster/node_capabilities.py
NODE_CAPABILITIES = {
    "codeai-fluxibriserver": {
        "gpu": "amd_rx7900xtx",
        "vram_gb": 24,
        "ram_gb": 47,
        "cpu_cores": 8,
        "optimal_for": ["4k_encoding", "8k_encoding", "final_render"],
        "encoder": "h264_amf",
    },
    "codeaijetson-desktop": {
        "gpu": "nvidia_jetson",
        "vram_gb": 8,
        "ram_gb": 7.4,
        "cpu_cores": 6,
        "optimal_for": ["1080p_encoding", "preview_render"],
        "encoder": "h264_nvenc",
    },
    "codeai-thinkpad-t14s-gen-6": {
        "gpu": None,
        "ram_gb": 29,
        "cpu_cores": 12,
        "optimal_for": ["scene_detection", "audio_analysis", "cpu_intensive"],
    },
    "codeai-esprimo-q958": {
        "gpu": None,
        "ram_gb": 31,
        "cpu_cores": 8,
        "optimal_for": ["4k_analysis", "memory_intensive", "batch_processing"],
    },
}

def select_node_for_task(task_type: str, resolution: tuple) -> str:
    """Select optimal node for a given task."""
    if task_type == "final_render" and resolution[1] >= 2160:  # 4K+
        return "codeai-fluxibriserver"  # AMD GPU
    elif task_type == "scene_detection":
        return "codeai-thinkpad-t14s-gen-6"  # Most CPU cores
    elif task_type == "audio_analysis":
        return "codeai-esprimo-q958"  # Most RAM
    # ... etc
```

---

## Erwartete Verbesserungen

### Render-Zeiten (30s Montage aus 4K Material)

| Phase | Aktuell | Mit Cluster | Speedup |
|-------|---------|-------------|---------|
| Scene Detection | 20min | 5min | 4x |
| Clip Enhancement | 10min | 2min | 5x |
| Segment Render | 15min | 3min | 5x |
| Final Concat | 5min | 1min | 5x |
| **Gesamt** | **50min** | **11min** | **4.5x** |

### Für 4K Langform (60min Video → 5min Montage)

| Phase | Aktuell | Mit Cluster | Speedup |
|-------|---------|-------------|---------|
| Proxy Generation | 30min | 8min | 4x |
| Scene Detection | 60min | 15min | 4x |
| Clip Selection | 20min | 5min | 4x |
| Rendering | 45min | 5min | 9x |
| **Gesamt** | **2.5h** | **33min** | **4.5x** |

---

## Implementierungs-Reihenfolge

1. **Phase 1** (Tag 1): Distributed Scene Detection
   - NFS Storage für shared data
   - Job-Template für Sharding
   - Ergebnis-Aggregation

2. **Phase 2** (Tag 2): GPU Encoding
   - AMD AMF Integration
   - Jetson NVENC Fallback
   - Encoder-Router

3. **Phase 3** (Tag 3-4): Worker DaemonSet
   - Redis Queue Integration
   - Segment-Distribution
   - Progress Tracking

4. **Phase 4** (Tag 5): Smart Scheduling
   - Node-Capability Detection
   - Automatic Task Routing
   - Load Balancing

---

## Voraussetzungen

- [x] NFS Storage bereits konfiguriert
- [x] Redis bereits deployed
- [x] AMD GPU Device Plugin aktiv
- [x] NVIDIA GPU Device Plugin aktiv
- [x] **Cluster Module implementiert** (src/montage_ai/cluster/)
- [x] **Task Router** - Hardware-basiertes Task Routing
- [x] **Job Submitter** - K8s Job Submission API
- [x] **Distributed Scene Detection** - Sharding Support
- [ ] Multi-arch Image für ARM64 (Jetson, Pi)
- [ ] Worker-Mode in montage-ai implementieren

---

## Cluster Module - Generalisiert & Stabilisiert

Das Cluster-Modul ist jetzt **vollständig generalisiert** für heterogene Hardware:

### Features
- **Auto-Detection**: Erkennt lokale Hardware (CPU, RAM, GPU) automatisch
- **Multi-Mode**: LOCAL, K8S, CONFIG, AUTO Modi
- **GPU-Support**: NVIDIA, AMD, Intel, Apple Silicon, Qualcomm
- **YAML-Config**: Cluster-Definition per Konfigurationsdatei
- **Graceful Degradation**: Fällt auf lokalen Modus zurück wenn K8s nicht verfügbar

### Environment Variables
```bash
MONTAGE_CLUSTER_MODE=auto    # local, k8s, config, auto
MONTAGE_CLUSTER_CONFIG=/path/to/cluster.yaml
MONTAGE_FORCE_CPU=1          # Disable GPU detection
```

Das Cluster-Modul (`src/montage_ai/cluster/`) bietet:

### Python API

```python
from montage_ai.cluster import (
    get_cluster_manager,
    TaskRouter,
    TaskType,
    JobSubmitter
)

# Cluster-Übersicht
cluster = get_cluster_manager()
cluster.print_cluster_summary()

# Task Routing
router = TaskRouter(cluster)
node = router.route_task(TaskType.GPU_ENCODING, resolution=(3840, 2160))
print(f"Best GPU node: {node.name}")

# Parallele Jobs erstellen
videos = ["/data/input/v1.mp4", "/data/input/v2.mp4"]
jobs = router.create_parallel_jobs(TaskType.CPU_SCENE_DETECTION, videos)

# K8s Job Submission
submitter = JobSubmitter()
job = submitter.submit_scene_detection(videos, parallelism=4)
submitter.wait_for_job(job.name)
scenes = submitter.get_scene_results(job.name)
```

### CLI - Distributed Scene Detection

```bash
# Einzelnes Video (Time-based Sharding)
python -m montage_ai.cluster.distributed_scene_detection \
    --video /data/input/large_video.mp4 \
    --shard-index 0 --shard-count 4

# Multiple Videos (File-based Sharding)
python -m montage_ai.cluster.distributed_scene_detection \
    --videos /data/input/v1.mp4,/data/input/v2.mp4 \
    --shard-index 0 --shard-count 2

# Ergebnisse aggregieren
python -m montage_ai.cluster.distributed_scene_detection \
    --aggregate --job-id scene-detect-abc123
```

### K8s Manifests

```bash
# Distributed Scene Detection Job
kubectl apply -f deploy/k3s/distributed/scene-detection-job.yaml

# Worker DaemonSet (alle Nodes)
kubectl apply -f deploy/k3s/distributed/worker-daemonset.yaml

# Vollständige Kustomization
kubectl apply -k deploy/k3s/distributed/
```

### Task Types

| TaskType | Beschreibung | Optimale Nodes |
|----------|--------------|----------------|
| GPU_ENCODING | H.264/HEVC GPU-Encoding | AMD GPU, Jetson |
| CPU_SCENE_DETECTION | Scene Detection (CPU) | High-CPU Nodes |
| MEMORY_INTENSIVE | RAM-intensive Tasks | High-Memory Nodes |
| PROXY_GENERATION | Proxy-Erstellung | Any Node |
| FINAL_RENDER | Final Compositing | GPU Nodes |

---

## Quick Wins (Sofort umsetzbar)

1. **GPU-Encoding aktivieren** auf codeai-fluxibriserver
   ```bash
   kubectl set env deploy/montage-ai-web -n montage-ai \
     FFMPEG_HWACCEL=amf OUTPUT_CODEC=h264_amf
   ```

2. **Mehr Worker-Replicas**
   ```bash
   kubectl scale deploy/montage-ai-worker -n montage-ai --replicas=4
   ```

3. **Memory Limits erhöhen** für 4K-Jobs
   ```bash
   kubectl set resources deploy/montage-ai-worker -n montage-ai \
     --requests=memory=8Gi --limits=memory=16Gi
   ```
