# Open-Source Integration Plan

> Montage AI - Erweiterung durch externe Open-Source Projekte

## Status Übersicht

| Projekt | Status | Lizenz | Priorität | GPU Anforderung |
|---------|--------|--------|-----------|-----------------|
| VideoAgent | ✅ Verfügbar | MIT (ECCV 2024) | 🔴 Hoch | ~8GB VRAM (Video-LLaVA) |
| Open-Sora 2.0 | ✅ Verfügbar | Apache-2.0 | 🟡 Mittel | 256p: 1 GPU, 768p: 8 GPUs |
| Wan2.1-VACE | ✅ Verfügbar | Apache-2.0 | 🟡 Mittel | 1.3B: 8GB, 14B: 24GB+ |
| FFmpeg-MCP | ❌ Nicht gefunden | - | → Alternative | - |
| Frame AI | ❌ Nicht gefunden | - | → Alternative | - |

## Architektur nach Integration

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           MONTAGE AI ORCHESTRATOR                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐  │
│  │ Creative        │      │ VideoAgent      │      │ Footage         │  │
│  │ Director (LLM)  │─────▶│ Memory Agent    │─────▶│ Manager         │  │
│  │ cgpu/Gemini     │      │ (Clip-Analyse)  │      │ (Selection)     │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘  │
│           │                       │                        │            │
│           │                       ▼                        │            │
│           │               ┌─────────────────┐              │            │
│           │               │ Temporal Memory │              │            │
│           │               │ Object Memory   │              │            │
│           │               │ (SQL Database)  │              │            │
│           │               └─────────────────┘              │            │
│           │                                                │            │
│           ▼                                                ▼            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        EDITOR (editor.py)                       │    │
│  │   Beat Detection │ Scene Assembly │ Transitions │ Rendering     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                       │                        │            │
│           ▼                       ▼                        ▼            │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐  │
│  │ FFmpeg Tools    │      │ Wan2.1-VACE     │      │ Open-Sora 2.0   │  │
│  │ (Local Wrapper) │      │ Video Editing   │      │ B-Roll Gen      │  │
│  │                 │      │ cgpu GPU        │      │ cgpu GPU        │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘  │
│           │                       │                        │            │
│           └───────────────────────┴────────────────────────┘            │
│                                   │                                      │
│                                   ▼                                      │
│                    ┌─────────────────────────────┐                       │
│                    │ cgpu Upscaler (Real-ESRGAN) │                       │
│                    │ T4 GPU via Colab            │                       │
│                    └─────────────────────────────┘                       │
│                                   │                                      │
│                                   ▼                                      │
│                           Final Video Output                             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Sprint 1: Foundation (Woche 1-2)

### 1.1 MAX_SCENE_REUSE ENV Variable
**Status:** ✅ Bereits implementiert

```python
# src/montage_ai/footage_manager.py:213
MAX_REUSE = int(os.environ.get("MAX_SCENE_REUSE", "3"))

# src/montage_ai/editor.py:106
MAX_SCENE_REUSE = int(os.environ.get("MAX_SCENE_REUSE", "3"))
```

### 1.2 FFmpeg Tool Wrapper (Alternative zu FFmpeg-MCP)

Da FFmpeg-MCP nicht gefunden wurde, implementieren wir einen eigenen LLM-callable Wrapper.

**Neue Datei:** `src/montage_ai/ffmpeg_tools.py`

```python
"""
FFmpeg Tool Wrapper - LLM-Callable Interface

Ersetzt FFmpeg-MCP mit direkten Tool-Funktionen die vom 
Creative Director aufgerufen werden können.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import subprocess
import json

@dataclass
class FFmpegTool:
    """Tool definition for LLM calling."""
    name: str
    description: str
    parameters: Dict[str, Any]
    
TOOLS = [
    FFmpegTool(
        name="extract_frames",
        description="Extract frames from video at specific timestamps",
        parameters={
            "input": "Path to input video",
            "timestamps": "List of timestamps in seconds",
            "output_dir": "Directory for extracted frames"
        }
    ),
    FFmpegTool(
        name="create_segment",
        description="Extract video segment between two timestamps",
        parameters={
            "input": "Path to input video",
            "start": "Start timestamp in seconds",
            "end": "End timestamp in seconds", 
            "output": "Path to output video"
        }
    ),
    FFmpegTool(
        name="apply_transition",
        description="Apply transition between two clips",
        parameters={
            "clip_a": "Path to first clip",
            "clip_b": "Path to second clip",
            "transition": "Type: crossfade, wipe, slide",
            "duration": "Transition duration in seconds"
        }
    ),
    FFmpegTool(
        name="color_grade",
        description="Apply color grading LUT to video",
        parameters={
            "input": "Path to input video",
            "lut": "LUT name or path",
            "intensity": "0.0-1.0"
        }
    )
]
```

### 1.3 Clip-Pool Scoring mit PerfectFrameAI-Alternative

Da PerfectFrameAI nicht gefunden wurde, erweitern wir den bestehenden `footage_analyzer.py`:

**Erweiterung:** `src/montage_ai/footage_analyzer.py`

- Frame-Quality-Scoring (Schärfe, Belichtung, Komposition)
- Motion-Blur-Detection
- Aesthetic Score via CLIP/BLIP

---

## Sprint 2: AI Integration (Woche 3-4)

### 2.1 VideoAgent Integration

**Repository:** https://github.com/YueFan1014/VideoAgent

**Komponenten:**
1. **ReActAgent** - LangChain-basierter Agent mit 4 Tools
2. **Temporal Memory** - Caption-basierte Szenen-Suche
3. **Object Memory** - SQL-Datenbank für erkannte Objekte
4. **Video-LLaVA** - Visual Question Answering

**Neue Datei:** `src/montage_ai/video_agent.py`

```python
"""
VideoAgent Integration - Memory-Augmented Clip Analysis

Basiert auf: ECCV 2024 Paper "VideoAgent"
Repo: github.com/YueFan1014/VideoAgent

Features:
- Temporal Memory für Szenen-Retrieval
- Object Memory für Objekt-Tracking
- 4 Tools: caption_retrieval, segment_localization, VQA, object_memory
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import sqlite3

@dataclass
class TemporalMemoryEntry:
    """Ein Eintrag in der Temporal Memory."""
    segment_id: str
    start_time: float
    end_time: float
    caption: str
    embedding: List[float]  # ViCLIP embedding

@dataclass  
class ObjectMemoryEntry:
    """Ein Eintrag in der Object Memory."""
    object_id: str
    object_class: str
    first_seen: float
    last_seen: float
    appearances: List[Dict]  # [{timestamp, bbox, confidence}]

class VideoAgentAdapter:
    """
    Adapter für VideoAgent Integration.
    
    Nutzt die 4 Tools des VideoAgent:
    1. caption_retrieval - Finde Szenen basierend auf Beschreibung
    2. segment_localization - Lokalisiere spezifische Segmente
    3. visual_question_answering - Beantworte Fragen über Video
    4. object_memory_querying - Finde Objekte über Zeit
    """
    
    def __init__(self, db_path: str = "/tmp/video_agent.db"):
        self.temporal_memory: List[TemporalMemoryEntry] = []
        self.object_memory: List[ObjectMemoryEntry] = []
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite for object memory."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                id TEXT PRIMARY KEY,
                class TEXT,
                first_seen REAL,
                last_seen REAL,
                appearances TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                caption TEXT,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()
    
    def analyze_footage(self, video_path: str) -> Dict:
        """
        Analysiere Video und baue Memory auf.
        
        Führt aus:
        1. Szenen-Erkennung
        2. Caption-Generierung pro Segment
        3. Objekt-Erkennung und Tracking
        4. ViCLIP Embedding-Berechnung
        """
        pass  # Implementation folgt
    
    def caption_retrieval(self, query: str, top_k: int = 5) -> List[TemporalMemoryEntry]:
        """Tool 1: Finde Szenen basierend auf natürlicher Sprache."""
        pass
    
    def segment_localization(self, description: str) -> Optional[TemporalMemoryEntry]:
        """Tool 2: Lokalisiere ein spezifisches Segment."""
        pass
    
    def visual_question_answering(self, question: str, timestamp: float) -> str:
        """Tool 3: Beantworte Frage über Frame/Segment."""
        pass
    
    def object_memory_querying(self, object_class: str) -> List[ObjectMemoryEntry]:
        """Tool 4: Finde alle Vorkommen eines Objekttyps."""
        pass
```

### 2.2 Wan2.1-VACE Service

**Repository:** https://github.com/Wan-Video/Wan2.1

**Modelle:**
- `Wan2.1-T2V-1.3B` - 8GB VRAM, 480p Generation
- `Wan2.1-T2V-14B` - 24GB+ VRAM, 720p Generation
- `Wan2.1-VACE-1.3B` - Video Editing/Inpainting

**Neue Datei:** `src/montage_ai/wan_vace.py`

```python
"""
Wan2.1-VACE Integration - Video Editing & Generation

Basiert auf: Alibaba Wan2.1
Repo: github.com/Wan-Video/Wan2.1

Features:
- Text-to-Video Generation
- Video Inpainting/Editing
- Reference-based Generation
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class WanVACEConfig:
    """Konfiguration für Wan2.1-VACE."""
    model_size: str = "1.3B"  # "1.3B" oder "14B"
    resolution: str = "480p"   # "480p" oder "720p"
    use_cgpu: bool = True      # Nutze cgpu für Cloud GPU
    
class WanVACEService:
    """
    Wan2.1-VACE Service für Video-Editing.
    
    Anwendungsfälle:
    1. B-Roll Generation aus Text
    2. Video Inpainting (Objekte entfernen)
    3. Style Transfer
    4. Video Extension
    """
    
    def __init__(self, config: WanVACEConfig):
        self.config = config
        self.cgpu_available = self._check_cgpu()
    
    def _check_cgpu(self) -> bool:
        """Prüfe ob cgpu verfügbar ist."""
        return os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true"
    
    def generate_broll(
        self, 
        prompt: str,
        duration: float = 3.0,
        reference_frame: Optional[str] = None
    ) -> str:
        """
        Generiere B-Roll Video aus Text-Prompt.
        
        Args:
            prompt: Beschreibung des gewünschten Videos
            duration: Länge in Sekunden (max 5s für 1.3B)
            reference_frame: Optional - Referenzbild für Stil
            
        Returns:
            Pfad zum generierten Video
        """
        pass
    
    def inpaint_video(
        self,
        video_path: str,
        mask_path: str,
        prompt: str
    ) -> str:
        """
        Video Inpainting - Ersetze maskierte Bereiche.
        
        Args:
            video_path: Eingabe-Video
            mask_path: Maske (weiß = zu ersetzen)
            prompt: Was soll eingefügt werden
            
        Returns:
            Pfad zum bearbeiteten Video
        """
        pass
```

### 2.3 Open-Sora 2.0 Generator

**Repository:** https://github.com/hpcaitech/Open-Sora

**HuggingFace:** `hpcai-tech/Open-Sora-v2`

**Spezifikationen:**
- 11B Parameter
- Text-to-Video (T2V)
- Image-to-Video (I2V)
- 256p: 1 GPU
- 768p: 8 GPUs (via cgpu nicht realistisch)

**Neue Datei:** `src/montage_ai/open_sora.py`

```python
"""
Open-Sora 2.0 Integration - Text-to-Video Generation

Basiert auf: HPC-AI Tech Open-Sora
Repo: github.com/hpcaitech/Open-Sora
Model: hpcai-tech/Open-Sora-v2

Features:
- Text-to-Video Generation
- Image-to-Video Erweiterung
- Apache-2.0 Lizenz
"""

import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class OpenSoraConfig:
    """Konfiguration für Open-Sora."""
    resolution: str = "256p"  # Realistisch für 1 GPU
    num_frames: int = 51      # ~2 Sekunden
    use_cgpu: bool = True
    model_id: str = "hpcai-tech/Open-Sora-v2"

class OpenSoraGenerator:
    """
    Open-Sora Video Generator.
    
    Hinweis: 768p benötigt 8 GPUs und ist via cgpu
    nicht praktikabel. Nutze 256p für Generierung,
    dann cgpu Real-ESRGAN für Upscaling.
    
    Pipeline:
    1. Open-Sora generiert 256p Video
    2. Real-ESRGAN upscaled auf 1024p
    """
    
    def __init__(self, config: OpenSoraConfig):
        self.config = config
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        duration: float = 2.0,
        reference_image: Optional[str] = None
    ) -> str:
        """
        Generiere Video aus Text-Prompt.
        
        Args:
            prompt: Beschreibung des Videos
            negative_prompt: Was vermieden werden soll
            duration: Länge in Sekunden
            reference_image: Für I2V - Startbild
            
        Returns:
            Pfad zum generierten Video
        """
        # cgpu run für Colab-Ausführung
        pass
    
    def extend_video(
        self,
        video_path: str,
        prompt: str,
        extend_seconds: float = 2.0
    ) -> str:
        """
        Verlängere bestehendes Video.
        
        Args:
            video_path: Eingabe-Video
            prompt: Beschreibung der Fortsetzung
            extend_seconds: Wie viel hinzufügen
            
        Returns:
            Pfad zum verlängerten Video
        """
        pass
```

---

## Sprint 3: Integration & Testing (Woche 5-6)

### 3.1 Editor Integration

Erweitere `editor.py` um die neuen Services:

```python
# In editor.py

# Import neue Services
from .video_agent import VideoAgentAdapter
from .wan_vace import WanVACEService, WanVACEConfig
from .open_sora import OpenSoraGenerator, OpenSoraConfig
from .ffmpeg_tools import TOOLS as FFMPEG_TOOLS

# Feature Flags
ENABLE_VIDEO_AGENT = os.environ.get("ENABLE_VIDEO_AGENT", "false").lower() == "true"
ENABLE_WAN_VACE = os.environ.get("ENABLE_WAN_VACE", "false").lower() == "true"
ENABLE_OPEN_SORA = os.environ.get("ENABLE_OPEN_SORA", "false").lower() == "true"
```

### 3.2 Docker Compose Updates

```yaml
# docker-compose.yml - Neue Services

services:
  montage-ai:
    # ... existing config
    environment:
      # Neue Feature Flags
      - ENABLE_VIDEO_AGENT=${ENABLE_VIDEO_AGENT:-false}
      - ENABLE_WAN_VACE=${ENABLE_WAN_VACE:-false}
      - ENABLE_OPEN_SORA=${ENABLE_OPEN_SORA:-false}
      # cgpu Config
      - CGPU_GPU_ENABLED=${CGPU_GPU_ENABLED:-false}
      - CGPU_HOST=${CGPU_HOST:-localhost}
      - CGPU_PORT=${CGPU_PORT:-5021}

  # cgpu serve als Sidecar (bereits geplant)
  cgpu-serve:
    image: alpine
    command: ["cgpu", "serve", "--port", "5021"]
    ports:
      - "5021:5021"
```

---

## Abhängigkeiten

```
requirements.txt (Ergänzungen)
================================
# VideoAgent Dependencies
langchain>=0.1.0
langchain-community>=0.0.20
transformers>=4.36.0
sentence-transformers>=2.2.0

# Video Processing
decord>=0.6.0          # Für schnelles Video-Loading
viclip                 # Video-CLIP Embeddings (optional)

# Database
sqlalchemy>=2.0.0      # Für Object Memory
```

---

## Implementierungs-Reihenfolge

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Foundation (Diese Woche)                             │
│  ──────────────────────────────────────────────────────────────│
│  [x] MAX_SCENE_REUSE bereits implementiert                     │
│  [ ] ffmpeg_tools.py - LLM-callable Wrapper                    │
│  [ ] footage_analyzer.py - Frame Quality Scoring erweitern     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: VideoAgent (Woche 2-3)                               │
│  ──────────────────────────────────────────────────────────────│
│  [ ] video_agent.py - Adapter implementieren                   │
│  [ ] Temporal Memory mit ViCLIP Embeddings                     │
│  [ ] Object Memory mit SQLite                                  │
│  [ ] Integration in footage_manager.py                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Generative AI (Woche 4-5)                            │
│  ──────────────────────────────────────────────────────────────│
│  [ ] wan_vace.py - B-Roll Generation                           │
│  [ ] open_sora.py - Text-to-Video                              │
│  [ ] cgpu GPU Pipeline für beide                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Polish (Woche 6)                                     │
│  ──────────────────────────────────────────────────────────────│
│  [ ] End-to-End Tests                                          │
│  [ ] Documentation                                             │
│  [ ] Performance Optimization                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Nächste Schritte

1. **Sofort:** `ffmpeg_tools.py` implementieren (Ersatz für FFmpeg-MCP)
2. **Diese Woche:** `video_agent.py` Grundstruktur
3. **Review:** Entscheiden ob Wan2.1 oder Open-Sora Priorität hat

---

## Risiken & Mitigationen

| Risiko | Mitigation |
|--------|------------|
| VideoAgent Models zu groß für cgpu | Nutze kleinere Video-LLaVA Variante |
| Open-Sora 768p braucht 8 GPUs | 256p + Real-ESRGAN Upscaling |
| Wan2.1 14B zu langsam | Bleibe bei 1.3B für 480p |
| cgpu Rate Limits | Caching, Batch Processing |

---

*Erstellt: $(date)*
*Letzte Aktualisierung: -* 
