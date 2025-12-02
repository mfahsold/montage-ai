# Open-Source Integration Plan

> Montage AI – extending the platform with external open-source projects

## Status Overview

| Project        | Status        | License         | Priority | GPU Requirement             |
| -------------- | ------------- | --------------- | -------- | --------------------------- |
| VideoAgent     | ✅ Available  | MIT (ECCV 2024) | 🔴 High  | ~8GB VRAM (Video-LLaVA)     |
| Open-Sora 2.0  | ✅ Available  | Apache-2.0      | 🟡 Medium| 256p: 1 GPU, 768p: 8 GPUs   |
| Wan2.1-VACE    | ✅ Available  | Apache-2.0      | 🟡 Medium| 1.3B: 8GB, 14B: 24GB+       |
| FFmpeg-MCP     | ❌ Not found  | -               | → Alt    | -                           |
| Frame AI       | ❌ Not found  | -               | → Alt    | -                           |

## Target Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           MONTAGE AI ORCHESTRATOR                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐  │
│  │ Creative        │      │ VideoAgent      │      │ Footage         │  │
│  │ Director (LLM)  │─────▶│ Memory Agent    │─────▶│ Manager         │  │
│  │ cgpu/Gemini     │      │ (Clip Analysis) │      │ (Selection)     │  │
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

## Sprint 1: Foundation (Weeks 1–2)

### 1.1 MAX_SCENE_REUSE environment variable
**Status:** ✅ already implemented

```python
# src/montage_ai/footage_manager.py:213
MAX_REUSE = int(os.environ.get("MAX_SCENE_REUSE", "3"))

# src/montage_ai/editor.py:106
MAX_SCENE_REUSE = int(os.environ.get("MAX_SCENE_REUSE", "3"))
```

### 1.2 FFmpeg tool wrapper (FFmpeg-MCP alternative)

Because FFmpeg-MCP is unavailable, add an in-house LLM-callable wrapper.

**New file:** `src/montage_ai/ffmpeg_tools.py`

```python
"""
FFmpeg Tool Wrapper - LLM-callable interface

Replaces FFmpeg-MCP with direct tool functions that the Creative Director
can invoke.
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

### 1.3 Clip-pool scoring (PerfectFrameAI fallback)

Extend the existing `footage_analyzer.py` with:

- Frame quality scoring (sharpness, exposure, composition)
- Motion blur detection
- Aesthetic score via CLIP/BLIP

---

## Sprint 2: AI Integration (Weeks 3–4)

### 2.1 VideoAgent integration

**Repository:** https://github.com/YueFan1014/VideoAgent

**Components:**
1. **ReActAgent** – LangChain agent with four tools
2. **Temporal Memory** – caption-based scene search
3. **Object Memory** – SQL database for detected objects
4. **Video-LLaVA** – visual question answering

**New file:** `src/montage_ai/video_agent.py`

```python
"""
VideoAgent Integration - memory-augmented clip analysis

Based on ECCV 2024 paper "VideoAgent"
Repo: github.com/YueFan1014/VideoAgent

Features:
- Temporal memory for scene retrieval
- Object memory for object tracking
- Four tools: caption_retrieval, segment_localization, vqa, object_memory
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import sqlite3

@dataclass
class TemporalMemoryEntry:
    """Entry stored in temporal memory."""
    segment_id: str
    start_time: float
    end_time: float
    caption: str
    embedding: List[float]  # ViCLIP embedding

@dataclass
class ObjectMemoryEntry:
    """Entry stored in object memory."""
    object_id: str
    object_class: str
    first_seen: float
    last_seen: float
    appearances: List[Dict]  # [{timestamp, bbox, confidence}]

class VideoAgentAdapter:
    """
    Adapter for VideoAgent integration.
    
    Uses the four VideoAgent tools:
    1. caption_retrieval – find scenes by description
    2. segment_localization – localize a specific segment
    3. visual_question_answering – answer questions about frames/segments
    4. object_memory_querying – find objects over time
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
        Analyze video and build memory.
        
        Executes:
        1. Scene detection
        2. Caption generation per segment
        3. Object detection and tracking
        4. ViCLIP embedding computation
        """
        pass  # Implementation to follow
    
    def caption_retrieval(self, query: str, top_k: int = 5) -> List[TemporalMemoryEntry]:
        """Tool 1: find scenes based on natural language."""
        pass
    
    def segment_localization(self, description: str) -> Optional[TemporalMemoryEntry]:
        """Tool 2: localize a specific segment."""
        pass
    
    def visual_question_answering(self, question: str, timestamp: float) -> str:
        """Tool 3: answer a question about a frame/segment."""
        pass
    
    def object_memory_querying(self, object_class: str) -> List[ObjectMemoryEntry]:
        """Tool 4: find all occurrences of an object class."""
        pass
```

### 2.2 Wan2.1-VACE service

**Repository:** https://github.com/Wan-Video/Wan2.1

**Models:**
- `Wan2.1-T2V-1.3B` – 8GB VRAM, 480p generation
- `Wan2.1-T2V-14B` – 24GB+ VRAM, 720p generation
- `Wan2.1-VACE-1.3B` – video editing/inpainting

**New file:** `src/montage_ai/wan_vace.py`

```python
"""
Wan2.1-VACE Integration - video editing & generation

Based on Alibaba Wan2.1
Repo: github.com/Wan-Video/Wan2.1

Features:
- Text-to-video generation
- Video inpainting/editing
- Reference-based generation
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class WanVACEConfig:
    """Configuration for Wan2.1-VACE."""
    model_size: str = "1.3B"  # "1.3B" or "14B"
    resolution: str = "480p"   # "480p" or "720p"
    use_cgpu: bool = True      # Use cgpu for cloud GPU
    
class WanVACEService:
    """
    Wan2.1-VACE service for video editing.
    
    Use cases:
    1. B-roll generation from text
    2. Video inpainting (remove objects)
    3. Style transfer
    4. Video extension
    """
    
    def __init__(self, config: WanVACEConfig):
        self.config = config
        self.cgpu_available = self._check_cgpu()
    
    def _check_cgpu(self) -> bool:
        """Check whether cgpu is available."""
        return os.environ.get("CGPU_GPU_ENABLED", "false").lower() == "true"
    
    def generate_broll(
        self, 
        prompt: str,
        duration: float = 3.0,
        reference_frame: Optional[str] = None
    ) -> str:
        """
        Generate a B-roll clip from a text prompt.
        
        Args:
            prompt: Description of the desired video
            duration: Length in seconds (max 5s for 1.3B)
            reference_frame: Optional reference image for style
            
        Returns:
            Path to the generated video
        """
        pass
    
    def inpaint_video(
        self,
        video_path: str,
        mask_path: str,
        prompt: str
    ) -> str:
        """
        Video inpainting – replace masked areas.
        
        Args:
            video_path: Input video
            mask_path: Mask (white = replace)
            prompt: What should be inserted
            
        Returns:
            Path to the edited video
        """
        pass
```

### 2.3 Open-Sora 2.0 generator

**Repository:** https://github.com/hpcaitech/Open-Sora

**Hugging Face:** `hpcai-tech/Open-Sora-v2`

**Specs:**
- 11B parameters
- Text-to-video (T2V)
- Image-to-video (I2V)
- 256p: 1 GPU
- 768p: 8 GPUs (not realistic via cgpu)

**New file:** `src/montage_ai/open_sora.py`

```python
"""
Open-Sora 2.0 Integration - text-to-video generation

Based on HPC-AI Tech Open-Sora
Repo: github.com/hpcaitech/Open-Sora
Model: hpcai-tech/Open-Sora-v2

Features:
- Text-to-video generation
- Image-to-video extension
- Apache-2.0 license
"""

import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class OpenSoraConfig:
    """Configuration for Open-Sora."""
    resolution: str = "256p"  # realistic for one GPU
    num_frames: int = 51      # ~2 seconds
    use_cgpu: bool = True
    model_id: str = "hpcai-tech/Open-Sora-v2"


class OpenSoraGenerator:
    """
    Open-Sora video generator.
    
    Note: 768p requires eight GPUs and is not practical via cgpu.
    Use 256p for generation, then upscale with Real-ESRGAN.
    
    Pipeline:
    1. Open-Sora generates 256p video
    2. Real-ESRGAN upscales to 1024p
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
        Generate a video from a text prompt.
        
        Args:
            prompt: Description of the video
            negative_prompt: What to avoid
            duration: Length in seconds
            reference_image: For I2V – starting image
            
        Returns:
            Path to the generated video
        """
        # cgpu run for Colab execution
        pass
    
    def extend_video(
        self,
        video_path: str,
        prompt: str,
        extend_seconds: float = 2.0
    ) -> str:
        """
        Extend an existing video.
        
        Args:
            video_path: Input video
            prompt: Description of the extension
            extend_seconds: How many seconds to add
            
        Returns:
            Path to the extended video
        """
        pass
```

---

## Sprint 3: Integration & Testing (Weeks 5–6)

### 3.1 Editor integration

Extend `editor.py` with the new services:

```python
# In editor.py

# Import new services
from .video_agent import VideoAgentAdapter
from .wan_vace import WanVACEService, WanVACEConfig
from .open_sora import OpenSoraGenerator, OpenSoraConfig
from .ffmpeg_tools import TOOLS as FFMPEG_TOOLS

# Feature flags
ENABLE_VIDEO_AGENT = os.environ.get("ENABLE_VIDEO_AGENT", "false").lower() == "true"
ENABLE_WAN_VACE = os.environ.get("ENABLE_WAN_VACE", "false").lower() == "true"
ENABLE_OPEN_SORA = os.environ.get("ENABLE_OPEN_SORA", "false").lower() == "true"
```

### 3.2 Docker Compose updates

```yaml
# docker-compose.yml - new services

services:
  montage-ai:
    # ... existing config
    environment:
      # Feature flags
      - ENABLE_VIDEO_AGENT=${ENABLE_VIDEO_AGENT:-false}
      - ENABLE_WAN_VACE=${ENABLE_WAN_VACE:-false}
      - ENABLE_OPEN_SORA=${ENABLE_OPEN_SORA:-false}
      # cgpu config
      - CGPU_GPU_ENABLED=${CGPU_GPU_ENABLED:-false}
      - CGPU_HOST=${CGPU_HOST:-localhost}
      - CGPU_PORT=${CGPU_PORT:-5021}

  # cgpu serve as sidecar (already planned)
  cgpu-serve:
    image: alpine
    command: ["cgpu", "serve", "--port", "5021"]
    ports:
      - "5021:5021"
```

---

## Dependencies

```
requirements.txt (additions)
================================
# VideoAgent dependencies
langchain>=0.1.0
langchain-community>=0.0.20
transformers>=4.36.0
sentence-transformers>=2.2.0

# Video processing
decord>=0.6.0          # fast video loading
viclip                 # Video-CLIP embeddings (optional)

# Database
sqlalchemy>=2.0.0      # for object memory
```

---

## Implementation order

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Foundation (this week)                               │
│  ──────────────────────────────────────────────────────────────│
│  [x] MAX_SCENE_REUSE already implemented                       │
│  [ ] ffmpeg_tools.py - LLM-callable wrapper                     │
│  [ ] footage_analyzer.py - extend frame quality scoring         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: VideoAgent (weeks 2–3)                               │
│  ──────────────────────────────────────────────────────────────│
│  [ ] video_agent.py - implement adapter                         │
│  [ ] Temporal Memory with ViCLIP embeddings                     │
│  [ ] Object Memory with SQLite                                  │
│  [ ] Integration in footage_manager.py                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Generative AI (weeks 4–5)                            │
│  ──────────────────────────────────────────────────────────────│
│  [ ] wan_vace.py - B-roll generation                            │
│  [ ] open_sora.py - text-to-video                               │
│  [ ] cgpu GPU pipeline for both                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Polish (week 6)                                      │
│  ──────────────────────────────────────────────────────────────│
│  [ ] End-to-end tests                                           │
│  [ ] Documentation                                              │
│  [ ] Performance optimization                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Next steps

1. **Immediate:** implement `ffmpeg_tools.py` (replacement for FFmpeg-MCP)
2. **This week:** scaffold `video_agent.py`
3. **Review:** pick priority between Wan2.1 and Open-Sora

---

## Risks & mitigations

| Risk                              | Mitigation                            |
| --------------------------------- | ------------------------------------- |
| VideoAgent models too large for cgpu | Use smaller Video-LLaVA variant     |
| Open-Sora 768p needs eight GPUs   | Use 256p + Real-ESRGAN upscaling      |
| Wan2.1 14B too slow               | Stay on 1.3B for 480p                 |
| cgpu rate limits                  | Caching, batch processing             |

---

*Created: 2025-12-02*
*Last updated: 2025-12-02*
