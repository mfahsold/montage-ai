# Professional Export Architecture (OTIO & NLE Workflow)

## 1. Executive Summary
To transform Montage AI into a professional post-production assistant, we must support industry-standard interchange formats. We will adopt **OpenTimelineIO (OTIO)** as our primary export format, enabling seamless integration with DaVinci Resolve, Adobe Premiere Pro, and Final Cut Pro.

**Goal:** Users generate a "rough cut" in Montage AI and finish/grade it in a professional NLE.

## 2. Standards & Trends (2024/2025)

### OpenTimelineIO (OTIO)
*   **Status:** Academy Software Foundation (ASWF) standard.
*   **Adoption:** Native support in DaVinci Resolve 17+, Unreal Engine 5, Blender 3+.
*   **Version:** We will target **OTIO 0.15+** (latest stable).
*   **Advantage:** Supports multiple tracks, transitions, markers, and metadata (unlike EDL).

### CMX 3600 EDL
*   **Status:** Legacy but universal.
*   **Use Case:** Fallback for older systems or simple "cuts-only" transfers.

### FCPXML (Final Cut Pro XML)
*   **Status:** Standard for Apple ecosystem.
*   **Use Case:** Richer metadata support for Final Cut Pro X and Premiere.

## 3. The "Proxy Workflow"

Montage AI often runs on limited hardware (laptops). We will implement a **Proxy-First Workflow**:

1.  **Ingest:** Montage AI analyzes high-res footage but generates/uses low-res proxies for internal processing (optional).
2.  **Edit:** The AI constructs the timeline using internal references.
3.  **Export:** The OTIO file references the **original high-res source files** on disk.
4.  **Relink:** When opened in Resolve/Premiere, the NLE automatically links to the high-res source.

## 4. Metadata Mapping Strategy

We will enrich the NLE timeline with AI analysis data:

| Montage AI Data | OTIO/NLE Mapping | Visual Result in NLE |
| :--- | :--- | :--- |
| **Clip Energy (High)** | Marker (Red) | Red flag on timeline |
| **Clip Energy (Low)** | Marker (Blue) | Blue flag on timeline |
| **Scene Description** | Clip Comment / Note | Text visible in inspector |
| **Beat Detection** | Timeline Markers | Vertical lines on beat points |
| **Stabilization Fix** | Metadata Tag | "Needs Stabilization" note |

## 5. Architecture: `timeline_exporter.py`

We will refactor the existing module into a robust class structure:

```python
class TimelineExporter:
    def export(self, timeline: MontageTimeline, format: str = "otio") -> str:
        ...

class OTIOAdapter(ExportAdapter):
    """Handles conversion to OpenTimelineIO schema."""
    def build_timeline(self):
        # 1. Create OTIO Timeline
        # 2. Add Video Track
        # 3. Add Audio Track
        # 4. Map Clips & Effects
        # 5. Serialize
```

## 6. Implementation Roadmap

### Step 1: Dependencies
*   Add `opentimelineio>=0.15.0` to `requirements.txt`.

### Step 2: Core Implementation
*   Create `src/montage_ai/exports/` package.
*   Implement `otio_adapter.py`.
*   Implement `edl_adapter.py` (fallback).

### Step 3: Integration
*   Update `editor.py` to call `TimelineExporter` at the end of a run if `--export` is set.

### Step 4: Validation
*   Create unit tests verifying generated OTIO files against the schema.
*   Manual test: Import generated `.otio` into DaVinci Resolve Free.
