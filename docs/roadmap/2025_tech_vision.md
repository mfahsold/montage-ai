# Montage AI - 2025 Tech Vision

> "Polish, don't generate" - Our guiding principle remains unchanged.

## Executive Summary

Based on 2025 market trends (CapCut, DaVinci Resolve 19, Descript, Opus Clip), we've identified three **high-impact feature pillars** that align perfectly with our core philosophy and existing architecture.

| Feature | Market Trend | Montage AI Fit | Infrastructure Status |
|---------|--------------|----------------|----------------------|
| **Text-Based Editing** | Must-Have | Perfect | Ready (transcriber.py) |
| **Shorts/Viral Clips** | Exploding | Strong | Partial (needs Smart Reframe) |
| **Voice Isolation** | Rising | Perfect for "Polish" | Ready (cgpu pipeline) |

---

## Strategic Pillars 2025

### 1. Text-Based Editing (Priority: HIGH)

**The Trend**: Every major player (Descript, Premiere, Resolve, CapCut) now offers "edit by transcript". Users delete text, the corresponding video gets removed.

**Why Us**: We can do this **locally and privately** - no cloud upload required.

**Current Status**:
- `transcriber.py` already uses Whisper (via cgpu) for transcription
- Word-level timestamps available from Whisper
- Transcripts stored as JSON sidecar files

**Implementation Path**:
```
Phase 1: "Rough Cut by Text" CLI mode
  └─ montage-ai.sh text-edit video.mp4
  └─ Opens transcript in $EDITOR
  └─ User deletes/keeps sentences
  └─ Generates cut list from remaining text

Phase 2: Web UI Integration
  └─ Interactive transcript view
  └─ Click-to-mark segments
  └─ Real-time preview of changes

Phase 3: Smart Suggestions
  └─ Auto-detect filler words ("um", "uh")
  └─ Highlight low-energy segments
  └─ Suggest cuts based on audio analysis
```

**Files to Create/Modify**:
- `src/montage_ai/text_editor.py` (NEW) - Text-to-timeline conversion
- `src/montage_ai/web_ui/app.py` - Add transcript editing endpoint
- `src/montage_ai/transcriber.py` - Add word-level timestamp export

---

### 2. Shorts/Viral Clip Workflow (Priority: HIGH)

**The Trend**: Tools like Opus Clip, Munch, and Vidyo.ai are exploding. They take long-form content (podcasts, interviews) and extract "viral" short clips.

**Why Us**: We offer privacy + creative control. No cloud upload, no subscription.

**Sub-Features**:

#### 2.1 Smart Reframing (9:16 Crop)

**Status**: NOT YET IMPLEMENTED - Critical gap!

**Requirements**:
- Face/speaker detection
- Center-weighted crop for talking heads
- Motion tracking for action scenes

**Implementation Options**:
- **Option A**: FFmpeg + OpenCV (local, CPU-intensive)
- **Option B**: cgpu job with MediaPipe/YOLO (fast, GPU)
- **Option C**: Hybrid (detect locally, render on cgpu)

**Files to Create**:
- `src/montage_ai/cgpu_jobs/smart_reframe.py` (NEW)
- `src/montage_ai/reframer.py` (NEW) - High-level API

#### 2.2 Burn-in Captions

**Status**: EASY TO IMPLEMENT - FFmpeg can do this

**Implementation**:
```python
# FFmpeg drawtext filter with word-by-word highlighting
ffmpeg -i input.mp4 -vf "drawtext=text='Hello':fontcolor=yellow:fontsize=48:x=(w-text_w)/2:y=h-100" output.mp4
```

**Files to Modify**:
- `src/montage_ai/render_helpers.py` - Add `burn_captions()` function
- `src/montage_ai/styles/` - Add caption style presets (TikTok, YouTube, etc.)

#### 2.3 Highlight Detection

**Status**: PARTIAL - We have energy analysis, need "hook" detection

**Current**: `clip_selector.py` finds "good scenes" for montages
**Needed**: Algorithm for "viral moments" based on:
- Audio peaks (laughter, applause, emphasis)
- Face expressions (surprise, excitement)
- Text content (quotable phrases)

**Files to Create/Modify**:
- `src/montage_ai/highlight_detector.py` (NEW)
- `src/montage_ai/audio_analysis.py` - Add `detect_highlights()`

#### 2.4 New CLI Mode

```bash
# Generate 10 vertical shorts from a podcast
montage-ai.sh shorts podcast.mp4 --count 10 --duration 60

# With burn-in captions
montage-ai.sh shorts interview.mp4 --captions --style tiktok
```

---

### 3. Voice Isolation / Audio Polish (Priority: MEDIUM)

**The Trend**: DaVinci 19 and CapCut offer "one-click voice isolation". Bad audio ruins videos.

**Why Us**: Open-source models (demucs, DeepFilterNet) can run on cgpu.

**Current Status**:
- We do loudness normalization
- cgpu infrastructure ready for new models
- Feature flag `VOICE_ISOLATION=true` (to be added)

**Implementation Path**:
```
Phase 1: cgpu Voice Isolation Job
  └─ Add VoiceIsolationJob to cgpu_jobs/
  └─ Use demucs or DeepFilterNet
  └─ Return clean audio track

Phase 2: Pipeline Integration
  └─ Auto-detect noisy audio (SNR analysis)
  └─ Offer cleanup during ingest phase
  └─ Preserve original as fallback

Phase 3: Advanced Audio Polish
  └─ De-reverb
  └─ Compression/limiting
  └─ EQ presets (broadcast, podcast, music)
```

**Files to Create**:
- `src/montage_ai/cgpu_jobs/voice_isolation.py` (NEW)
- `src/montage_ai/audio_polish.py` (NEW) - High-level API

---

## Existing Vision Items (Infrastructure Ready)

These features already have infrastructure in place from Phase 1-4 development:

### 4. AI Colorist (LUT Generator)

**Status**: Skeleton implemented (`cgpu_jobs/lut_generator.py`)

**Feature Flag**: `AI_LUT_GENERATION=true`

**Concept**: Generate .cube LUT files from reference images or style prompts for consistent color grading.

### 5. Motion-Aware Interpolation

**Status**: Skeleton implemented (`cgpu_jobs/interpolation.py`)

**Feature Flag**: `FRAME_INTERPOLATION=true`

**Concept**: RIFE/FILM-based frame interpolation for smooth slow-motion effects.

### 6. Episodic Memory

**Status**: Infrastructure ready (`core/analysis_cache.py`)

**Feature Flag**: `EPISODIC_MEMORY=true`

**Concept**: Track clip usage across montages to ensure variety and learn from user feedback.

**Available Methods**:
- `save_episodic_memory(entry)` - Record clip usage
- `load_episodic_for_clip(path)` - Get usage history
- `get_clip_reuse_count(path, phase)` - Count reuse per story phase

### 7. Embedding Similarity Search

**Status**: Implemented (`core/embedding_search.py`)

**Concept**: "Find clips like this" functionality using cached semantic embeddings.

**Available Methods**:
- `find_similar(embedding, k=5)` - Find similar by vector
- `find_similar_to_clip(path, k=5)` - Find similar to existing clip
- `search_by_query(text, k=5)` - Natural language search

### 8. Agentic Creative Loop

**Status**: Implemented (`creative_evaluator.py`)

**Feature Flag**: `CREATIVE_LOOP=true`

**Concept**: LLM evaluates generated montage and suggests refinements iteratively.

---

## Implementation Priority Matrix

| Feature | User Impact | Technical Effort | Dependencies | Priority |
|---------|-------------|------------------|--------------|----------|
| Text-Based Editing | HIGH | MEDIUM | transcriber.py (ready) | P0 |
| Burn-in Captions | HIGH | LOW | FFmpeg (ready) | P0 |
| Smart Reframing | HIGH | HIGH | Face detection (new) | P1 |
| Voice Isolation | MEDIUM | MEDIUM | cgpu + model (new) | P1 |
| Highlight Detection | MEDIUM | MEDIUM | audio_analysis.py (ready) | P2 |
| AI Colorist | LOW | MEDIUM | Skeleton ready | P3 |
| Frame Interpolation | LOW | HIGH | Skeleton ready | P3 |

---

## Technical Infrastructure Summary

### Already Built (This Session)

| Component | File | Status |
|-----------|------|--------|
| Job Phase Tracking | `web_ui/models.py` | Implemented |
| InterpolationJob | `cgpu_jobs/interpolation.py` | Skeleton |
| LUTGeneratorJob | `cgpu_jobs/lut_generator.py` | Skeleton |
| EpisodicMemoryEntry | `core/analysis_cache.py` | Implemented |
| Story Phase Fields | `core/analysis_cache.py` | Implemented |
| Embedding Search | `core/embedding_search.py` | Implemented |
| Feature Flags | `config.py` | Implemented |
| Low Memory Mode | `config.py` | Implemented |

### To Build (2025 Roadmap)

| Component | Estimated Effort | Blocking For |
|-----------|------------------|--------------|
| `text_editor.py` | 2-3 days | Text-Based Editing |
| `cgpu_jobs/smart_reframe.py` | 1 week | Shorts Workflow |
| `cgpu_jobs/voice_isolation.py` | 2-3 days | Audio Polish |
| `highlight_detector.py` | 1 week | Shorts Workflow |
| `burn_captions()` in render_helpers | 1 day | Shorts Workflow |

---

## Out of Scope (2025)

These trends exist but don't fit our philosophy:

| Feature | Why Not |
|---------|---------|
| **Generative Fill** | "Generate pixels" - against our philosophy |
| **AI Avatars** | "Fake content" - not polish |
| **Real-time Compositing** | Not our domain (use After Effects) |
| **Social Hosting** | We're a tool, not a platform |

---

## Success Metrics

By end of 2025, Montage AI should support:

- [ ] `montage-ai.sh text-edit` - Edit video via transcript
- [ ] `montage-ai.sh shorts` - Generate vertical clips with captions
- [ ] `VOICE_ISOLATION=true` - One-click audio cleanup
- [ ] `SMART_REFRAME=true` - Auto 9:16 cropping for talking heads
- [ ] All features work on low-resource hardware (LOW_MEMORY_MODE)

---

## References

- [Descript](https://descript.com/) - Text-based editing leader
- [Opus Clip](https://opus.pro/) - Viral clip extraction
- [demucs](https://github.com/facebookresearch/demucs) - Voice isolation model
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Noise suppression
- [CapCut](https://www.capcut.com/) - Mobile-first editing with AI
- [DaVinci Resolve 19](https://www.blackmagicdesign.com/products/davinciresolve) - Professional AI features
