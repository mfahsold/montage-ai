# Storytelling Engine Implementation Plan

> **Goal:** Move beyond random clip shuffling to intentional, narrative-driven editing.

## 1. Core Concepts

### The "Tension" Variable
We introduce a normalized variable `tension` (0.0 to 1.0) that drives all editing decisions.

| Tension Level | Editing Style | Shot Selection | Audio |
| :--- | :--- | :--- | :--- |
| **Low (0.0 - 0.3)** | Slow cuts, long takes, dissolves | Wide shots, landscapes, static | Ambient, soft music |
| **Medium (0.3 - 0.7)** | On-beat cuts, moderate pace | Medium shots, character interaction | Rhythmic, driving |
| **High (0.7 - 1.0)** | Rapid fire, jump cuts, hard cuts | Close-ups, shaky cam, motion | Intense, loud, staccato |

### The Story Arc
A montage is defined by a `StoryArc` curve over time.

```python
class StoryArc:
    def get_target_tension(self, progress: float) -> float:
        """Returns target tension (0-1) for a given progress (0-1) in the timeline."""
        pass
```

**Standard Arcs:**
*   **Hero's Journey:** Low -> Build (Med) -> Climax (High) -> Resolution (Low)
*   **Constant Energy (MTV):** High -> High -> High
*   **Slow Burn (Hitchcock):** Low -> Low -> Build -> Extreme High -> Cut to Black

## 2. Scope and Considerations

### Product Intent (Non-Negotiables)
*   **Montage AI is an AI rough-cut tool**, not a full NLE.
*   We support **two primary workflows**:
    1) **Music-driven montage** (B-roll only, tension from audio).
    2) **Dialogue-driven storytelling** (A-roll audio + B-roll visuals).
*   Output must be **professionally editable** (OTIO/EDL/XML export).
*   Must behave **deterministically** given the same seed and inputs.
*   Must degrade gracefully with **limited footage** (no hard failure).

### Practical Constraints
*   Analysis must be **cacheable** and **parallelizable** (cluster + cgpu).
*   Story arc logic cannot add excessive latency to the pipeline.
*   Mixed-quality inputs must not destabilize the solver.

## 3. Knifflige technische Fragestellungen (mit Plan)

### A) How do we compute tension reliably across video + audio?
**Plan**: Normalize and smooth features per clip, then combine with weights and track-level calibration.

```python
class TensionAnalyzer:
    def analyze_clip(self, clip):
        motion = optical_flow_magnitude(clip)
        shake = camera_shake_score(clip)
        complexity = edge_density(clip)
        # Normalize per-run using robust percentiles
        return normalize(0.5 * motion + 0.3 * shake + 0.2 * complexity)

    def analyze_audio(self, audio, t0, t1):
        loudness = rms(audio, t0, t1)
        flux = spectral_flux(audio, t0, t1)
        tempo = tempo_at(audio, t0, t1)
        return normalize(loudness * flux * tempo)

    def fuse(self, clip_tension, audio_tension, alpha=0.6):
        # alpha biases audio for music-driven cuts
        return clamp(alpha * audio_tension + (1 - alpha) * clip_tension)
```

### B) How do we map a story arc to actual cuts (beats/timestamps)?
**Plan**: Build a cost function and solve with a greedy + local repair or DP-lite.

```python
def plan_cuts(beats, story_arc, clips):
    plan = []
    last_clip = None
    for t in beats:
        target = story_arc.get_target_tension(t / beats[-1])
        best = None
        best_score = -1e9
        for clip in clips:
            if clip.used: continue
            tension_match = 1 - abs(clip.tension - target)
            novelty_penalty = 0.2 if clip.source == last_clip else 0.0
            score = tension_match - novelty_penalty
            if score > best_score:
                best, best_score = clip, score
        plan.append((t, best))
        best.used = True
        last_clip = best.source
    return plan
```

### C) When do we cut away from A-roll to B-roll?
**Plan**: Use a combined trigger: semantic relevance OR boredom threshold.

```python
def decide_visual(a_segment, b_pool, script):
    topic = script.topic_at(a_segment.start)
    b_match = best_broll_for(topic, b_pool)
    boredom = static_head_score(a_segment.video)

    if b_match.score > 0.7 or boredom > 0.6:
        return b_match.clip  # L-cut / J-cut candidate
    return a_segment.video
```

### D) How do we avoid failure with limited footage?
**Plan**: Relax constraints progressively and allow re-use with decay.

```python
def pick_clip(candidates, target, allow_reuse=False):
    for clip in candidates:
        if not allow_reuse and clip.used:
            continue
        if abs(clip.tension - target) < 0.3:
            return clip
    return candidates[0]  # fallback
```

## 4. Technical Implementation (Pseudocode)

### A. Tension Analysis
We need to score raw clips and music to match them to the arc.

```python
def analyze_clip_tension(video_clip):
    """
    Estimates the inherent 'energy' or 'tension' of a raw clip.
    """
    # 1. Motion Analysis (Optical Flow)
    motion_score = calculate_optical_flow_magnitude(video_clip)
    
    # 2. Visual Complexity (Edge Density)
    complexity_score = calculate_edge_density(video_clip)
    
    # 3. Camera Movement
    camera_shake = detect_camera_shake(video_clip)
    
    # Weighted sum
    raw_tension = (0.5 * motion_score) + (0.3 * camera_shake) + (0.2 * complexity_score)
    return normalize(raw_tension)

def analyze_music_tension(audio_segment):
    """
    Estimates tension from audio features.
    """
    loudness = get_rms_amplitude(audio_segment)
    spectral_flux = get_spectral_flux(audio_segment) # Rate of change
    tempo = get_instantaneous_tempo(audio_segment)
    
    return normalize(loudness * spectral_flux * tempo)
```

### B. The Solver: Matching Arc to Clips
Instead of a linear pass, we treat this as a constraint satisfaction problem or a greedy optimization.

```python
def generate_montage(story_arc, music_track, available_clips):
    timeline = []
    current_time = 0
    beats = detect_beats(music_track)
    
    for beat in beats:
        # 1. Determine context
        progress = current_time / music_track.duration
        target_tension = story_arc.get_target_tension(progress)
        music_intensity = analyze_music_tension(music_track, current_time)
        
        # 2. Determine cut duration
        # High tension = cut on every beat (or half-beat)
        # Low tension = hold for 4-8 beats
        beats_to_hold = map_tension_to_duration(target_tension)
        
        if time_since_last_cut < beats_to_hold:
            continue # Hold the shot
            
        # 3. Select the best clip
        best_clip = None
        best_score = -infinity
        
        for clip in available_clips:
            if clip.is_used: continue
            
            # Score based on how well clip tension matches target tension
            tension_match = 1 - abs(clip.tension - target_tension)
            
            # Semantic relevance (if prompt provided)
            semantic_score = clip.semantic_match(current_segment_keywords)
            
            total_score = tension_match + semantic_score
            
            if total_score > best_score:
                best_score = total_score
                best_clip = clip
                
        timeline.append(Cut(time=current_time, clip=best_clip))
        current_time = beat.time
        
    return timeline
```

### C. Hybrid Workflow: Dialogue (A-Roll) + Montage (B-Roll)
The most complex scenario: We have a main interview/story (A-Roll) and want to cover cuts or illustrate points with B-Roll.

```python
def interleave_dialogue_and_broll(a_roll_track, b_roll_pool, script_analysis):
    """
    a_roll_track: The main audio/video (e.g., interview)
    b_roll_pool: Collection of illustrative clips
    script_analysis: Timestamps of topics spoken in A-Roll
    """
    final_sequence = []
    
    for segment in a_roll_track.segments:
        # Default: Show A-Roll speaker
        current_visual = segment.video 
        
        # Check for B-Roll opportunities
        topic = script_analysis.get_topic_at(segment.start_time)
        
        # 1. Semantic Match: Does the speaker mention something we have footage of?
        matching_broll = find_best_broll(topic, b_roll_pool)
        
        # 2. Visual Interest: Has the speaker been on screen too long? (e.g. > 10s)
        time_on_camera = segment.start_time - last_cut_time
        boring_factor = calculate_static_head_score(segment.video)
        
        if matching_broll and (matching_broll.score > threshold or time_on_camera > 10):
            # INSERT B-ROLL (L-Cut or J-Cut potential)
            # Audio remains A-Roll, Video switches to B-Roll
            current_visual = matching_broll
            
        final_sequence.append(CompositeClip(audio=segment.audio, video=current_visual))
        
    return final_sequence
```

## 5. Data Contracts (Draft)

```json
{
  "id": "hero_journey_v1",
  "type": "curve",
  "points": [
    {"t": 0.00, "tension": 0.20},
    {"t": 0.35, "tension": 0.45},
    {"t": 0.70, "tension": 0.90},
    {"t": 0.90, "tension": 0.60},
    {"t": 1.00, "tension": 0.25}
  ],
  "smoothing": "cubic",
  "min_hold_beats": 2,
  "max_hold_beats": 8
}
```

## 6. Next Steps

1.  **Implement `TensionAnalyzer` class**:
    *   Wrap `librosa` for audio tension.
    *   Wrap `opencv` (optical flow) for video tension.
2.  **Define `StoryArc` JSON schema**: Allow users to define custom curves.
3.  **Prototype the `Solver`**: Write a simple script that takes pre-scored dummy clips and arranges them according to a curve.
