# Storytelling Engine - Implementation Plan (Phase 1)

> **Constraint:** No heavy compute locally. All pixel/audio analysis must be offloaded to `cgpu` or read from pre-computed metadata.

## 1. Module Scaffolding & Feature Flags

We will create a new package `src/montage_ai/storytelling/` to encapsulate the logic.

### A. Feature Flag & Configuration
**File:** `src/montage_ai/config.py` (Update)
```python
import os

# Toggle for the new engine
ENABLE_STORY_ENGINE = os.getenv("ENABLE_STORY_ENGINE", "false").lower() == "true"

# If true, we strictly forbid local heavy analysis
STRICT_CLOUD_COMPUTE = True 
```

### B. The Story Arc Definition
**File:** `src/montage_ai/storytelling/story_arc.py`
Defines the "shape" of the video.

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class StoryArc:
    # List of (time_progress_0_to_1, tension_0_to_1)
    # e.g. [(0.0, 0.1), (0.5, 0.4), (0.9, 1.0), (1.0, 0.0)]
    curve_points: List[Tuple[float, float]]

    def get_target_tension(self, progress: float) -> float:
        """Interpolates the curve to find target tension at a specific progress point."""
        # Linear interpolation logic here
        pass

    @classmethod
    def from_preset(cls, name: str):
        """Loads standard arcs like 'hero_journey', 'mtv_energy', 'slow_burn'"""
        pass
```

### C. Tension Provider (The "Lightweight" Client)
**File:** `src/montage_ai/storytelling/tension_provider.py`
**Crucial:** This module does *not* compute. It only reads.

```python
import json
from pathlib import Path

class TensionProvider:
    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self.cache = {}

    def get_tension(self, clip_path: str) -> float:
        """
        Retrieves pre-computed tension score. 
        Raises MissingAnalysisError if not found (triggering a cgpu job elsewhere).
        """
        clip_id = self._get_clip_id(clip_path)
        
        if clip_id in self.cache:
            return self.cache[clip_id]
            
        # Try load from disk (populated by cgpu job)
        meta_file = self.metadata_dir / f"{clip_id}_analysis.json"
        if not meta_file.exists():
            raise MissingAnalysisError(f"Analysis missing for {clip_id}. Run cgpu job first.")
            
        with open(meta_file) as f:
            data = json.load(f)
            
        # Tension is a composite score of motion, complexity, and audio energy
        tension = (data['visual']['motion_score'] * 0.6) + (data['visual']['edge_density'] * 0.4)
        self.cache[clip_id] = tension
        return tension
```

### D. The Solver (The Brain)
**File:** `src/montage_ai/storytelling/story_solver.py`
Matches clips to the arc.

```python
class StorySolver:
    def __init__(self, arc: StoryArc, tension_provider: TensionProvider):
        self.arc = arc
        self.provider = tension_provider

    def solve(self, clips: List[str], duration: float, beats: List[float]) -> List[dict]:
        timeline = []
        used_clips = set()
        
        for beat_time in beats:
            progress = beat_time / duration
            target_tension = self.arc.get_target_tension(progress)
            
            # Find best matching unused clip
            best_clip = None
            min_error = float('inf')
            
            for clip in clips:
                if clip in used_clips: continue
                
                clip_tension = self.provider.get_tension(clip)
                error = abs(clip_tension - target_tension)
                
                if error < min_error:
                    min_error = error
                    best_clip = clip
            
            if best_clip:
                timeline.append({"time": beat_time, "clip": best_clip})
                used_clips.add(best_clip)
                
        return timeline
```

## 2. Integration into Montage Builder

We need to hook this into the main build pipeline, ensuring we trigger the remote analysis first.

**File:** `src/montage_ai/montage_builder.py`

```python
class MontageBuilder:
    def build(self):
        # ... setup ...
        
        if config.ENABLE_STORY_ENGINE:
            self._run_story_pipeline()
        else:
            self._run_legacy_pipeline()
            
    def _run_story_pipeline(self):
        # 1. Identify missing analysis
        missing_clips = self._check_missing_metadata(self.input_clips)
        
        # 2. Dispatch CGPU Job (Batch) if needed
        if missing_clips:
            print(f"Offloading analysis for {len(missing_clips)} clips to Cluster...")
            job_id = cgpu_client.submit_job(
                "analyze_tension_batch", 
                inputs=missing_clips,
                output_dir=self.metadata_dir
            )
            cgpu_client.wait_for(job_id)
            
        # 3. Solve Story
        provider = TensionProvider(self.metadata_dir)
        arc = StoryArc.from_preset(self.style)
        solver = StorySolver(arc, provider)
        
        self.timeline = solver.solve(self.input_clips, self.music_duration, self.beats)
```

## 3. Evaluation Harness

A script to verify if our solver is actually following the arc, without watching hours of video.

**File:** `scripts/evaluate_story_engine.py`

```python
def evaluate_timeline(timeline, arc, provider):
    errors = []
    
    print("Time  | Target | Actual | Clip")
    print("-" * 40)
    
    for event in timeline:
        time = event['time']
        progress = time / total_duration
        
        target = arc.get_target_tension(progress)
        actual = provider.get_tension(event['clip'])
        
        diff = abs(target - actual)
        errors.append(diff)
        
        print(f"{time:5.1f} | {target:0.2f}   | {actual:0.2f}   | {event['clip']}")
        
    mse = sum([e**2 for e in errors]) / len(errors)
    print(f"\nMean Squared Error (Tension): {mse:0.4f}")
    
    if mse > 0.2:
        print("FAIL: Solver is not adhering to the story arc.")
    else:
        print("PASS: Story arc followed successfully.")
```

## Next Immediate Actions

1.  Create the `src/montage_ai/storytelling/` directory.
2.  Implement `story_arc.py` with a simple linear interpolation.
3.  Implement `tension_provider.py` with a dummy mode (random values) for initial testing until the CGPU job is ready.
