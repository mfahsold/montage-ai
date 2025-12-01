# Montage AI - Footage Manager Module
# Implementation of Professional Editing Concepts
# 
# Based on: PROFESSIONAL_EDITING_CONCEPTS.md
# Status: Ready for Integration
# Version: 0.1.0

"""
Footage Management System for Montage AI

Implements professional editing workflows:
- Footage consumed once
- Story Arc-aware clip selection
- Variety scoring and optimization
- Continuity rules for smooth transitions

Integration: Import in smart_worker.py and replace scene selection logic.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import numpy as np


class UsageStatus(Enum):
    """Footage usage status."""
    UNUSED = "unused"
    USED = "used"
    RESERVED = "reserved"  # For planned future use


class SceneType(Enum):
    """Scene categorization based on content."""
    ESTABLISHING = "establishing"  # Wide shots, environment
    ACTION = "action"              # Movement, energy
    DETAIL = "detail"              # Close-ups, inserts
    PORTRAIT = "portrait"          # People focus
    SCENIC = "scenic"              # Nature, atmosphere
    TRANSITION = "transition"      # B-roll, cutaways


class StoryPhase(Enum):
    """Story arc phases with their characteristics."""
    INTRO = "intro"
    BUILD = "build"
    CLIMAX = "climax"
    SUSTAIN = "sustain"
    OUTRO = "outro"


@dataclass
class FootageClip:
    """
    Represents a single footage segment with metadata.
    
    Professional Concept: Each clip is a discrete unit that can be
    "consumed" once and then removed from the available pool.
    """
    
    # Identification
    clip_id: str
    source_file: str
    
    # Timing
    in_point: float              # Start time in seconds
    out_point: float             # End time in seconds
    duration: float              # Duration in seconds
    
    # Classification
    scene_type: SceneType = SceneType.ACTION
    energy_level: float = 0.5   # 0.0 - 1.0
    
    # Usage tracking
    usage_status: UsageStatus = UsageStatus.UNUSED
    usage_count: int = 0
    
    # Quality scoring
    quality_score: float = 0.5  # 0.0 - 1.0
    visual_interest: float = 0.5
    
    # Analysis metadata (optional)
    dominant_colors: List[Tuple] = field(default_factory=list)
    motion_intensity: float = 0.5
    brightness: float = 0.5
    
    # AI-Metadaten (from existing system)
    ai_description: str = ""
    ai_action_level: str = "medium"
    ai_shot_type: str = "medium"
    
    @property
    def is_available(self) -> bool:
        """Check if clip is available for use."""
        return self.usage_status == UsageStatus.UNUSED


@dataclass
class StoryArcPhaseConfig:
    """Configuration for a story arc phase."""
    position_range: Tuple[float, float]  # (start%, end%)
    energy_range: Tuple[float, float]    # (min, max)
    cut_rate: str                        # slow/medium/fast
    preferred_types: List[SceneType]
    description: str


class StoryArcController:
    """
    Controls narrative structure based on professional story arc principles.
    
    Maps timeline position to editing requirements:
    - Intro: Establishing, calm
    - Build: Rising energy
    - Climax: Peak intensity
    - Sustain: Maintain interest
    - Outro: Resolution
    """
    
    PHASES = {
        StoryPhase.INTRO: StoryArcPhaseConfig(
            position_range=(0.0, 0.15),
            energy_range=(0.2, 0.5),
            cut_rate="slow",
            preferred_types=[SceneType.SCENIC, SceneType.ESTABLISHING, SceneType.PORTRAIT],
            description="Introduction, establish context"
        ),
        StoryPhase.BUILD: StoryArcPhaseConfig(
            position_range=(0.15, 0.40),
            energy_range=(0.4, 0.7),
            cut_rate="medium",
            preferred_types=[SceneType.ACTION, SceneType.DETAIL, SceneType.PORTRAIT],
            description="Build tension"
        ),
        StoryPhase.CLIMAX: StoryArcPhaseConfig(
            position_range=(0.40, 0.70),
            energy_range=(0.6, 1.0),
            cut_rate="fast",
            preferred_types=[SceneType.ACTION, SceneType.DETAIL],
            description="Climax, peak intensity"
        ),
        StoryPhase.SUSTAIN: StoryArcPhaseConfig(
            position_range=(0.70, 0.90),
            energy_range=(0.5, 0.8),
            cut_rate="medium",
            preferred_types=[SceneType.ACTION, SceneType.SCENIC, SceneType.PORTRAIT],
            description="Sustain tension, add variety"
        ),
        StoryPhase.OUTRO: StoryArcPhaseConfig(
            position_range=(0.90, 1.0),
            energy_range=(0.2, 0.5),
            cut_rate="slow",
            preferred_types=[SceneType.SCENIC, SceneType.PORTRAIT, SceneType.ESTABLISHING],
            description="Outro, resolution"
        )
    }
    
    def get_current_phase(self, relative_position: float) -> Tuple[StoryPhase, StoryArcPhaseConfig]:
        """
        Determine current phase based on timeline position.
        
        Args:
            relative_position: Position as fraction of total duration (0.0-1.0)
            
        Returns:
            Tuple of (phase enum, phase config)
        """
        for phase, config in self.PHASES.items():
            start, end = config.position_range
            if start <= relative_position < end:
                return phase, config
        return StoryPhase.OUTRO, self.PHASES[StoryPhase.OUTRO]
    
    def get_recommended_duration(self, phase: StoryPhase, bpm: float) -> Tuple[float, float]:
        """
        Calculate recommended clip duration for phase.
        
        Args:
            phase: Current story arc phase
            bpm: Music tempo in BPM
            
        Returns:
            Tuple of (min_duration, max_duration) in seconds
        """
        beat_duration = 60 / bpm
        
        duration_map = {
            "slow": (beat_duration * 4, beat_duration * 8),    # 4-8 beats
            "medium": (beat_duration * 2, beat_duration * 4),  # 2-4 beats
            "fast": (beat_duration * 1, beat_duration * 2),    # 1-2 beats
        }
        
        cut_rate = self.PHASES[phase].cut_rate
        return duration_map.get(cut_rate, duration_map["medium"])


class FootagePoolManager:
    """
    Manages footage pool with consumption tracking.
    
    Core Principle: Footage is "consumed" once used and removed from
    the available pool to ensure variety and prevent repetition.
    
    Professional Workflow:
    1. Analyze and categorize all footage
    2. Select clips based on story arc requirements
    3. Mark used clips as "consumed"
    4. Track variety and utilization metrics
    """
    
    def __init__(self, 
                 clips: List[FootageClip],
                 strict_once: bool = True,
                 verbose: bool = True):
        """
        Initialize footage pool.
        
        Args:
            clips: List of FootageClip objects
            strict_once: If True, each clip can only be used once
            verbose: Print status messages
        """
        self.clips = {c.clip_id: c for c in clips}
        self.used_clips: Set[str] = set()
        self.strict_once = strict_once
        self.verbose = verbose
        
        # Timeline tracking
        self.timeline: List[Dict] = []
        
        # Story arc controller
        self.arc = StoryArcController()
    
    @classmethod
    def from_scenes(cls, scenes: List[Dict], strict_once: bool = True) -> 'FootagePoolManager':
        """
        Create FootagePoolManager from existing scene list.
        
        Compatible with smart_worker.py's all_scenes format:
        {'path': str, 'start': float, 'end': float, 'duration': float, 
         'usage_count': int, 'meta': {'action': str, 'shot': str, ...}}
        
        Args:
            scenes: List of scene dictionaries from smart_worker.py
            strict_once: If True, clips can only be used once
            
        Returns:
            Configured FootagePoolManager instance
        """
        clips = []
        
        for i, scene in enumerate(scenes):
            # Map action level to energy
            action_map = {"low": 0.3, "medium": 0.5, "high": 0.8}
            meta = scene.get('meta', {})
            action = meta.get('action', 'medium').lower()
            energy = action_map.get(action, 0.5)
            
            # Map shot type to scene type
            shot = meta.get('shot', 'medium').lower()
            if shot == 'wide':
                scene_type = SceneType.ESTABLISHING
            elif shot == 'close-up':
                scene_type = SceneType.DETAIL
            else:
                scene_type = SceneType.ACTION
            
            clip = FootageClip(
                clip_id=id(scene),  # Use Python object ID for matching
                source_file=scene['path'],
                in_point=scene['start'],
                out_point=scene['end'],
                duration=scene['duration'],
                scene_type=scene_type,
                energy_level=energy,
                ai_description=meta.get('description', ''),
                ai_action_level=action,
                ai_shot_type=shot,
                usage_count=scene.get('usage_count', 0),
                usage_status=UsageStatus.USED if scene.get('usage_count', 0) > 0 else UsageStatus.UNUSED
            )
            clips.append(clip)
        
        return cls(clips, strict_once=strict_once)
    
    def get_available_clips(self,
                            min_duration: float = 0.5,
                            scene_type: Optional[SceneType] = None,
                            energy_range: Optional[Tuple[float, float]] = None,
                            exclude_source: Optional[str] = None) -> List[FootageClip]:
        """
        Get available (unused) clips matching criteria.
        
        Args:
            min_duration: Minimum clip duration in seconds
            scene_type: Filter by scene type (optional)
            energy_range: Filter by energy (min, max) (optional)
            exclude_source: Exclude clips from this source file
            
        Returns:
            List of available FootageClip objects
        """
        available = []
        
        for clip in self.clips.values():
            # Skip used clips in strict mode
            if self.strict_once and clip.clip_id in self.used_clips:
                continue
            
            # Duration filter
            if clip.duration < min_duration:
                continue
            
            # Scene type filter
            if scene_type and clip.scene_type != scene_type:
                continue
            
            # Energy range filter
            if energy_range:
                min_e, max_e = energy_range
                if not (min_e <= clip.energy_level <= max_e):
                    continue
            
            # Source exclusion (avoid jump cuts)
            if exclude_source and clip.source_file == exclude_source:
                continue
            
            available.append(clip)
        
        return available
    
    def consume_clip(self, 
                     clip_id: str, 
                     timeline_position: float,
                     used_in_point: float,
                     used_out_point: float) -> bool:
        """
        Mark a clip as consumed and log to timeline.
        
        Professional Principle: Once footage is used, it's removed
        from the available pool to ensure variety.
        
        Args:
            clip_id: ID of clip to consume
            timeline_position: Position in final timeline
            used_in_point: Actual in-point used (may differ from clip.in_point)
            used_out_point: Actual out-point used
            
        Returns:
            True if consumption successful
        """
        if clip_id not in self.clips:
            return False
        
        clip = self.clips[clip_id]
        
        # Update clip status
        clip.usage_count += 1
        clip.usage_status = UsageStatus.USED
        
        # Add to used set
        self.used_clips.add(clip_id)
        
        # Log to timeline (EDL-style)
        self.timeline.append({
            'clip_id': clip_id,
            'source_file': clip.source_file,
            'source_in': used_in_point,
            'source_out': used_out_point,
            'timeline_position': timeline_position,
            'duration': used_out_point - used_in_point
        })
        
        if self.verbose:
            remaining = len(self.clips) - len(self.used_clips)
            print(f"   📼 Consumed clip {clip_id[:20]}... ({remaining} remaining)")
        
        return True
    
    def select_for_position(self,
                            timeline_position: float,
                            total_duration: float,
                            target_clip_duration: float,
                            last_source: Optional[str] = None) -> Optional[FootageClip]:
        """
        Select optimal clip for timeline position based on story arc.
        
        This is the core selection logic that implements professional
        editing principles:
        1. Determine current story phase
        2. Get energy requirements for phase
        3. Filter available clips
        4. Score and rank candidates
        5. Select best match
        
        Args:
            timeline_position: Current position in timeline (seconds)
            total_duration: Total target duration
            target_clip_duration: How long the clip should be
            last_source: Last used source file (for variety)
            
        Returns:
            Selected FootageClip or None if no suitable clip available
        """
        # 1. Determine phase
        relative_pos = timeline_position / total_duration
        phase, phase_config = self.arc.get_current_phase(relative_pos)
        
        # 2. Get available clips
        available = self.get_available_clips(
            min_duration=target_clip_duration,
            energy_range=phase_config.energy_range,
            exclude_source=last_source
        )
        
        # Fallback: Relax energy constraint if no clips match
        if not available:
            available = self.get_available_clips(
                min_duration=target_clip_duration,
                exclude_source=last_source
            )
        
        # Further fallback: Allow same source
        if not available:
            available = self.get_available_clips(min_duration=target_clip_duration)
        
        if not available:
            return None
        
        # 3. Score candidates
        def score_clip(clip: FootageClip) -> float:
            score = 0.0
            
            # Scene type match
            if clip.scene_type in phase_config.preferred_types:
                score += 30
            
            # Energy match
            min_e, max_e = phase_config.energy_range
            target_e = (min_e + max_e) / 2
            energy_diff = abs(clip.energy_level - target_e)
            score += (1 - energy_diff) * 25
            
            # Quality bonus
            score += clip.quality_score * 15
            
            # Visual interest
            score += clip.visual_interest * 15
            
            # Variety bonus (different source)
            if last_source and clip.source_file != last_source:
                score += 15
            
            return score
        
        # 4. Sort by score and select best
        available.sort(key=score_clip, reverse=True)
        
        if self.verbose:
            print(f"   🎬 Phase: {phase.value} | Energy: {phase_config.energy_range}")
            print(f"   📊 Candidates: {len(available)} clips available")
        
        return available[0]
    
    def get_stats(self) -> Dict:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with utilization metrics
        """
        total = len(self.clips)
        used = len(self.used_clips)
        
        return {
            'total_clips': total,
            'used_clips': used,
            'remaining_clips': total - used,
            'utilization_rate': used / total if total > 0 else 0.0,
            'variety_score': len(set(t['source_file'] for t in self.timeline)) / max(1, len(self.timeline)),
            'timeline_entries': len(self.timeline)
        }
    
    def export_edl(self, output_path: str):
        """
        Export timeline as Edit Decision List (JSON format).
        
        Professional format compatible with standard EDL workflows.
        
        Args:
            output_path: Path to write EDL JSON file
        """
        import json
        
        edl = {
            'version': '1.0',
            'generator': 'Montage AI',
            'stats': self.get_stats(),
            'timeline': self.timeline,
            'used_clips': list(self.used_clips)
        }
        
        with open(output_path, 'w') as f:
            json.dump(edl, f, indent=2)
        
        if self.verbose:
            print(f"   📝 EDL exported to {output_path}")


# =============================================================================
# INTEGRATION HELPER: Drop-in replacement for smart_worker.py
# =============================================================================

def integrate_footage_manager(all_scenes: List[Dict], 
                               strict_once: bool = True) -> FootagePoolManager:
    """
    Create FootagePoolManager from smart_worker.py's all_scenes list.
    
    Usage in smart_worker.py:
    
    ```python
    from footage_manager import integrate_footage_manager, select_next_clip
    
    # After scene detection
    pool = integrate_footage_manager(all_scenes, strict_once=True)
    
    # In the assembly loop, replace scene selection with:
    selected = select_next_clip(
        pool, current_time, target_duration, cut_duration, last_used_path
    )
    if selected:
        # Use selected.source_file, selected.in_point, etc.
        pool.consume_clip(selected.clip_id, current_time, ...)
    ```
    
    Args:
        all_scenes: List from smart_worker.py scene detection
        strict_once: If True, each clip used only once
        
    Returns:
        Configured FootagePoolManager
    """
    return FootagePoolManager.from_scenes(all_scenes, strict_once=strict_once)


def select_next_clip(pool: FootagePoolManager,
                     current_time: float,
                     total_duration: float,
                     clip_duration: float,
                     last_source: Optional[str] = None) -> Optional[FootageClip]:
    """
    Convenience wrapper for clip selection.
    
    Drop-in replacement for smart_worker.py's scene selection logic.
    
    Args:
        pool: FootagePoolManager instance
        current_time: Current timeline position
        total_duration: Total target duration
        clip_duration: How long the clip should be
        last_source: Last used source file
        
    Returns:
        Selected FootageClip or None
    """
    return pool.select_for_position(
        timeline_position=current_time,
        total_duration=total_duration,
        target_clip_duration=clip_duration,
        last_source=last_source
    )


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    # Demo with mock scenes
    mock_scenes = [
        {'path': 'video1.mp4', 'start': 0.0, 'end': 10.0, 'duration': 10.0, 
         'usage_count': 0, 'meta': {'action': 'high', 'shot': 'wide'}},
        {'path': 'video1.mp4', 'start': 10.0, 'end': 20.0, 'duration': 10.0,
         'usage_count': 0, 'meta': {'action': 'medium', 'shot': 'medium'}},
        {'path': 'video2.mp4', 'start': 0.0, 'end': 15.0, 'duration': 15.0,
         'usage_count': 0, 'meta': {'action': 'low', 'shot': 'close-up'}},
        {'path': 'video2.mp4', 'start': 15.0, 'end': 30.0, 'duration': 15.0,
         'usage_count': 0, 'meta': {'action': 'high', 'shot': 'wide'}},
        {'path': 'video3.mp4', 'start': 0.0, 'end': 25.0, 'duration': 25.0,
         'usage_count': 0, 'meta': {'action': 'medium', 'shot': 'medium'}},
    ]
    
    print("=" * 60)
    print("🎬 Footage Manager Demo")
    print("=" * 60)
    
    # Create pool
    pool = integrate_footage_manager(mock_scenes, strict_once=True)
    
    # Simulate timeline assembly
    total_duration = 60.0  # 60 second video
    current_time = 0.0
    last_source = None
    
    print(f"\n📊 Building {total_duration}s timeline...")
    print("-" * 40)
    
    while current_time < total_duration:
        clip_duration = 3.0  # 3 second clips
        
        clip = select_next_clip(pool, current_time, total_duration, clip_duration, last_source)
        
        if not clip:
            print(f"⚠️ No more footage at {current_time:.1f}s")
            break
        
        # Consume the clip
        pool.consume_clip(
            clip.clip_id,
            current_time,
            clip.in_point,
            clip.in_point + clip_duration
        )
        
        last_source = clip.source_file
        current_time += clip_duration
    
    # Show stats
    print("-" * 40)
    stats = pool.get_stats()
    print(f"\n📈 Final Statistics:")
    print(f"   Total clips:      {stats['total_clips']}")
    print(f"   Used clips:       {stats['used_clips']}")
    print(f"   Remaining:        {stats['remaining_clips']}")
    print(f"   Utilization:      {stats['utilization_rate']*100:.1f}%")
    print(f"   Variety score:    {stats['variety_score']*100:.1f}%")
