"""
Pacing Engine Module

Manages the rhythm and timing of the montage.
Determines cut durations based on:
1. Audio Beats & Tempo
2. Energy Levels (High/Low sections)
3. User Instructions (Fast/Slow pacing)
4. Dynamic Patterns (Pattern Matching)
"""

import numpy as np
import random
from typing import List, Optional, Any, Dict
from ..audio_analysis import calculate_dynamic_cut_length
from .context import MontageContext
from ..utils import coerce_float
from ..logger import logger

class PacingEngine:
    """
    Decides when to cut.
    """
    def __init__(self, context: MontageContext):
        self.ctx = context
        self.settings = context.settings

    def init_audio_duration(self):
        """Initialize audio and calculate target duration."""
        from ..utils import coerce_float
        
        # Get duration settings from config
        target_duration_setting = self.settings.creative.target_duration
        music_start = self.settings.creative.music_start
        music_end = self.settings.creative.music_end

        # Allow overrides from editing instructions
        if self.ctx.creative.editing_instructions:
            if self.ctx.creative.editing_instructions.music_start > 0:
                music_start = self.ctx.creative.editing_instructions.music_start
            
            if self.ctx.creative.editing_instructions.music_end is not None:
                music_end = self.ctx.creative.editing_instructions.music_end

        audio_duration = self.ctx.media.audio_result.duration

        # Apply music trimming
        if music_end and music_end < audio_duration:
            audio_duration = music_end - music_start
        elif music_start > 0:
            audio_duration = audio_duration - music_start

        # Determine target duration
        if target_duration_setting > 0:
            self.ctx.timeline.target_duration = target_duration_setting
        else:
            self.ctx.timeline.target_duration = audio_duration

        # Apply Creative Director constraints if provided (only when env didn't override)
        if target_duration_setting <= 0 and self.ctx.creative.editing_instructions:
            constraints = getattr(self.ctx.creative.editing_instructions, 'constraints', {})
            target_override = coerce_float(constraints.get("target_duration_sec"))
            if target_override and target_override > 0:
                self.ctx.timeline.target_duration = min(audio_duration, target_override)

        if self.settings.features.verbose:
            logger.info(f"   ⏱️ Target duration: {self.ctx.timeline.target_duration:.1f}s")

    def init_crossfade_settings(self):
        """Initialize crossfade settings from config and Creative Director."""
        enable_xfade = self.settings.creative.enable_xfade
        xfade_duration = self.settings.creative.xfade_duration

        if enable_xfade == "true":
            self.ctx.timeline.enable_xfade = True
        elif enable_xfade == "false":
            self.ctx.timeline.enable_xfade = False
        elif self.ctx.creative.editing_instructions is not None:
            transitions = getattr(self.ctx.creative.editing_instructions, 'transitions', {})
            transition_type = transitions.get('type', 'energy_aware')
            xfade_duration = transitions.get('crossfade_duration_sec', xfade_duration)
            if transition_type == 'crossfade':
                self.ctx.timeline.enable_xfade = True

        self.ctx.timeline.xfade_duration = xfade_duration

    def get_next_cut_duration(self, current_energy: float) -> float:
        """
        Calculate the duration of the next cut in seconds.
        Updates internal timeline state (beat_idx, pattern_idx).
        """
        audio_res = self.ctx.media.audio_result
        if not audio_res:
             # Fallback if no audio analysis
             return 4.0
             
        tempo = audio_res.tempo
        beat_times = audio_res.beat_times
        
        # Determine how many beats this shot should last
        beats = self._calculate_beats_per_cut(current_energy, tempo)
        
        # Convert beats to time duration
        duration = self._calculate_cut_duration(beats, beat_times)
        
        return duration

    def get_energy_at_time(self, time_sec: float) -> float:
        """Get energy level at a specific time."""
        audio_res = self.ctx.media.audio_result
        if not audio_res or len(audio_res.energy_times) == 0:
            return 0.5
            
        idx = np.searchsorted(audio_res.energy_times, time_sec)
        idx = min(idx, len(audio_res.energy_values) - 1)
        return float(audio_res.energy_values[idx])

    def _calculate_beats_per_cut(self, current_energy: float, tempo: float) -> float:
        """Calculate beats per cut based on pacing settings."""
        
        # 1. Parse Constraints
        min_beats = None
        max_beats = None
        
        # Handle dict vs Pydantic model for editing_instructions
        instr = self.ctx.creative.editing_instructions
        
        # Compatibility helper (since we have Pydantic now but maybe dict legacy logic here or there)
        # Actually context purification ensures we use Pydantic if converted
        # But instructions might be None
        
        constraints = {}
        pacing_speed = "dynamic"
        
        if instr:
            # If Pydantic model (expected)
            if hasattr(instr, 'extract_constraints'): # If we added methods to model? No.
               # Pydantic model doesn't have .get() unless we implemented it or its a dict
               pass
            elif isinstance(instr, dict):
                # Legacy safety
                constraints = instr.get("constraints", {})
                pacing_speed = instr.get("pacing", {}).get("speed", "dynamic")
            else:
                 # It's a Pydantic model. 
                 # We need to access fields. But models.py didn't show nested constraints yet?
                 # Let's assume for now we use getattr or it is not yet fully typed in structure
                 # The user code had: instr.get("constraints")
                 pass

        # To be safe, let's treat it carefully.
        # Ideally, we should update models.py to have full schema, but for now let's implement safe access.
        
        # Let's fetch cut_patterns first
        cut_patterns = self._get_cut_patterns()

        # Extract constraints override
        # We need to know the structure of EditingInstructions in models.py
        # Based on previous read, it had 'music_track', 'style', 'script', 'broll_plan'
        # It did NOT explicit 'constraints' or 'pacing'. 
        # But it had `class Config: extra = "allow"`. So these might be dynamic fields.
        
        instr_dict = {}
        if instr:
            if hasattr(instr, "model_dump"):
                instr_dict = instr.model_dump()
            elif hasattr(instr, "dict"):
                 instr_dict = instr.dict() # Pydantic v1
            elif isinstance(instr, dict):
                 instr_dict = instr
                 
        constraints = instr_dict.get("constraints", {})
        pacing_speed = instr_dict.get("pacing", {}).get("speed", "dynamic")

        min_clip = coerce_float(constraints.get("min_clip_duration_sec"))
        max_clip = coerce_float(constraints.get("max_clip_duration_sec"))
        beat_duration = 60.0 / tempo if tempo > 0 else 0.5
        
        if min_clip:
            min_beats = min_clip / beat_duration
        if max_clip:
            max_beats = max_clip / beat_duration
        if min_beats is not None and max_beats is not None and min_beats > max_beats:
            min_beats, max_beats = max_beats, min_beats

        def clamp_beats(b: float) -> float:
            if min_beats is not None:
                b = max(b, min_beats)
            if max_beats is not None:
                b = min(b, max_beats)
            return b

        # 2. Section-based pacing override (Pacing Curves)
        current_section = None
        current_time = self.ctx.timeline.current_time
        
        if self.ctx.media.audio_result and self.ctx.media.audio_result.sections:
            for section in self.ctx.media.audio_result.sections:
                if section.start_time <= current_time < section.end_time:
                    current_section = section
                    break
        
        if pacing_speed == "dynamic" and current_section:
            # Structure-aware pacing
            label = getattr(current_section, 'label', None)
            
            if label == 'intro':
                return 16 if tempo > 120 else 8
            elif label == 'outro':
                return 16
            elif label == 'build':
                # Progressive build: 8 -> 4 -> 2 -> 1
                duration = current_section.end_time - current_section.start_time
                if duration > 1.0:
                    progress = (current_time - current_section.start_time) / duration
                    if progress < 0.25: return 8
                    elif progress < 0.50: return 4
                    elif progress < 0.75: return 2
                    else: return 1
                return 4
            elif label == 'drop':
                return 1 if tempo < 130 else 2

            # Energy fallback
            elif current_section.energy_level == "high":
                # High energy section: 2 beats (fast) or 4 beats (medium-fast)
                return 2 if tempo < 130 else 4
            elif current_section.energy_level == "low":
                # Low energy section: 8 beats (slow) or 16 beats (very slow)
                return 16 if tempo < 100 else 8

        # 3. Static Pacing Modes
        if pacing_speed == "very_fast":
            return clamp_beats(1)
        elif pacing_speed == "fast":
            return clamp_beats(2 if tempo < 130 else 4)
        elif pacing_speed == "medium":
            return clamp_beats(4)
        elif pacing_speed == "slow":
            return clamp_beats(8)
        elif pacing_speed == "very_slow":
            return clamp_beats(16 if tempo < 100 else 8)
            
        # 4. Dynamic Pattern Matching
        else:
            # If pattern is exhausted or missing, generate new one
            if self.ctx.timeline.current_pattern is None or self.ctx.timeline.pattern_idx >= len(self.ctx.timeline.current_pattern):
                # Provide a deterministic first cut length before pattern generation
                if self.ctx.timeline.current_pattern is None and self.ctx.timeline.pattern_idx == 0:
                    return clamp_beats(4)
                self.ctx.timeline.current_pattern = calculate_dynamic_cut_length(
                    current_energy, tempo, self.ctx.timeline.current_time,
                    self.ctx.timeline.target_duration, cut_patterns
                )
                self.ctx.timeline.pattern_idx = 0

            beats = self.ctx.timeline.current_pattern[self.ctx.timeline.pattern_idx]
            self.ctx.timeline.pattern_idx += 1
            return clamp_beats(beats)

    def _calculate_cut_duration(self, beats_per_cut: float, beat_times: np.ndarray) -> float:
        """Calculate cut duration from beat times."""
        current_beat_idx = self.ctx.timeline.beat_idx
        target_beat_idx = current_beat_idx + beats_per_cut

        if target_beat_idx >= len(beat_times):
             # End of song/beats
            return self.ctx.timeline.target_duration - self.ctx.timeline.current_time

        # Interpolate for fractional beats
        idx_int = int(current_beat_idx)
        
        # Get start time of current beat
        # Note: current_time might not match beat_times[idx_int] exactly due to drift/edits, 
        # but usually we want to snap to nearest beat or use stored time.
        # The existing logic seemed to rely on current_time.
        # But technically duration should be beat_times[target] - beat_times[current]
        
        # Let's look at the original logic for _calculate_cut_duration to be precise.
        # Original:
        # idx_int = int(self.ctx.timeline.beat_idx)
        # ...
        # t_start - wait, current_time is what matters?
        
        # Actually better to calculate duration based on beat timestamps difference
        
        # Determine strict time range
        # Start Time:
        if idx_int < len(beat_times):
             # To keep sync, we might want to base off the beat time, 
             # OR just use current_time if we allow drift.
             # The standard montage logic usually aligns cuts to beats.
             pass
             
        # Targeted End Time:
        target_int = int(target_beat_idx)
        fraction = target_beat_idx - target_int
        
        if target_int >= len(beat_times):
            t_end = beat_times[-1]
        elif target_int < len(beat_times) - 1:
            t_end = beat_times[target_int] + (beat_times[target_int+1] - beat_times[target_int]) * fraction
        else:
            t_end = beat_times[target_int]
            
        # Current Time vs Beat Time
        # Ideally we want (t_end - t_current). 
        # But if we want to snap: (t_end - beat_times[current_beat_idx])
        # Let's trust the current_time in context is accurate cursor.
        
        # But wait! We need to update self.ctx.timeline.beat_idx for the NEXT call!
        # The original method didn't seem to update it? 
        # Ah, the caller `process_clip_task` updates `beat_idx`.
        # I should probably update it here or return it.
        # "Updates internal timeline state" says docstring.
        
        self.ctx.timeline.beat_idx = target_beat_idx
        
        # Calculate duration based on beat times to ensure sync
        # But we need to know "current time" in beat domain vs actual domain?
        # Let's stick to returning duration relative to current context time, targeting t_end.
        
        # Calculate start point from current beat_idx just like original
        fraction_start = current_beat_idx - idx_int
        if idx_int >= len(beat_times) - 1:
            t_start = beat_times[-1]
        elif idx_int < len(beat_times):
             t_start = beat_times[idx_int] + (beat_times[idx_int+1] - beat_times[idx_int]) * fraction_start
        else:
             t_start = beat_times[-1]

        cut_duration = t_end - t_start
        
        # Add micro-timing jitter for humanization
        jitter = random.uniform(-0.05, 0.05)
        if cut_duration + jitter > 0.5:
            cut_duration += jitter

        return max(0.5, cut_duration) # Ensure valid positive duration

    def _get_cut_patterns(self) -> List[List[float]]:
        # Default patterns
        cut_patterns = [
            [4, 4, 4, 4],     # Standard
            [2, 2, 4, 8],     # Build up
            [8, 4, 2, 2],     # Speed up
            [2, 2, 2, 2],     # Rapid fire
            [8, 8, 4, 4],     # Slow to medium
            [16, 8, 8, 4],    # Very slow start
            [4, 2, 1, 1, 2, 4] # Wave
        ]
        
        # Override from style
        if self.ctx.creative.style_params and "cut_patterns" in self.ctx.creative.style_params:
             cut_patterns = self.ctx.creative.style_params["cut_patterns"]
             
        return cut_patterns
