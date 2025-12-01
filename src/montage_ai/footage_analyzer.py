"""
Montage AI Video Editor - Deep Footage Analysis

# STATUS: Work in Progress - Not yet fully tested

Provides rich, human-readable analysis of video footage:
- Visual characteristics (motion, brightness, color temperature, composition)
- Emotional qualities (energy, mood, tension)
- Technical attributes (resolution, stability, focus)
- Narrative potential (establishing shots, action, transitions)

This analysis feeds into more sophisticated creative decisions
and enables natural language prompting for precise editing control.

Version: 1.0.0
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class ShotType(Enum):
    """Cinematographic shot classification"""
    EXTREME_WIDE = "extreme_wide"      # Landscape, establishing
    WIDE = "wide"                       # Full environment
    MEDIUM_WIDE = "medium_wide"         # Full body
    MEDIUM = "medium"                   # Waist up
    MEDIUM_CLOSE = "medium_close"       # Chest up
    CLOSE_UP = "close_up"               # Face/detail
    EXTREME_CLOSE = "extreme_close"     # Eye/texture detail


class MotionType(Enum):
    """Camera/subject motion classification"""
    STATIC = "static"                   # Tripod, no movement
    SUBTLE = "subtle"                   # Minor handheld shake
    SLOW_PAN = "slow_pan"               # Deliberate camera move
    TRACKING = "tracking"               # Following subject
    DYNAMIC = "dynamic"                 # Significant movement
    CHAOTIC = "chaotic"                 # Unstable, action


class MoodType(Enum):
    """Emotional/atmospheric classification"""
    CALM = "calm"
    CONTEMPLATIVE = "contemplative"
    TENSE = "tense"
    ENERGETIC = "energetic"
    DRAMATIC = "dramatic"
    INTIMATE = "intimate"
    EPIC = "epic"


@dataclass
class VisualCharacteristics:
    """Low-level visual analysis results"""
    avg_brightness: float = 0.0         # 0-1 scale
    brightness_variance: float = 0.0    # How much brightness changes
    dominant_colors: List[str] = field(default_factory=list)  # ["warm", "cool", "neutral"]
    color_saturation: float = 0.0       # 0-1 scale
    contrast: float = 0.0               # 0-1 scale
    edge_density: float = 0.0           # Visual complexity
    blur_amount: float = 0.0            # Focus quality (0=sharp, 1=blurry)


@dataclass
class MotionCharacteristics:
    """Motion analysis results"""
    motion_intensity: float = 0.0       # 0-1 scale
    motion_type: str = "unknown"        # From MotionType
    motion_direction: str = "none"      # "left", "right", "up", "down", "mixed"
    camera_shake: float = 0.0           # 0-1 scale (handheld detection)
    subject_movement: float = 0.0       # Movement within frame


@dataclass 
class NarrativeQualities:
    """High-level narrative/editorial analysis"""
    shot_type: str = "medium"           # From ShotType
    establishing_potential: float = 0.0  # Good for opening? 0-1
    transition_potential: float = 0.0    # Good for scene change? 0-1
    climax_potential: float = 0.0        # Good for peak moment? 0-1
    closing_potential: float = 0.0       # Good for ending? 0-1
    mood: str = "neutral"                # From MoodType
    energy_level: float = 0.5            # 0=calm, 1=intense


@dataclass
class SceneAnalysis:
    """Complete analysis of a video scene/clip"""
    source_file: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    
    # Technical
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    
    # Analysis results
    visual: VisualCharacteristics = field(default_factory=VisualCharacteristics)
    motion: MotionCharacteristics = field(default_factory=MotionCharacteristics)
    narrative: NarrativeQualities = field(default_factory=NarrativeQualities)
    
    # Human-readable summary
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Editing recommendations
    best_used_for: List[str] = field(default_factory=list)
    pairs_well_with: List[str] = field(default_factory=list)
    avoid_after: List[str] = field(default_factory=list)


class DeepFootageAnalyzer:
    """
    Performs comprehensive visual analysis of video clips.
    
    Extracts both technical metrics and high-level narrative qualities
    to enable more sophisticated editing decisions.
    """
    
    def __init__(self, sample_frames: int = 10, verbose: bool = True):
        """
        Args:
            sample_frames: Number of frames to analyze per clip
            verbose: Print detailed analysis results
        """
        self.sample_frames = sample_frames
        self.verbose = verbose
        self.analyses: List[SceneAnalysis] = []
    
    def analyze_clip(self, video_path: str, start_time: float = 0, 
                     end_time: float = None) -> SceneAnalysis:
        """
        Perform deep analysis of a video clip.
        
        Args:
            video_path: Path to video file
            start_time: Start of segment to analyze (seconds)
            end_time: End of segment (None = full video)
            
        Returns:
            SceneAnalysis with all metrics and descriptions
        """
        analysis = SceneAnalysis(source_file=os.path.basename(video_path))
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return analysis
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            analysis.resolution = (width, height)
            analysis.fps = fps
            analysis.start_time = start_time
            analysis.end_time = end_time or duration
            analysis.duration = analysis.end_time - analysis.start_time
            
            # Calculate frame positions to sample
            start_frame = int(start_time * fps)
            end_frame = int((end_time or duration) * fps)
            frame_step = max(1, (end_frame - start_frame) // self.sample_frames)
            
            frames = []
            prev_frame = None
            motion_magnitudes = []
            
            for i, frame_idx in enumerate(range(start_frame, end_frame, frame_step)):
                if i >= self.sample_frames:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frames.append(frame)
                
                # Motion detection via frame differencing
                if prev_frame is not None:
                    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray_curr, gray_prev)
                    motion_magnitudes.append(np.mean(diff) / 255.0)
                
                prev_frame = frame
            
            cap.release()
            
            if not frames:
                return analysis
            
            # Analyze visual characteristics
            analysis.visual = self._analyze_visuals(frames)
            
            # Analyze motion
            analysis.motion = self._analyze_motion(motion_magnitudes, frames)
            
            # Derive narrative qualities
            analysis.narrative = self._derive_narrative(analysis.visual, analysis.motion, 
                                                        width, height, analysis.duration)
            
            # Generate human-readable description
            analysis.description = self._generate_description(analysis)
            analysis.tags = self._generate_tags(analysis)
            analysis.best_used_for = self._suggest_usage(analysis)
            
            self.analyses.append(analysis)
            
            if self.verbose:
                self._print_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"   ⚠️ Analysis failed for {video_path}: {e}")
            return analysis
    
    def _analyze_visuals(self, frames: List[np.ndarray]) -> VisualCharacteristics:
        """Analyze visual characteristics from sampled frames"""
        vis = VisualCharacteristics()
        
        if not frames:
            return vis
        
        brightnesses = []
        saturations = []
        contrasts = []
        edge_densities = []
        blur_amounts = []
        color_temps = []
        
        for frame in frames:
            # Convert to different color spaces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Brightness (V channel of HSV)
            brightness = np.mean(hsv[:, :, 2]) / 255.0
            brightnesses.append(brightness)
            
            # Saturation
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            saturations.append(saturation)
            
            # Contrast (standard deviation of grayscale)
            contrast = np.std(gray) / 128.0  # Normalize to ~0-1
            contrasts.append(min(1.0, contrast))
            
            # Edge density (visual complexity)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255.0
            edge_densities.append(edge_density)
            
            # Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Higher variance = sharper image
            blur = max(0, min(1, 1 - (laplacian_var / 500.0)))
            blur_amounts.append(blur)
            
            # Color temperature (warm vs cool)
            b, g, r = cv2.split(frame)
            warmth = (np.mean(r) - np.mean(b)) / 255.0  # -1 to 1
            color_temps.append(warmth)
        
        vis.avg_brightness = np.mean(brightnesses)
        vis.brightness_variance = np.std(brightnesses)
        vis.color_saturation = np.mean(saturations)
        vis.contrast = np.mean(contrasts)
        vis.edge_density = np.mean(edge_densities)
        vis.blur_amount = np.mean(blur_amounts)
        
        # Determine dominant color temperature
        avg_warmth = np.mean(color_temps)
        if avg_warmth > 0.1:
            vis.dominant_colors = ["warm", "golden"]
        elif avg_warmth < -0.1:
            vis.dominant_colors = ["cool", "blue"]
        else:
            vis.dominant_colors = ["neutral", "balanced"]
        
        return vis
    
    def _analyze_motion(self, motion_mags: List[float], 
                        frames: List[np.ndarray]) -> MotionCharacteristics:
        """Analyze motion characteristics"""
        motion = MotionCharacteristics()
        
        if not motion_mags:
            motion.motion_type = MotionType.STATIC.value
            return motion
        
        avg_motion = np.mean(motion_mags)
        motion_std = np.std(motion_mags)
        
        motion.motion_intensity = min(1.0, avg_motion * 10)  # Scale up for visibility
        
        # Classify motion type
        if avg_motion < 0.01:
            motion.motion_type = MotionType.STATIC.value
        elif avg_motion < 0.03:
            motion.motion_type = MotionType.SUBTLE.value
        elif avg_motion < 0.06:
            motion.motion_type = MotionType.SLOW_PAN.value
        elif avg_motion < 0.10:
            motion.motion_type = MotionType.DYNAMIC.value
        else:
            motion.motion_type = MotionType.CHAOTIC.value
        
        # Camera shake estimation (high frequency motion)
        if motion_std > 0.02:
            motion.camera_shake = min(1.0, motion_std * 20)
        
        return motion
    
    def _derive_narrative(self, vis: VisualCharacteristics, 
                          motion: MotionCharacteristics,
                          width: int, height: int,
                          duration: float) -> NarrativeQualities:
        """Derive high-level narrative qualities from visual/motion analysis"""
        narr = NarrativeQualities()
        
        # Estimate shot type from edge density and resolution
        # (More edges often = closer shot with more detail)
        if vis.edge_density > 0.3:
            narr.shot_type = ShotType.CLOSE_UP.value
        elif vis.edge_density > 0.2:
            narr.shot_type = ShotType.MEDIUM.value
        elif vis.edge_density > 0.1:
            narr.shot_type = ShotType.MEDIUM_WIDE.value
        else:
            narr.shot_type = ShotType.WIDE.value
        
        # Establishing potential (wide, static, good exposure)
        narr.establishing_potential = (
            (0.3 if narr.shot_type in [ShotType.WIDE.value, ShotType.EXTREME_WIDE.value] else 0.0) +
            (0.3 if motion.motion_type == MotionType.STATIC.value else 0.1) +
            (0.2 if 0.3 < vis.avg_brightness < 0.7 else 0.0) +
            (0.2 if duration > 3.0 else 0.0)
        )
        
        # Transition potential (motion, brightness changes)
        narr.transition_potential = (
            (0.3 if motion.motion_type in [MotionType.SLOW_PAN.value, MotionType.TRACKING.value] else 0.0) +
            (0.3 if vis.brightness_variance > 0.1 else 0.0) +
            (0.2 if vis.blur_amount > 0.3 else 0.0) +
            (0.2 if 1.0 < duration < 4.0 else 0.0)
        )
        
        # Climax potential (dynamic, high contrast, saturated)
        narr.climax_potential = (
            (0.3 if motion.motion_intensity > 0.5 else motion.motion_intensity * 0.6) +
            (0.2 if vis.contrast > 0.6 else 0.0) +
            (0.2 if vis.color_saturation > 0.5 else 0.0) +
            (0.3 if narr.shot_type in [ShotType.CLOSE_UP.value, ShotType.MEDIUM_CLOSE.value] else 0.0)
        )
        
        # Closing potential (static, calm, resolved feeling)
        narr.closing_potential = (
            (0.3 if motion.motion_type == MotionType.STATIC.value else 0.0) +
            (0.2 if vis.avg_brightness > 0.5 else 0.1) +
            (0.2 if narr.shot_type in [ShotType.WIDE.value, ShotType.MEDIUM_WIDE.value] else 0.0) +
            (0.3 if duration > 2.0 else 0.0)
        )
        
        # Mood classification
        energy = motion.motion_intensity * 0.6 + vis.contrast * 0.2 + vis.color_saturation * 0.2
        narr.energy_level = energy
        
        if energy < 0.2:
            if vis.avg_brightness < 0.4:
                narr.mood = MoodType.CONTEMPLATIVE.value
            else:
                narr.mood = MoodType.CALM.value
        elif energy < 0.4:
            if vis.contrast > 0.5:
                narr.mood = MoodType.TENSE.value
            else:
                narr.mood = MoodType.INTIMATE.value
        elif energy < 0.7:
            narr.mood = MoodType.DRAMATIC.value
        else:
            if narr.shot_type in [ShotType.WIDE.value, ShotType.EXTREME_WIDE.value]:
                narr.mood = MoodType.EPIC.value
            else:
                narr.mood = MoodType.ENERGETIC.value
        
        return narr
    
    def _generate_description(self, analysis: SceneAnalysis) -> str:
        """Generate human-readable description of the clip"""
        vis = analysis.visual
        motion = analysis.motion
        narr = analysis.narrative
        
        # Build description parts
        parts = []
        
        # Shot type
        shot_desc = {
            "extreme_wide": "An expansive wide shot",
            "wide": "A wide establishing shot",
            "medium_wide": "A medium-wide shot",
            "medium": "A medium shot",
            "medium_close": "A medium close-up",
            "close_up": "A close-up shot",
            "extreme_close": "An extreme close-up"
        }
        parts.append(shot_desc.get(narr.shot_type, "A shot"))
        
        # Motion
        motion_desc = {
            "static": "with steady, tripod-like stability",
            "subtle": "with subtle handheld movement",
            "slow_pan": "with a deliberate camera movement",
            "tracking": "following the action",
            "dynamic": "with energetic camera work",
            "chaotic": "with intense, unstable motion"
        }
        parts.append(motion_desc.get(motion.motion_type, ""))
        
        # Lighting/mood
        if vis.avg_brightness > 0.7:
            parts.append("in bright, high-key lighting")
        elif vis.avg_brightness < 0.3:
            parts.append("in moody, low-key lighting")
        elif "warm" in vis.dominant_colors:
            parts.append("with warm, golden tones")
        elif "cool" in vis.dominant_colors:
            parts.append("with cool, blue tones")
        
        # Mood
        mood_desc = {
            "calm": "The mood is peaceful and serene.",
            "contemplative": "The mood is thoughtful and introspective.",
            "tense": "There's an underlying tension.",
            "energetic": "The energy is vibrant and active.",
            "dramatic": "The atmosphere is dramatic.",
            "intimate": "The feel is personal and intimate.",
            "epic": "The scale feels epic and grand."
        }
        parts.append(mood_desc.get(narr.mood, ""))
        
        # Duration context
        if analysis.duration < 2:
            parts.append(f"A brief {analysis.duration:.1f}s moment.")
        elif analysis.duration > 8:
            parts.append(f"A sustained {analysis.duration:.1f}s take.")
        
        return " ".join(filter(None, parts))
    
    def _generate_tags(self, analysis: SceneAnalysis) -> List[str]:
        """Generate searchable tags for the clip"""
        tags = []
        vis = analysis.visual
        motion = analysis.motion
        narr = analysis.narrative
        
        # Shot type tag
        tags.append(narr.shot_type.replace("_", "-"))
        
        # Motion tags
        tags.append(motion.motion_type)
        if motion.camera_shake > 0.3:
            tags.append("handheld")
        if motion.motion_intensity < 0.1:
            tags.append("stable")
        
        # Visual tags
        if vis.avg_brightness > 0.6:
            tags.append("bright")
        elif vis.avg_brightness < 0.4:
            tags.append("dark")
        
        if vis.color_saturation > 0.6:
            tags.append("saturated")
        elif vis.color_saturation < 0.3:
            tags.append("desaturated")
        
        tags.extend(vis.dominant_colors)
        
        # Mood tag
        tags.append(narr.mood)
        
        # Narrative potential tags
        if narr.establishing_potential > 0.6:
            tags.append("establishing")
        if narr.climax_potential > 0.6:
            tags.append("climax")
        if narr.transition_potential > 0.6:
            tags.append("transition")
        if narr.closing_potential > 0.6:
            tags.append("closing")
        
        # Duration tags
        if analysis.duration < 2:
            tags.append("quick-cut")
        elif analysis.duration > 6:
            tags.append("long-take")
        
        return list(set(tags))  # Remove duplicates
    
    def _suggest_usage(self, analysis: SceneAnalysis) -> List[str]:
        """Suggest best editorial uses for this clip"""
        suggestions = []
        narr = analysis.narrative
        motion = analysis.motion
        
        if narr.establishing_potential > 0.5:
            suggestions.append("opening_sequence")
            suggestions.append("scene_introduction")
        
        if narr.climax_potential > 0.5:
            suggestions.append("beat_drop")
            suggestions.append("emotional_peak")
        
        if narr.transition_potential > 0.5:
            suggestions.append("scene_transition")
            suggestions.append("time_passage")
        
        if narr.closing_potential > 0.5:
            suggestions.append("sequence_ending")
            suggestions.append("resolution")
        
        if motion.motion_type == "static" and narr.mood == "calm":
            suggestions.append("breathing_room")
            suggestions.append("contemplative_pause")
        
        if motion.motion_intensity > 0.6:
            suggestions.append("energy_burst")
            suggestions.append("action_moment")
        
        if narr.shot_type == "close_up":
            suggestions.append("detail_emphasis")
            suggestions.append("emotional_connection")
        
        return suggestions if suggestions else ["general_coverage"]
    
    def _print_analysis(self, analysis: SceneAnalysis):
        """Print formatted analysis results"""
        print(f"\n   {'─' * 50}")
        print(f"   📹 CLIP ANALYSIS: {analysis.source_file}")
        print(f"   {'─' * 50}")
        print(f"   ⏱️  Duration: {analysis.duration:.1f}s ({analysis.start_time:.1f}s - {analysis.end_time:.1f}s)")
        print(f"   📐 Resolution: {analysis.resolution[0]}x{analysis.resolution[1]} @ {analysis.fps:.0f}fps")
        print()
        print(f"   🎬 VISUAL CHARACTERISTICS:")
        print(f"      • Brightness: {analysis.visual.avg_brightness:.2f} (variance: {analysis.visual.brightness_variance:.2f})")
        print(f"      • Saturation: {analysis.visual.color_saturation:.2f}")
        print(f"      • Contrast: {analysis.visual.contrast:.2f}")
        print(f"      • Color Temp: {', '.join(analysis.visual.dominant_colors)}")
        print(f"      • Sharpness: {1-analysis.visual.blur_amount:.2f}")
        print()
        print(f"   🎥 MOTION ANALYSIS:")
        print(f"      • Type: {analysis.motion.motion_type}")
        print(f"      • Intensity: {analysis.motion.motion_intensity:.2f}")
        print(f"      • Camera Shake: {analysis.motion.camera_shake:.2f}")
        print()
        print(f"   📖 NARRATIVE QUALITIES:")
        print(f"      • Shot Type: {analysis.narrative.shot_type}")
        print(f"      • Mood: {analysis.narrative.mood}")
        print(f"      • Energy Level: {analysis.narrative.energy_level:.2f}")
        print(f"      • Establishing: {analysis.narrative.establishing_potential:.2f}")
        print(f"      • Climax: {analysis.narrative.climax_potential:.2f}")
        print(f"      • Transition: {analysis.narrative.transition_potential:.2f}")
        print(f"      • Closing: {analysis.narrative.closing_potential:.2f}")
        print()
        print(f"   📝 DESCRIPTION:")
        print(f"      {analysis.description}")
        print()
        print(f"   🏷️  TAGS: {', '.join(analysis.tags)}")
        print(f"   💡 BEST FOR: {', '.join(analysis.best_used_for)}")
        print(f"   {'─' * 50}")
    
    def get_footage_summary(self) -> Dict[str, Any]:
        """Generate a summary of all analyzed footage"""
        if not self.analyses:
            return {}
        
        summary = {
            "total_clips": len(self.analyses),
            "total_duration": sum(a.duration for a in self.analyses),
            "shot_distribution": {},
            "mood_distribution": {},
            "motion_distribution": {},
            "narrative_inventory": {
                "establishing_shots": [],
                "climax_candidates": [],
                "transition_shots": [],
                "closing_shots": []
            },
            "energy_profile": {
                "low": 0,
                "medium": 0,
                "high": 0
            }
        }
        
        for a in self.analyses:
            # Shot distribution
            shot = a.narrative.shot_type
            summary["shot_distribution"][shot] = summary["shot_distribution"].get(shot, 0) + 1
            
            # Mood distribution
            mood = a.narrative.mood
            summary["mood_distribution"][mood] = summary["mood_distribution"].get(mood, 0) + 1
            
            # Motion distribution
            motion = a.motion.motion_type
            summary["motion_distribution"][motion] = summary["motion_distribution"].get(motion, 0) + 1
            
            # Narrative inventory
            if a.narrative.establishing_potential > 0.5:
                summary["narrative_inventory"]["establishing_shots"].append(a.source_file)
            if a.narrative.climax_potential > 0.5:
                summary["narrative_inventory"]["climax_candidates"].append(a.source_file)
            if a.narrative.transition_potential > 0.5:
                summary["narrative_inventory"]["transition_shots"].append(a.source_file)
            if a.narrative.closing_potential > 0.5:
                summary["narrative_inventory"]["closing_shots"].append(a.source_file)
            
            # Energy profile
            if a.narrative.energy_level < 0.3:
                summary["energy_profile"]["low"] += 1
            elif a.narrative.energy_level < 0.6:
                summary["energy_profile"]["medium"] += 1
            else:
                summary["energy_profile"]["high"] += 1
        
        return summary
    
    def print_footage_summary(self):
        """Print human-readable footage summary"""
        summary = self.get_footage_summary()
        if not summary:
            print("   No footage analyzed yet.")
            return
        
        print(f"\n{'=' * 60}")
        print(f"📊 DEEP FOOTAGE ANALYSIS SUMMARY")
        print(f"{'=' * 60}")
        print(f"\n   📁 Total Clips: {summary['total_clips']}")
        print(f"   ⏱️  Total Duration: {summary['total_duration']:.1f}s ({summary['total_duration']/60:.1f} min)")
        
        print(f"\n   🎬 SHOT TYPE DISTRIBUTION:")
        for shot, count in sorted(summary["shot_distribution"].items(), key=lambda x: -x[1]):
            pct = count / summary["total_clips"] * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"      {shot.replace('_', ' ').title():15} [{bar}] {count:2} ({pct:.0f}%)")
        
        print(f"\n   🎭 MOOD DISTRIBUTION:")
        for mood, count in sorted(summary["mood_distribution"].items(), key=lambda x: -x[1]):
            pct = count / summary["total_clips"] * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"      {mood.title():15} [{bar}] {count:2} ({pct:.0f}%)")
        
        print(f"\n   🎥 MOTION DISTRIBUTION:")
        for motion, count in sorted(summary["motion_distribution"].items(), key=lambda x: -x[1]):
            pct = count / summary["total_clips"] * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"      {motion.title():15} [{bar}] {count:2} ({pct:.0f}%)")
        
        print(f"\n   ⚡ ENERGY PROFILE:")
        for level in ["low", "medium", "high"]:
            count = summary["energy_profile"][level]
            pct = count / summary["total_clips"] * 100
            emoji = {"low": "🧘", "medium": "🚶", "high": "🏃"}[level]
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"      {emoji} {level.title():10} [{bar}] {count:2} ({pct:.0f}%)")
        
        print(f"\n   📖 NARRATIVE INVENTORY:")
        inv = summary["narrative_inventory"]
        print(f"      🌅 Establishing Shots: {len(inv['establishing_shots'])} clips")
        print(f"      ⚡ Climax Candidates:  {len(inv['climax_candidates'])} clips")
        print(f"      🔀 Transition Shots:   {len(inv['transition_shots'])} clips")
        print(f"      🌇 Closing Shots:      {len(inv['closing_shots'])} clips")
        
        print(f"\n{'=' * 60}\n")
    
    def export_analysis(self, filepath: str):
        """Export all analyses to JSON"""
        data = {
            "analyses": [asdict(a) for a in self.analyses],
            "summary": self.get_footage_summary()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"   ✅ Analysis exported to {filepath}")


# Convenience function for quick analysis
def analyze_footage(video_paths: List[str], verbose: bool = True) -> DeepFootageAnalyzer:
    """
    Analyze multiple video files and return the analyzer instance.
    
    Example:
        analyzer = analyze_footage(["/data/input/video1.mp4", "/data/input/video2.mp4"])
        analyzer.print_footage_summary()
    """
    analyzer = DeepFootageAnalyzer(verbose=verbose)
    for path in video_paths:
        analyzer.analyze_clip(path)
    return analyzer
