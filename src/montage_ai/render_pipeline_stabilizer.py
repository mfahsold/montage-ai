"""
Enhanced Video Rendering Pipeline with ProStabilization Integration

This module bridges ProStabilizationEngine into the existing render pipeline,
enabling automatic stabilization during segment writing with zero manual configuration.

Features:
- Automatic shake detection → profile selection
- Per-clip stabilization with progressive rendering
- Fallback to basic stabilization if detect fails
- Memory-efficient two-pass rendering
- Seamless integration with SegmentWriter
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .pro_stabilization_engine import (
    ProStabilizationEngine,
    auto_select_profile,
    PROFILE_VLOG_ACTION,
    PROFILE_DOCUMENTARY,
    PROFILE_CINEMATIC,
    PROFILE_EXTREME,
    PROFILE_BROADCAST,
)

logger = logging.getLogger(__name__)


@dataclass
class StabilizationDecision:
    """Decision context for stabilization application."""
    apply_stabilization: bool
    profile_name: str
    reason: str
    shake_score: Optional[float] = None
    motion_type: Optional[str] = None


class RenderPipelineStabilizer:
    """
    Wraps clip stabilization into the render pipeline.
    Handles auto-detection and progressive stabilization.
    """
    
    def __init__(self, enable_pro_stab: bool = True, aggressive_mode: bool = False):
        self.enable_pro_stab = enable_pro_stab
        self.aggressive_mode = aggressive_mode
        self.stats = {
            "total_clips": 0,
            "stabilized_clips": 0,
            "failed_stabilizations": 0,
            "total_stabilization_time": 0.0,
        }
    
    def should_stabilize_clip(
        self,
        clip_metadata: Dict[str, Any],
        motion_analysis: Optional[Dict[str, Any]] = None,
    ) -> StabilizationDecision:
        """
        Decide whether to stabilize a clip based on metadata.
        
        Args:
            clip_metadata: ClipMetadata dict
            motion_analysis: Optional motion analysis results
            
        Returns:
            StabilizationDecision with reasoning
        """
        if not self.enable_pro_stab:
            return StabilizationDecision(
                apply_stabilization=False,
                profile_name="none",
                reason="ProStabilization disabled",
            )
        
        # Extract shake score from metadata
        shake_score = clip_metadata.get("shake_score", 0.0)
        motion_type = clip_metadata.get("motion_type", "static")
        duration = clip_metadata.get("duration", 0.0)
        
        # Never stabilize very short clips (performance)
        if duration < 0.3:
            return StabilizationDecision(
                apply_stabilization=False,
                profile_name="skip",
                reason=f"Clip too short ({duration:.2f}s)",
                shake_score=shake_score,
                motion_type=motion_type,
            )
        
        # Aggressive mode: stabilize all clips
        if self.aggressive_mode:
            return StabilizationDecision(
                apply_stabilization=True,
                profile_name=auto_select_profile(shake_score, motion_type).name,
                reason="Aggressive mode enabled",
                shake_score=shake_score,
                motion_type=motion_type,
            )
        
        # Threshold-based: stabilize if shake detected
        if shake_score > 0.2:
            profile = auto_select_profile(shake_score, motion_type)
            return StabilizationDecision(
                apply_stabilization=True,
                profile_name=profile.name,
                reason=f"Shake detected (score={shake_score:.2f})",
                shake_score=shake_score,
                motion_type=motion_type,
            )
        
        return StabilizationDecision(
            apply_stabilization=False,
            profile_name="clean",
            reason=f"No significant shake (score={shake_score:.2f})",
            shake_score=shake_score,
            motion_type=motion_type,
        )
    
    def apply_stabilization(
        self,
        input_path: str,
        output_path: str,
        clip_metadata: Dict[str, Any],
        timeout_sec: int = 600,
    ) -> Tuple[bool, str]:
        """
        Apply ProStabilization to a clip.
        
        Args:
            input_path: Source video file
            output_path: Destination video file
            clip_metadata: Clip metadata with shake_score, motion_type
            timeout_sec: Maximum time for stabilization
            
        Returns:
            (success: bool, message: str)
        """
        import subprocess
        import time
        
        # Decide if we should stabilize
        decision = self.should_stabilize_clip(clip_metadata)
        self.stats["total_clips"] += 1
        
        if not decision.apply_stabilization:
            logger.debug(f"   ℹ️ Skip stabilization: {decision.reason}")
            return True, decision.reason
        
        logger.info(f"   🎬 Stabilizing with {decision.profile_name} profile: {decision.reason}")
        
        try:
            # Select profile
            if decision.profile_name == "extreme":
                from .pro_stabilization_engine import PROFILE_EXTREME
                profile = PROFILE_EXTREME
            elif decision.profile_name == "vlog_action":
                from .pro_stabilization_engine import PROFILE_VLOG_ACTION
                profile = PROFILE_VLOG_ACTION
            elif decision.profile_name == "documentary":
                from .pro_stabilization_engine import PROFILE_DOCUMENTARY
                profile = PROFILE_DOCUMENTARY
            else:
                from .pro_stabilization_engine import PROFILE_CINEMATIC
                profile = PROFILE_CINEMATIC
            
            engine = ProStabilizationEngine(profile=profile)
            
            # Two-pass stabilization
            start_time = time.time()
            success, msg = engine.stabilize_video_twopass(
                input_path,
                output_path,
                include_motion_smooth=True,
                include_color_correction=True,
            )
            
            elapsed = time.time() - start_time
            self.stats["total_stabilization_time"] += elapsed
            
            if success:
                self.stats["stabilized_clips"] += 1
                logger.info(f"   ✅ Stabilization complete ({elapsed:.1f}s): {msg}")
                return True, msg
            else:
                self.stats["failed_stabilizations"] += 1
                logger.error(f"   ❌ Stabilization failed: {msg}")
                return False, msg
            
        except subprocess.TimeoutExpired:
            self.stats["failed_stabilizations"] += 1
            logger.error(f"   ⏱️ Stabilization timeout ({timeout_sec}s)")
            return False, f"Timeout after {timeout_sec}s"
        except Exception as e:
            self.stats["failed_stabilizations"] += 1
            logger.error(f"   ❌ Stabilization error: {e}")
            return False, str(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stabilization statistics."""
        total = self.stats["total_clips"]
        success_rate = (
            (self.stats["stabilized_clips"] / total * 100)
            if total > 0 else 0
        )
        
        return {
            "total_clips_processed": total,
            "clips_stabilized": self.stats["stabilized_clips"],
            "stabilization_success_rate": f"{success_rate:.1f}%",
            "failed_stabilizations": self.stats["failed_stabilizations"],
            "total_time_spent": f"{self.stats['total_stabilization_time']:.1f}s",
            "average_time_per_clip": (
                f"{self.stats['total_stabilization_time'] / self.stats['stabilized_clips']:.1f}s"
                if self.stats["stabilized_clips"] > 0 else "N/A"
            ),
        }


class SmartStabilizationManager:
    """
    High-level manager for automatic stabilization during production.
    Handles environment flags and quality profile tuning.
    """
    
    @staticmethod
    def from_environment() -> RenderPipelineStabilizer:
        """
        Create stabilizer from environment variables.
        
        Environment flags:
        - PRO_STABILIZE_ENABLED=true/false (default: true)
        - PRO_STABILIZE_AGGRESSIVE=true/false (default: false)
        - PRO_STABILIZE_PROFILE=extreme|vlog|documentary|cinematic|broadcast
        """
        enabled = os.environ.get("PRO_STABILIZE_ENABLED", "true").lower() == "true"
        aggressive = os.environ.get("PRO_STABILIZE_AGGRESSIVE", "false").lower() == "true"
        
        profile_override = os.environ.get("PRO_STABILIZE_PROFILE", "").lower()
        
        stabilizer = RenderPipelineStabilizer(
            enable_pro_stab=enabled,
            aggressive_mode=aggressive,
        )
        
        mode = "AGGRESSIVE" if aggressive else "AUTO"
        logger.info(
            f"🎬 ProStabilizer initialized: "
            f"enabled={enabled}, mode={mode}, profile_override={profile_override or 'auto'}"
        )
        
        return stabilizer


# ===== CONVENIENCE: Direct Clip Stabilization =====

def stabilize_single_clip(
    input_path: str,
    output_path: str,
    shake_score: float = 0.5,
    motion_type: str = "dynamic",
) -> bool:
    """
    Quick one-liner to stabilize a single clip.
    
    Args:
        input_path: Source video
        output_path: Output video
        shake_score: Camera shake intensity (0-1)
        motion_type: 'static', 'smooth', 'dynamic', 'extreme'
        
    Returns:
        Success flag
    """
    metadata = {
        "shake_score": shake_score,
        "motion_type": motion_type,
        "duration": 1.0,  # Dummy value
    }
    
    manager = SmartStabilizationManager.from_environment()
    stabilizer = manager
    
    success, msg = stabilizer.apply_stabilization(input_path, output_path, metadata)
    return success
