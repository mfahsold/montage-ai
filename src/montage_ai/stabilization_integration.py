"""
Stabilization Integration Bridge

Connects ProStabilizationEngine to the main render pipeline.
Handles STABILIZE_MODE environment variable and auto-selects profiles.
"""

import os
import logging
from typing import Optional
from montage_ai.pro_stabilization_engine import (
    ProStabilizationEngine,
    StabilizationProfile,
    PROFILE_EXTREME,
    PROFILE_SUPER_EXTREME,
    PROFILE_VLOG_ACTION,
    PROFILE_CINEMATIC,
    PROFILE_DOCUMENTARY,
    PROFILE_BROADCAST,
)

logger = logging.getLogger(__name__)


class StabilizationBridge:
    """Bridge between render pipeline and ProStabilizationEngine."""
    
    def __init__(self):
        """Initialize with environment settings."""
        self.mode = os.getenv("STABILIZE_MODE", "professional").lower()
        self.enabled = os.getenv("STABILIZE_AI", "false").lower() == "true"
        self.aggressive = os.getenv("AGGRESSIVE_SMOOTHING", "false").lower() == "true"
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._log_config()
    
    def _log_config(self):
        """Log current configuration."""
        self.logger.info(f"StabilizationBridge initialized:")
        self.logger.info(f"  ✓ Enabled: {self.enabled}")
        self.logger.info(f"  ✓ Mode: {self.mode}")
        self.logger.info(f"  ✓ Aggressive: {self.aggressive}")
    
    def select_profile(self, shake_score: Optional[float] = None) -> StabilizationProfile:
        """
        Select stabilization profile based on STABILIZE_MODE.
        
        Args:
            shake_score: Optional 0-1 camera shake intensity
        
        Returns:
            Selected StabilizationProfile
        """
        profile_map = {
            "extreme": PROFILE_EXTREME,
            "super_extreme": PROFILE_SUPER_EXTREME,
            "aggressive": PROFILE_VLOG_ACTION,
            "professional": PROFILE_VLOG_ACTION,
            "documentary": PROFILE_DOCUMENTARY,
            "broadcast": PROFILE_BROADCAST,
            "cinematic": PROFILE_CINEMATIC,
        }
        
        selected = profile_map.get(self.mode, PROFILE_VLOG_ACTION)
        
        # Override to super_extreme if aggressive flag set
        if self.aggressive and selected != PROFILE_SUPER_EXTREME:
            selected = PROFILE_SUPER_EXTREME
            self.logger.info(f"🔥 AGGRESSIVE_SMOOTHING=true → SUPER_EXTREME profile")
        
        self.logger.info(f"Selected profile: {selected.name}")
        return selected
    
    def get_engine(self, shake_score: Optional[float] = None) -> ProStabilizationEngine:
        """
        Create ProStabilizationEngine with appropriate profile.
        
        Args:
            shake_score: Optional shake intensity for auto-selection
        
        Returns:
            Configured ProStabilizationEngine
        """
        if not self.enabled:
            self.logger.info("Stabilization disabled (STABILIZE_AI=false)")
            return None
        
        profile = self.select_profile(shake_score)
        engine = ProStabilizationEngine(profile=profile)
        
        self.logger.info(f"🎬 Created stabilization engine: {profile.name}")
        return engine
    
    def should_skip_motion_smooth(self) -> bool:
        """Check if motion smoothing should be skipped (for speed)."""
        fast_mode = os.getenv("FAST_STABILIZATION", "false").lower() == "true"
        return fast_mode
    
    def should_skip_color_correction(self) -> bool:
        """Check if color correction should be skipped."""
        skip_color = os.getenv("SKIP_COLOR_CORRECTION", "false").lower() == "true"
        return skip_color
    
    def get_ffmpeg_filters_for_clip(
        self,
        clip_path: str,
        shake_score: Optional[float] = 0.5,
    ) -> Optional[str]:
        """
        Get FFmpeg filter string for a clip with stabilization.
        
        Args:
            clip_path: Path to video clip
            shake_score: Optional shake intensity (0-1)
        
        Returns:
            FFmpeg filter string, or None if stabilization disabled
        """
        if not self.enabled:
            return None
        
        engine = self.get_engine(shake_score)
        if not engine:
            return None
        
        # Note: This would require the .trf file from prior detection pass
        # For now, return None and let render_engine.py handle full 2-pass
        self.logger.warning("Use stabilize_video_twopass() for full processing")
        return None
    
    def stabilize_clip(
        self,
        input_path: str,
        output_path: str,
        shake_score: Optional[float] = 0.5,
    ) -> bool:
        """
        Stabilize a single clip with selected profile.
        
        Args:
            input_path: Source video
            output_path: Output video
            shake_score: Optional shake intensity
        
        Returns:
            Success flag
        """
        if not self.enabled:
            self.logger.info("Stabilization disabled")
            return True  # Not an error, just skipped
        
        engine = self.get_engine(shake_score)
        if not engine:
            return False
        
        include_motion_smooth = not self.should_skip_motion_smooth()
        include_color_correction = not self.should_skip_color_correction()
        
        self.logger.info(f"Stabilizing: {input_path}")
        self.logger.info(f"  Motion smooth: {include_motion_smooth}")
        self.logger.info(f"  Color correction: {include_color_correction}")
        
        success, msg = engine.stabilize_video_twopass(
            input_path,
            output_path,
            include_motion_smooth=include_motion_smooth,
            include_color_correction=include_color_correction,
        )
        
        if success:
            self.logger.info(f"✅ {msg}")
        else:
            self.logger.error(f"❌ {msg}")
        
        return success


# ===== SINGLETON INSTANCE =====

_bridge_instance: Optional[StabilizationBridge] = None


def get_stabilization_bridge() -> StabilizationBridge:
    """Get or create stabilization bridge singleton."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = StabilizationBridge()
    return _bridge_instance


def initialize_stabilization() -> None:
    """Initialize stabilization system."""
    bridge = get_stabilization_bridge()
    logger.info(f"Stabilization initialized: mode={bridge.mode}, enabled={bridge.enabled}")


# ===== CONVENIENCE FUNCTIONS FOR RENDER PIPELINE =====

def should_stabilize() -> bool:
    """Check if stabilization should be applied."""
    return get_stabilization_bridge().enabled


def get_stabilization_mode() -> str:
    """Get current stabilization mode."""
    return get_stabilization_bridge().mode


def stabilize_for_render(
    input_path: str,
    output_path: str,
    shake_score: float = 0.5,
) -> bool:
    """
    Main entry point for render pipeline stabilization.
    
    Args:
        input_path: Source video
        output_path: Enhanced video
        shake_score: Camera shake intensity (0-1)
    
    Returns:
        Success flag
    """
    bridge = get_stabilization_bridge()
    return bridge.stabilize_clip(input_path, output_path, shake_score)
