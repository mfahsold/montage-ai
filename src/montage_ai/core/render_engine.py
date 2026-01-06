"""
Render Engine Module

Extracts rendering logic from MontageBuilder to separate concerns.
Handles progressive rendering finalization and FFmpeg concatenation.
"""

import os
import time
from typing import Optional, Any
from ..logger import logger
from .context import MontageContext
from ..segment_writer import SegmentWriter

class RenderEngine:
    """
    Handles final rendering of the montage.
    """
    def __init__(self, context: MontageContext):
        self.ctx = context
        self.settings = context.settings
        self._progressive_renderer: Optional[SegmentWriter] = None

    def init_progressive_renderer(self):
        """Initialize segment writer for progressive output."""
        # This matches _init_progressive_renderer from MontageBuilder roughly,
        # but typically the renderer is initialized during the planning phase.
        # For this refactor, we assume the builder assigns the renderer to this engine
        # OR we just handle the finalization part.
        
        # Checking existing code: MontageBuilder calls _init_progressive_renderer() in plan_montage().
        pass
    
    def set_renderer(self, renderer: SegmentWriter):
        self._progressive_renderer = renderer

    def render_output(self) -> None:
        """
        Phase 5: Render final output.
        """
        logger.info("\n   🎬 Rendering output...")

        if self.settings.processing.should_skip_output(self.ctx.render.output_filename):
            logger.info(f"   ♻️ Output exists, skipping render: {os.path.basename(self.ctx.render.output_filename)}")
            self.ctx.render.render_duration = 0.0
            return

        render_start_time = time.time()

        if self._progressive_renderer:
            # Progressive path: finalize with FFmpeg
            logger.info(f"   🔗 Finalizing with Progressive Renderer ({self._progressive_renderer.get_segment_count()} segments)...")

            audio_duration = self.ctx.timeline.target_duration
            audio_path = self.ctx.media.audio_result.music_path

            # Apply dialogue ducking if enabled
            if self.ctx.features.dialogue_duck and audio_path:
                try:
                    from ..dialogue_ducking import apply_ducking_to_audio
                    # Use voice track if available (from voice isolation), else try original audio
                    voice_path = getattr(self.ctx.media.audio_result, 'voice_path', None) or audio_path
                    ducked_audio_path = os.path.join(str(self.ctx.paths.temp_dir), "ducked_audio.m4a")
                    duck_level = self.settings.features.dialogue_duck_level
                    logger.info(f"   🔇 Applying dialogue ducking ({duck_level}dB)...")
                    result = apply_ducking_to_audio(
                        music_path=audio_path,
                        voice_path=voice_path,
                        output_path=ducked_audio_path,
                        duck_level_db=duck_level
                    )
                    if result and os.path.exists(result):
                        audio_path = result
                        logger.info("   ✅ Dialogue ducking applied")
                except Exception as e:
                    logger.warning(f"   ⚠️ Dialogue ducking failed: {e}")

            success = self._progressive_renderer.finalize(
                output_path=self.ctx.render.output_filename,
                audio_path=audio_path,
                audio_duration=audio_duration,
                logo_path=self.ctx.render.logo_path
            )

            if success:
                method_str = "xfade" if self.ctx.timeline.enable_xfade else "-c copy"
                self.ctx.render.render_duration = time.time() - render_start_time
                logger.info(f"   ✅ Final video rendered via FFmpeg ({method_str}) in {self.ctx.render.render_duration:.1f}s")
                if self.ctx.render.logo_path:
                    logger.info(f"   🏷️ Logo overlay: {os.path.basename(self.ctx.render.logo_path)}")
            else:
                raise RuntimeError("Progressive render failed")
        else:
            # Legacy path
            raise NotImplementedError("Legacy rendering not implemented in RenderEngine")

    def cleanup(self):
        """Cleanup render resources."""
        if self._progressive_renderer:
             try:
                self._progressive_renderer.cleanup()
             except Exception as e:
                logger.warning(f"   ⚠️ Progressive renderer cleanup failed: {e}")
