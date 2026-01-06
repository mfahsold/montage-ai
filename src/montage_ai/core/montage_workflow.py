"""
Montage Creator Workflow - Concrete Implementation

Beat-synced video editing with scene detection and intelligent clip selection.
"""

from typing import Any, Optional, Dict
from pathlib import Path

from .workflow import VideoWorkflow, WorkflowOptions, WorkflowPhase
from ..logger import logger
from .montage_builder import MontageBuilder
from ..config import get_settings


class MontageWorkflow(VideoWorkflow):
    """
    Montage Creator workflow implementation.
    
    Pipeline:
    1. Initialize: Setup builders, analyzers
    2. Validate: Check footage + music exists
    3. Analyze: Scene detection + beat detection
    4. Process: Clip selection + sequencing
    5. Render: FFmpeg rendering with effects
    6. Export: Finalize output
    """
    
    def __init__(self, options: WorkflowOptions):
        super().__init__(options)
        self.builder: Optional[MontageBuilder] = None
        self.settings = get_settings()

    @property
    def workflow_name(self) -> str:
        return "Montage Creator"
    
    @property
    def workflow_type(self) -> str:
        return "montage"
    
    # =========================================================================
    # Workflow Steps
    # =========================================================================
    
    def initialize(self) -> None:
        """Initialize montage builder."""
        # Extract variant_id from options.extras or default to 1
        variant_id = self.options.extras.get('variant_id', 1)
        editing_instructions = self.options.extras.get('editing_instructions', {}) or {}
        
        # Inject Music preferences into instructions (so Builder can find them)
        if self.options.extras.get('music_track'):
            editing_instructions['music_track'] = self.options.extras.get('music_track')
        if self.options.extras.get('music_start'):
            editing_instructions['music_start'] = self.options.extras.get('music_start')
        if self.options.extras.get('music_end'):
            editing_instructions['music_end'] = self.options.extras.get('music_end')
            
        # Define progress callback wrapper
        def builder_progress_callback(percent: int, message: str):
            # Map builder progress (0-100 of a specific phase) to global workflow progress
            # This is a simplification; ideally we map phases more accurately
            if self.current_phase == WorkflowPhase.ANALYZING:
                # Analyzing is 10-40% of standard workflow
                global_percent = 10 + int(percent * 0.3)
                self._update_progress(global_percent, message)
            elif self.current_phase == WorkflowPhase.RENDERING:
                # Rendering is 60-90% of standard workflow
                global_percent = 60 + int(percent * 0.3)
                self._update_progress(global_percent, message)
            else:
                # Default pass-through
                self._update_progress(percent, message)

        # Initialize builder
        self.builder = MontageBuilder(
            variant_id=variant_id,
            settings=self.settings,
            editing_instructions=editing_instructions,
            job_id=self.options.job_id,
            progress_callback=builder_progress_callback
        )
        
        # Apply workflow options to builder context
        self.builder.ctx.features.stabilize = self.options.stabilize
        self.builder.ctx.features.upscale = self.options.upscale
        self.builder.ctx.features.enhance = self.options.enhance
        
        # Apply advanced features from extras
        feats = self.builder.ctx.features
        extras = self.options.extras
        
        if 'color_grading' in extras:
            feats.color_grade = extras['color_grading']
        if 'color_intensity' in extras:
            feats.color_intensity = float(extras['color_intensity'])
        if 'denoise' in extras:
            feats.denoise = extras['denoise']
        if 'sharpen' in extras:
            feats.sharpen = extras['sharpen']
        if 'film_grain' in extras:
            feats.film_grain = extras['film_grain']
        if 'dialogue_duck' in extras:
            feats.dialogue_duck = extras['dialogue_duck']
            
        # Phase 1: Setup
        self.builder.setup_workspace()
    
    def validate(self) -> None:
        """Validate inputs."""
        # Basic validation
        if not self.builder:
            raise RuntimeError("Builder not initialized")
            
        # Check input directory
        if not self.builder.ctx.paths.input_dir.exists():
             logger.warning(f"Input directory does not exist: {self.builder.ctx.paths.input_dir}")
    
    def analyze(self) -> Any:
        """Analyze assets."""
        if self.builder:
            self.builder.analyze_assets()
        return None

    def process(self, analysis_result: Any) -> Any:
        """Plan montage."""
        if self.builder:
            self.builder.plan_montage()
        return None

    def render(self, processing_result: Any) -> Any:
        """Render output."""
        if self.builder:
            # Enhance if enabled (handled in builder based on ctx flags set in initialize)
            if self.builder.ctx.features.stabilize or self.builder.ctx.features.upscale or self.builder.ctx.features.enhance:
                self.builder.enhance_assets()
            
            self.builder.render_output()
        return None

    def export(self, render_result: Any) -> str:
        """Export timeline and return output path."""
        if self.builder:
            self.builder.export_timeline()
            # Save episodic memory
            self.builder._save_episodic_memory()
            
            return str(self.builder.ctx.render.output_filename)
        return ""
    
    def cleanup(self) -> None:
        """Cleanup."""
        if self.builder:
            self.builder.cleanup()

    def get_metadata(self) -> Dict[str, Any]:
        """Get montage-specific metadata."""
        base = super().get_metadata()
        base.update({
            "style": self.options.extras.get('style', 'dynamic'),
            "beat_sync": self.options.extras.get('beat_sync', True),
        })
        return base
