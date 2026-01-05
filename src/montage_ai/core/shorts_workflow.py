"""
Shorts Studio Workflow - Concrete Implementation

Converts horizontal video to vertical (9:16) with smart reframing and captions.
"""

from typing import Any, Optional, Dict
from pathlib import Path

from .workflow import VideoWorkflow, WorkflowOptions, WorkflowPhase
from ..auto_reframe import AutoReframeEngine, CropWindow
from ..transcriber import transcribe_audio
from ..caption_burner import CaptionBurner, CaptionStyle
from ..logger import logger


class ShortsWorkflow(VideoWorkflow):
    """
    Shorts Studio workflow implementation.
    
    Pipeline:
    1. Initialize: Setup reframer
    2. Validate: Check video exists and is suitable
    3. Analyze: Face detection + subject tracking
    4. Process: Calculate crop windows
    5. Render: Apply reframing
    6. Export: Add captions (if requested)
    """
    
    def __init__(self, options: WorkflowOptions):
        super().__init__(options)
        self.reframer: Optional[AutoReframeEngine] = None
        self.crop_data: Optional[list] = None
        self.reframed_path: Optional[Path] = None
    
    @property
    def workflow_name(self) -> str:
        return "Shorts Studio"
    
    @property
    def workflow_type(self) -> str:
        return "shorts"
    
    # =========================================================================
    # Workflow Steps
    # =========================================================================
    
    def initialize(self) -> None:
        """Initialize reframer engine."""
        self.reframer = AutoReframeEngine(target_aspect=9/16)
        logger.debug("AutoReframeEngine initialized")
    
    def validate(self) -> None:
        """Validate input video exists."""
        input_path = Path(self.options.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Check file size (basic sanity check)
        size_mb = input_path.stat().st_size / (1024 * 1024)
        if size_mb < 0.1:
            raise ValueError(f"Input video too small: {size_mb:.2f} MB")
        
        logger.debug(f"Input validated: {size_mb:.1f} MB")
    
    def analyze(self) -> Optional[list]:
        """Analyze video for smart reframing."""
        reframe_mode = self.options.extras.get('reframe_mode', 'auto')
        
        if reframe_mode == 'center':
            logger.info("Center crop mode - skipping analysis")
            return None
        
        logger.info("Analyzing video for subject tracking...")
        self._update_progress(20, "Detecting faces...")
        
        self.crop_data = self.reframer.analyze(self.options.input_path)
        
        logger.info(f"Analysis complete: {len(self.crop_data)} frames")
        return self.crop_data
    
    def process(self, analysis_result: Any) -> str:
        """Process = Apply reframing."""
        self._update_progress(40, "Applying smart reframe...")
        
        timestamp = self.options.job_id
        self.reframed_path = Path(self.options.output_dir) / f"shorts_reframed_{timestamp}.mp4"
        
        logger.info(f"Reframing to: {self.reframed_path}")
        self.reframer.apply(
            self.crop_data,
            self.options.input_path,
            str(self.reframed_path)
        )
        
        return str(self.reframed_path)
    
    def render(self, processing_result: Any) -> str:
        """Render = Transcription (if captions enabled)."""
        add_captions = self.options.extras.get('add_captions', True)
        
        if not add_captions:
            logger.info("Captions disabled, skipping transcription")
            return processing_result
        
        self._update_progress(60, "Transcribing audio...")
        
        logger.info("Running Whisper transcription...")
        transcript = transcribe_audio(
            processing_result,
            model='base',
            word_timestamps=True
        )
        
        logger.info(f"Transcribed {len(transcript.get('segments', []))} segments")
        return (processing_result, transcript)
    
    def export(self, render_result: Any) -> str:
        """Export = Burn captions and finalize."""
        add_captions = self.options.extras.get('add_captions', True)
        
        if not add_captions:
            # No captions - reframed video is final
            return render_result
        
        self._update_progress(80, "Burning captions...")
        
        reframed_video, transcript = render_result
        
        # Generate SRT
        srt_path = Path(reframed_video).with_suffix('.srt')
        self._generate_srt(transcript, str(srt_path))
        
        # Final output
        timestamp = self.options.job_id
        output_path = Path(self.options.output_dir) / f"shorts_{timestamp}.mp4"
        
        # Burn captions
        caption_style = self.options.extras.get('caption_style', 'tiktok')
        style_map = {
            'default': CaptionStyle.TIKTOK,
            'tiktok': CaptionStyle.TIKTOK,
            'bold': CaptionStyle.BOLD,
            'minimal': CaptionStyle.MINIMAL,
            'gradient': CaptionStyle.KARAOKE,
            'karaoke': CaptionStyle.KARAOKE,
        }
        burner_style = style_map.get(caption_style, CaptionStyle.TIKTOK)
        
        logger.info(f"Burning captions ({caption_style} style)...")
        burner = CaptionBurner(style=burner_style)
        burner.burn(reframed_video, str(srt_path), str(output_path))
        
        # Cleanup SRT
        srt_path.unlink(missing_ok=True)
        
        return str(output_path)
    
    def cleanup(self) -> None:
        """Cleanup intermediate files."""
        if self.reframed_path and self.reframed_path.exists():
            add_captions = self.options.extras.get('add_captions', True)
            if add_captions:
                # Only delete if we created a captioned version
                logger.debug(f"Cleaning up: {self.reframed_path}")
                self.reframed_path.unlink(missing_ok=True)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get Shorts-specific metadata."""
        base = super().get_metadata()
        base.update({
            "reframe_mode": self.options.extras.get('reframe_mode', 'auto'),
            "caption_style": self.options.extras.get('caption_style', 'tiktok'),
            "add_captions": self.options.extras.get('add_captions', True),
            "platform": self.options.extras.get('platform', 'tiktok'),
        })
        return base
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _generate_srt(self, transcript: dict, srt_path: str) -> None:
        """Generate SRT file from Whisper transcript."""
        segments = transcript.get('segments', [])
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                text = seg.get('text', '').strip()
                
                # Format timecodes: HH:MM:SS,mmm
                def tc(seconds):
                    h = int(seconds // 3600)
                    m = int((seconds % 3600) // 60)
                    s = int(seconds % 60)
                    ms = int((seconds - int(seconds)) * 1000)
                    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                
                f.write(f"{i}\n")
                f.write(f"{tc(start)} --> {tc(end)}\n")
                f.write(f"{text}\n\n")
