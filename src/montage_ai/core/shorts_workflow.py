"""
Shorts Studio Workflow - Concrete Implementation

Converts horizontal video to vertical (9:16) with smart reframing and captions.

Features:
- Smart subject tracking (face detection + Kalman smoothing)
- Audio-aware highlight detection
- Caption burning with multiple styles
- Voice isolation & noise reduction (Clean Audio)
"""

from typing import Any, Optional, Dict, List
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
    3. Analyze: Face detection + subject tracking + audio analysis
    4. Process: Calculate crop windows + detect highlights
    5. Render: Apply reframing
    6. Export: Add captions (if requested)
    """

    def __init__(self, options: WorkflowOptions):
        super().__init__(options)
        self.reframer: Optional[AutoReframeEngine] = None
        self.crop_data: Optional[list] = None
        self.reframed_path: Optional[Path] = None
        self.audio_highlights: List[Dict[str, Any]] = []
    
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
        """Analyze video for smart reframing + audio highlights."""
        reframe_mode = self.options.extras.get('reframe_mode', 'auto')
        audio_aware = self.options.extras.get('audio_aware', False)

        # 1. Video Analysis (Reframing)
        if reframe_mode == 'center':
            logger.info("Center crop mode - skipping video analysis")
        else:
            logger.info("Analyzing video for subject tracking...")
            self._update_progress(20, "Detecting faces...")
            self.crop_data = self.reframer.analyze(self.options.input_path)
            logger.info(f"Video analysis complete: {len(self.crop_data)} frames")

        # 2. Audio Analysis (Highlights)
        if audio_aware:
            self._update_progress(30, "Analyzing audio for highlights...")
            self.audio_highlights = self._analyze_audio_highlights()
            logger.info(f"Audio analysis complete: {len(self.audio_highlights)} highlights")

        return self.crop_data

    def _analyze_audio_highlights(self) -> List[Dict[str, Any]]:
        """Detect audio-based highlight moments."""
        try:
            import numpy as np
            from ..audio_analysis import AudioAnalyzer

            analyzer = AudioAnalyzer(self.options.input_path)
            analyzer.analyze()

            highlights = []
            energy = analyzer.energy_curve if hasattr(analyzer, 'energy_curve') else []
            beats = analyzer.beat_times if hasattr(analyzer, 'beat_times') else []

            if len(energy) == 0:
                return []

            # Find high-energy regions
            threshold = np.percentile(energy, 85)
            hop_time = getattr(analyzer, 'hop_length', 512) / getattr(analyzer, 'sr', 22050)

            in_highlight = False
            start_time = 0.0
            peak_energy = 0.0

            for i, e in enumerate(energy):
                time = i * hop_time
                if e > threshold and not in_highlight:
                    in_highlight = True
                    start_time = max(0, time - 0.5)
                    peak_energy = e
                elif in_highlight:
                    peak_energy = max(peak_energy, e)
                    if e <= threshold * 0.8:
                        in_highlight = False
                        duration = time - start_time
                        if 2.0 <= duration <= 60.0:
                            score = min(1.0, peak_energy / (np.max(energy) + 0.001))
                            highlights.append({
                                "start": start_time,
                                "end": time,
                                "duration": duration,
                                "score": score,
                                "type": "energy"
                            })

            # Sort by score and return top highlights
            highlights.sort(key=lambda x: x['score'], reverse=True)
            return highlights[:10]

        except Exception as e:
            logger.warning(f"Audio highlight detection failed: {e}")
            return []
    
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
