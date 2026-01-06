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
from ..timeline_exporter import TimelineExporter, Timeline, Clip


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
            self.audio_highlights = self._analyze_smart_highlights()
            logger.info(f"Audio analysis complete: {len(self.audio_highlights)} highlights")

        return self.crop_data

    def _analyze_smart_highlights(self) -> List[Dict[str, Any]]:
        """Detect highlights using multi-modal signals (Audio + Face + Action)."""
        try:
            import numpy as np
            from ..audio_analysis import analyze_music_energy, EnergyProfile
            from ..scene_analysis import Scene, SceneAnalysis
            from .highlight_detector import HighlightDetector

            # 1. Get Audio Energy
            _, energy_profile = analyze_music_energy(self.options.input_path)
            
            # 2. visual Analysis (simulate Scenes from Crop Data)
            # If we don't have crop data (e.g. center crop mode), we need to generate dummy scenes
            scenes = self._create_scenes_from_crops(self.crop_data) if self.crop_data else []
            
            if not scenes:
                # If no visual data, create dummy scenes based on audio duration
                max_time = energy_profile.times[-1] if len(energy_profile.times) > 0 else 60.0
                scenes = [
                    Scene(start=i, end=min(i+3.0, max_time), path="", meta={"analysis": SceneAnalysis(description="", face_count=0, action="MEDIUM", quality="YES", shot="medium")})
                    for i in np.arange(0, max_time, 3.0)
                ]

            # 3. Detect
            detector = HighlightDetector(scenes, energy_profile)
            results = detector.detect(top_k=10)
            
            return [
                {
                    "start": r.start, 
                    "end": r.end, 
                    "duration": r.end - r.start,
                    "score": r.score,
                    "type": "smart_highlight",
                    "signals": r.signals
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Highlight detection failed: {e}", exc_info=True)
            return []

    def _create_scenes_from_crops(self, crop_data: List[Any], step_sec: float = 3.0) -> List[Any]:
        """Convert frame-level crop data into Scene objects."""
        from ..scene_analysis import Scene, SceneAnalysis
        import numpy as np
        
        if not crop_data:
            return []
            
        total_time = crop_data[-1].time
        scenes = []
        
        # Split into fixed windows
        for t in np.arange(0, total_time, step_sec):
            start = float(t)
            end = min(float(t + step_sec), total_time)
            
            # Find crops in this window
            # Assumes crop_data is sorted by time
            window_crops = [c for c in crop_data if start <= c.time < end]
            
            if not window_crops:
                continue
                
            # Compute aggregate stats
            avg_score = np.mean([c.score for c in window_crops])
            # std_x = np.std([c.x for c in window_crops]) # Maybe proxy for action?
            
            # Map to SceneAnalysis
            # If avg_score (face confidence) is high, face_count = 1
            face_count = 1 if avg_score > 0.6 else 0
            
            # Detect "Action" via motion? For now default to MEDIUM.
            action = "MEDIUM"
            
            analysis = SceneAnalysis(
                description="Auto-generated segment",
                face_count=face_count,
                action=action,
                quality="YES",
                tags=[],
                mood="neutral",
                shot="medium"
            )
            
            scenes.append(Scene(start=start, end=end, path="", meta={"analysis": analysis}))
            
        return scenes
    
    def process(self, analysis_result: Any) -> str:
        """Process = Apply reframing."""
        self._update_progress(40, "Applying smart reframe...")
        
        timestamp = self.options.job_id
        self.reframed_path = Path(self.options.output_dir) / f"shorts_reframed_{timestamp}.mp4"
        
        logger.info(f"Reframing to: {self.reframed_path}")
        
        # Performance Tuning: Use preview preset if this is a preview?
        # The user/UI doesn't explicitly send "preview" mode for shorts yet, but 
        # config.quality_profile might be set.
        from ..config import get_settings
        settings = get_settings()
        # Default to high quality unless env var set or passed (TODO: pass from options)
        # Using settings.ffmpeg.preset if available or hardcoded "medium"
        
        self.reframer.apply(
            self.crop_data,
            self.options.input_path,
            str(self.reframed_path),
            preset="fast" # Use fast for shorts to ensure responsiveness (<3min goal)
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
        
        # --- AUDIO POLISH ---
        clean_audio = self.options.extras.get('clean_audio', False)
        if clean_audio:
            logger.info("Applying Audio Polish (Voice Isolation)...")
            from ..audio_enhancer import AudioEnhancer
            enhancer = AudioEnhancer()
            polished_path = Path(output_path).with_suffix('.polished.mp4')
            
            # Since we have video+audio in output_path, we need to process its audio
            # and replace it. Doing this IN-PLACE is tricky.
            # Simpler: Process output_path -> polished_path
            # But wait, audio processing usually happens BEFORE caption burn in NLEs.
            # Here video is already rendered. It's fine to clean audio of the final render.
            
            # Extract audio -> Enhance -> Replace
            try:
                # We can do this in one pass if we are clever, but let's be safe.
                # Just run the enhance_voice filter on the video file audio stream.
                success = enhancer.enhance_voice(str(output_path), str(polished_path))
                if success and polished_path.exists():
                    # Replace original
                    output_path.unlink()
                    polished_path.rename(output_path)
            except Exception as e:
                logger.warning(f"Audio Polish failed: {e}")

        # --- CAPTIONS ---
        style_map = {
            'default': CaptionStyle.TIKTOK,
            'tiktok': CaptionStyle.TIKTOK,
            'bold': CaptionStyle.BOLD,
            'minimal': CaptionStyle.MINIMAL,
            'gradient': CaptionStyle.KARAOKE,
            'karaoke': CaptionStyle.KARAOKE,
            'cinematic': CaptionStyle.CINEMATIC,
            'youtube': CaptionStyle.YOUTUBE,
        }
        burner_style = style_map.get(caption_style.lower(), CaptionStyle.TIKTOK)
        
        logger.info(f"Burning captions ({caption_style} style)...")
        burner = CaptionBurner(style=burner_style)
        burner.burn(reframed_video, str(srt_path), str(output_path))
        
        # Cleanup SRT
        srt_path.unlink(missing_ok=True)

        # --- OTIO/EDL EXPORT ---
        try:
            self._update_progress(95, "Exporting timeline...")
            from ..utils import get_video_duration

            final_path_str = str(output_path)
            duration = get_video_duration(final_path_str)

            # Create a simple timeline with one clip (the result)
            timeline = Timeline(
                clips=[Clip(
                    source_path=final_path_str,
                    start_time=0.0,
                    duration=duration,
                    timeline_start=0.0
                )],
                audio_path=final_path_str,
                total_duration=duration,
                resolution=(1080, 1920),
                project_name=output_path.stem
            )

            exporter = TimelineExporter(output_dir=self.options.output_dir)
            
            # Check if user wanted proxies (passed in options or default to True for "Pro"?)
            # Let's assume True if quality is HIGH, or just default to False unless specified.
            generate_proxies = self.options.extras.get('generate_proxies', False)
            
            exporter.export_timeline(
                timeline,
                generate_proxies=generate_proxies,
                link_to_source=True,
                export_otio=True,
                export_edl=True
            )
            logger.info("OTIO/EDL export complete")

        except Exception as e:
            logger.warning(f"OTIO Export failed: {e}")

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
            "highlights": self.audio_highlights,
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
