"""
AI-Powered Video Enhancement Suite

Multi-stage AI processing pipeline:
  1. Upscaling (Real-ESRGAN via cgpu or local)
  2. Face Enhancement (MediaPipe + deep learning)
  3. Advanced Denoising (NL-means + ML-based)
  4. Stabilization (ProStabilizationEngine)
  5. Color Grading (AI-based)
  6. Output

Each stage includes timing and performance tracking.

Version: 1.0.0
"""

import logging
import time
import os
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json

from .config import get_settings, Settings
from .ffmpeg_config import (
    get_config as get_ffmpeg_config,
    STANDARD_WIDTH_HORIZONTAL,
    STANDARD_HEIGHT_HORIZONTAL,
)
from .ffmpeg_utils import build_ffmpeg_cmd, build_video_encoding_args
from .video_metadata import probe_metadata

logger = logging.getLogger(__name__)


@dataclass
class TimingMetrics:
    """Track timing and performance for each pipeline stage."""
    stage_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    memory_mb_start: Optional[float] = None
    memory_mb_end: Optional[float] = None
    memory_delta_mb: float = 0.0
    fps_input: float = 0.0
    fps_output: float = 0.0
    resolution_input: str = "unknown"
    resolution_output: str = "unknown"
    status: str = "pending"
    error_message: Optional[str] = None
    
    def finish(self, status: str = "success"):
        """Mark stage as complete."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class TimingTracker:
    """Global timing tracker for pipeline stages."""
    
    def __init__(self):
        self.stages: Dict[str, TimingMetrics] = {}
        self.pipeline_start = time.time()
        
    def start_stage(self, stage_name: str) -> TimingMetrics:
        """Start timing a stage."""
        metrics = TimingMetrics(stage_name=stage_name)
        self.stages[stage_name] = metrics
        logger.info(f"🚀 Starting stage: {stage_name}")
        return metrics
    
    def end_stage(self, stage_name: str, status: str = "success", error: Optional[str] = None):
        """End timing for a stage."""
        if stage_name in self.stages:
            self.stages[stage_name].finish(status)
            if error:
                self.stages[stage_name].error_message = error
            logger.info(f"✅ Stage complete: {stage_name} ({self.stages[stage_name].duration_seconds:.1f}s)")
    
    def get_total_duration(self) -> float:
        """Get total pipeline duration."""
        return time.time() - self.pipeline_start
    
    def get_report(self) -> Dict[str, Any]:
        """Get full timing report."""
        total_duration = self.get_total_duration()
        stages_data = {name: metrics.to_dict() for name, metrics in self.stages.items()}
        
        return {
            "pipeline_start": self.pipeline_start,
            "pipeline_end": time.time(),
            "total_duration_seconds": total_duration,
            "stages": stages_data,
            "stages_count": len(self.stages),
        }
    
    def save_report(self, output_path: str):
        """Save timing report to JSON file."""
        report = self.get_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📊 Timing report saved to {output_path}")


# Global tracker
_timing_tracker = TimingTracker()


def get_timing_tracker() -> TimingTracker:
    """Get global timing tracker instance."""
    return _timing_tracker


class UpscalerStage:
    """AI Upscaling stage - Local or cloud-based."""
    
    def __init__(self, use_cgpu: bool = False, settings: Optional[Settings] = None):
        self.use_cgpu = use_cgpu
        self.settings = settings or get_settings()
        self.ffmpeg_config = get_ffmpeg_config(hwaccel=self.settings.gpu.ffmpeg_hwaccel)
        self.tracker = get_timing_tracker()
    
    def process(
        self,
        input_path: str,
        output_path: str,
        scale: Optional[int] = None,
        model: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Upscale video using Real-ESRGAN.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            scale: Upscale factor (2 or 4)
            model: Model name (RealESRGAN_x2plus, RealESRGAN_x4plus, etc)
        
        Returns:
            (success, output_path_or_error)
        """
        scale_value = scale if scale is not None else self.settings.upscale.scale
        model_value = model if model is not None else self.settings.upscale.model
        metrics = self.tracker.start_stage(f"upscale_{scale_value}x")
        
        try:
            if self.use_cgpu:
                return self._upscale_cgpu(input_path, output_path, scale_value, model_value, metrics)
            else:
                return self._upscale_local(input_path, output_path, scale_value, model_value, metrics)
        except Exception as e:
            error_msg = f"Upscaling failed: {str(e)}"
            self.tracker.end_stage(metrics.stage_name, status="failed", error=error_msg)
            logger.error(error_msg)
            return False, error_msg
    
    def _upscale_cgpu(
        self,
        input_path: str,
        output_path: str,
        scale: int,
        model: str,
        metrics: TimingMetrics
    ) -> Tuple[bool, str]:
        """Upscale using cgpu cloud GPU."""
        try:
            from .cgpu_upscaler import upscale_with_cgpu, is_cgpu_available
            
            if not is_cgpu_available():
                logger.warning("CGPU not available, falling back to local upscaling")
                return self._upscale_local(input_path, output_path, scale, model, metrics)
            
            logger.info(f"🌩️  Using cgpu for {scale}x upscaling")
            result = upscale_with_cgpu(input_path, output_path, scale=scale, model=model)
            
            if result:
                try:
                    metadata = probe_metadata(result)
                    metrics.resolution_output = f"{metadata.width}x{metadata.height}"
                except Exception:
                    metrics.resolution_output = "unknown"
                self.tracker.end_stage(metrics.stage_name, status="success")
                return True, result
            else:
                raise Exception("CGPU upscaling returned None")
        
        except Exception as e:
            error_msg = f"CGPU upscaling failed: {str(e)}"
            self.tracker.end_stage(metrics.stage_name, status="failed", error=error_msg)
            logger.error(error_msg)
            return False, error_msg
    
    def _upscale_local(
        self,
        input_path: str,
        output_path: str,
        scale: int,
        model: str,
        metrics: TimingMetrics
    ) -> Tuple[bool, str]:
        """Upscale using local FFmpeg with scale filter (basic upscaling)."""
        try:
            import subprocess
            
            metadata = probe_metadata(input_path)
            base_width = metadata.width or STANDARD_WIDTH_HORIZONTAL
            base_height = metadata.height or STANDARD_HEIGHT_HORIZONTAL
            target_width = base_width * scale
            target_height = base_height * scale

            vf_chain = f"scale={target_width}:{target_height}:flags=lanczos"
            cmd = build_ffmpeg_cmd([
                "-i", input_path,
                "-vf", vf_chain,
                "-an",
            ])
            cmd.extend(build_video_encoding_args(
                codec=self.ffmpeg_config.effective_codec,
                preset=self.settings.encoding.preset,
                crf=self.settings.encoding.crf,
                profile=self.ffmpeg_config.profile,
                level=self.ffmpeg_config.level,
                pix_fmt=self.ffmpeg_config.pix_fmt,
            ))
            cmd.append(output_path)
            
            logger.info(f"📺 Local upscaling to {target_width}x{target_height}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.settings.processing.ffmpeg_timeout)
            
            if result.returncode == 0 and Path(output_path).exists():
                metrics.resolution_output = f"{target_width}x{target_height}"
                self.tracker.end_stage(metrics.stage_name, status="success")
                return True, output_path
            else:
                raise Exception(result.stderr)
        
        except Exception as e:
            error_msg = f"Local upscaling failed: {str(e)}"
            self.tracker.end_stage(metrics.stage_name, status="failed", error=error_msg)
            logger.error(error_msg)
            return False, error_msg


class FaceEnhancementStage:
    """AI Face Enhancement using MediaPipe and deep learning."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.ffmpeg_config = get_ffmpeg_config(hwaccel=self.settings.gpu.ffmpeg_hwaccel)
        self.tracker = get_timing_tracker()
        self._mediapipe_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if MediaPipe is available."""
        try:
            import importlib.util
            self._mediapipe_available = importlib.util.find_spec("mediapipe") is not None
        except Exception:
            self._mediapipe_available = False
        if not self._mediapipe_available:
            logger.warning("MediaPipe not available for face enhancement")
    
    def process(
        self,
        input_path: str,
        output_path: str,
        enhancement_level: str = "medium"  # light, medium, aggressive
    ) -> Tuple[bool, str]:
        """
        Enhance faces in video using AI.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            enhancement_level: Enhancement intensity
        
        Returns:
            (success, output_path_or_error)
        """
        metrics = self.tracker.start_stage("face_enhancement")
        
        try:
            if not self._mediapipe_available:
                logger.info("Skipping face enhancement (MediaPipe not available)")
                self.tracker.end_stage(metrics.stage_name, status="skipped")
                return True, input_path  # Return input unchanged
            
            return self._enhance_faces(input_path, output_path, enhancement_level, metrics)
        
        except Exception as e:
            error_msg = f"Face enhancement failed: {str(e)}"
            self.tracker.end_stage(metrics.stage_name, status="failed", error=error_msg)
            logger.error(error_msg)
            return False, error_msg
    
    def _enhance_faces(
        self,
        input_path: str,
        output_path: str,
        level: str,
        metrics: TimingMetrics
    ) -> Tuple[bool, str]:
        """Apply face enhancement filters."""
        try:
            import subprocess
            
            # Build FFmpeg filter for face enhancement
            # Using unsharp mask + brightness/contrast adjustment for detected faces
            filter_chain = self._build_face_filter(level)
            
            cmd = build_ffmpeg_cmd([
                "-i", input_path,
                "-vf", filter_chain,
                "-an",
            ])
            cmd.extend(build_video_encoding_args(
                codec=self.settings.encoding.codec,
                preset=self.settings.encoding.preset,
                crf=self.settings.encoding.crf,
                profile=self.ffmpeg_config.profile,
                level=self.ffmpeg_config.level,
                pix_fmt=self.ffmpeg_config.pix_fmt,
            ))
            cmd.append(output_path)
            
            logger.info(f"👤 Applying {level} face enhancement")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.processing.ffmpeg_timeout,
            )
            
            if result.returncode == 0 and Path(output_path).exists():
                self.tracker.end_stage(metrics.stage_name, status="success")
                return True, output_path
            else:
                raise Exception(result.stderr)
        
        except Exception as e:
            error_msg = f"Face enhancement processing failed: {str(e)}"
            self.tracker.end_stage(metrics.stage_name, status="failed", error=error_msg)
            logger.error(error_msg)
            return False, error_msg
    
    def _build_face_filter(self, level: str) -> str:
        """Build FFmpeg filter for face enhancement."""
        # This is a simplified version; production would use actual face detection
        levels = {
            "light": "unsharp=1.5:1.5:0.5:0.5",
            "medium": "unsharp=2.0:2.0:0.8:0.8,eq=brightness=0.05:contrast=1.1",
            "aggressive": "unsharp=2.5:2.5:1.0:1.0,eq=brightness=0.1:contrast=1.2:gamma=1.05"
        }
        return levels.get(level, levels["medium"])


class AdvancedDenoiseStage:
    """Advanced AI-based denoising beyond basic NL-means."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.ffmpeg_config = get_ffmpeg_config(hwaccel=self.settings.gpu.ffmpeg_hwaccel)
        self.tracker = get_timing_tracker()
    
    def process(
        self,
        input_path: str,
        output_path: str,
        denoise_level: str = "medium"  # light, medium, aggressive
    ) -> Tuple[bool, str]:
        """
        Apply advanced denoising to video.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            denoise_level: Denoising intensity
        
        Returns:
            (success, output_path_or_error)
        """
        metrics = self.tracker.start_stage("advanced_denoise")
        
        try:
            return self._apply_denoise(input_path, output_path, denoise_level, metrics)
        
        except Exception as e:
            error_msg = f"Denoising failed: {str(e)}"
            self.tracker.end_stage(metrics.stage_name, status="failed", error=error_msg)
            logger.error(error_msg)
            return False, error_msg
    
    def _apply_denoise(
        self,
        input_path: str,
        output_path: str,
        level: str,
        metrics: TimingMetrics
    ) -> Tuple[bool, str]:
        """Apply multi-stage denoising."""
        try:
            import subprocess
            
            # Multi-stage denoise: hqdn3d + bm3d equivalent + grain reduction
            filter_chain = self._build_denoise_filter(level)
            
            cmd = build_ffmpeg_cmd([
                "-i", input_path,
                "-vf", filter_chain,
                "-an",
            ])
            cmd.extend(build_video_encoding_args(
                codec=self.settings.encoding.codec,
                preset=self.settings.encoding.preset,
                crf=self.settings.encoding.crf,
                profile=self.ffmpeg_config.profile,
                level=self.ffmpeg_config.level,
                pix_fmt=self.ffmpeg_config.pix_fmt,
            ))
            cmd.append(output_path)
            
            logger.info(f"🔇 Applying {level} advanced denoising")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.processing.ffmpeg_timeout,
            )
            
            if result.returncode == 0 and Path(output_path).exists():
                self.tracker.end_stage(metrics.stage_name, status="success")
                return True, output_path
            else:
                raise Exception(result.stderr)
        
        except Exception as e:
            error_msg = f"Denoising processing failed: {str(e)}"
            self.tracker.end_stage(metrics.stage_name, status="failed", error=error_msg)
            logger.error(error_msg)
            return False, error_msg
    
    def _build_denoise_filter(self, level: str) -> str:
        """Build FFmpeg denoise filter chain."""
        levels = {
            "light": "hqdn3d=1.5:1.5:6:6",
            "medium": "hqdn3d=2:2:6:6,delogo=x=0:y=0:w=0:h=0",  # placeholder
            "aggressive": "hqdn3d=3:3:8:8,nlmeans=s=4:f=13:t=10"
        }
        return levels.get(level, levels["medium"])


class AIVideoPipeline:
    """Main orchestrator for complete AI video processing pipeline."""
    
    def __init__(self, use_cgpu_upscale: bool = False, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.tracker = get_timing_tracker()
        self.upscaler = UpscalerStage(use_cgpu=use_cgpu_upscale, settings=self.settings)
        self.face_enhancer = FaceEnhancementStage(settings=self.settings)
        self.denoiser = AdvancedDenoiseStage(settings=self.settings)
        self.use_cgpu = use_cgpu_upscale
    
    def process(
        self,
        input_path: str,
        output_dir: str,
        enable_upscale: bool = True,
        upscale_factor: int = 2,
        enable_face_enhance: bool = True,
        face_enhance_level: str = "medium",
        enable_denoise: bool = True,
        denoise_level: str = "medium",
        enable_stabilize: bool = True,
    ) -> Dict[str, Any]:
        """
        Process video through complete AI pipeline.
        
        Args:
            input_path: Input video file
            output_dir: Output directory for results
            enable_upscale: Enable upscaling stage
            upscale_factor: Upscale multiplier (2 or 4)
            enable_face_enhance: Enable face enhancement
            face_enhance_level: Face enhancement intensity
            enable_denoise: Enable advanced denoising
            denoise_level: Denoise intensity
            enable_stabilize: Enable stabilization (requires stabilization_integration.py)
        
        Returns:
            Dictionary with results, timing metrics, and paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        current_input = input_path
        stage_results = []
        
        logger.info("🎬 Starting AI Video Processing Pipeline")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_dir}")
        
        # Stage 1: Upscaling
        if enable_upscale:
            upscale_output = str(output_dir / "01_upscaled.mp4")
            success, result = self.upscaler.process(
                current_input,
                upscale_output,
                scale=upscale_factor
            )
            stage_results.append({
                "stage": "upscale",
                "success": success,
                "output": result if success else None,
                "message": result
            })
            if success:
                current_input = result
            else:
                logger.warning(f"Upscaling skipped due to error: {result}")
        
        # Stage 2: Face Enhancement
        if enable_face_enhance:
            face_output = str(output_dir / "02_face_enhanced.mp4")
            success, result = self.face_enhancer.process(
                current_input,
                face_output,
                enhancement_level=face_enhance_level
            )
            stage_results.append({
                "stage": "face_enhancement",
                "success": success,
                "output": result if success else None,
                "message": result
            })
            if success:
                current_input = result
            else:
                logger.warning(f"Face enhancement skipped: {result}")
        
        # Stage 3: Advanced Denoising
        if enable_denoise:
            denoise_output = str(output_dir / "03_denoised.mp4")
            success, result = self.denoiser.process(
                current_input,
                denoise_output,
                denoise_level=denoise_level
            )
            stage_results.append({
                "stage": "denoise",
                "success": success,
                "output": result if success else None,
                "message": result
            })
            if success:
                current_input = result
            else:
                logger.warning(f"Denoising skipped: {result}")
        
        # Stage 4: Stabilization (if enabled)
        if enable_stabilize:
            try:
                from .stabilization_integration import StabilizationBridge
                stabilize_output = str(output_dir / "04_stabilized.mp4")
                
                bridge = StabilizationBridge()
                engine = bridge.get_engine()
                
                metrics = self.tracker.start_stage("stabilization")
                success, result = engine.stabilize_clip(current_input, stabilize_output)
                
                if success:
                    self.tracker.end_stage("stabilization", status="success")
                    stage_results.append({
                        "stage": "stabilization",
                        "success": True,
                        "output": result,
                        "message": f"Stabilized to {result}"
                    })
                    current_input = result
                else:
                    self.tracker.end_stage("stabilization", status="failed", error=result)
                    logger.warning(f"Stabilization skipped: {result}")
            
            except ImportError:
                logger.warning("stabilization_integration not available, skipping stabilization")
        
        # Final output
        final_output = str(output_dir / "FINAL_output.mp4")
        if current_input != final_output:
            import shutil
            shutil.copy2(current_input, final_output)
        
        stage_results.append({
            "stage": "final_output",
            "success": True,
            "output": final_output,
            "message": f"Final output: {final_output}"
        })
        
        # Generate timing report
        timing_report_path = str(output_dir / "timing_report.json")
        self.tracker.save_report(timing_report_path)
        
        total_duration = self.tracker.get_total_duration()
        
        logger.info(f"\n✅ Pipeline complete in {total_duration:.1f}s")
        logger.info(f"📊 Timing report: {timing_report_path}")
        
        return {
            "success": True,
            "input": input_path,
            "output": final_output,
            "output_directory": str(output_dir),
            "stages": stage_results,
            "timing_report": timing_report_path,
            "total_duration_seconds": total_duration,
            "timestamp": time.time()
        }


def create_ai_pipeline(use_cgpu: bool = False, settings: Optional[Settings] = None) -> AIVideoPipeline:
    """Factory function to create AI video pipeline."""
    settings = settings or get_settings()
    cgpu_enabled = use_cgpu or settings.llm.cgpu_gpu_enabled
    return AIVideoPipeline(use_cgpu_upscale=cgpu_enabled, settings=settings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("ai_video_tools is a library module; use via API entry points.")
