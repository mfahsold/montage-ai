"""
Wan2.1-VACE Integration - Video Generation & Editing via cgpu

Based on: Alibaba Wan2.1 Video Generation Model
Repository: github.com/Wan-Video/Wan2.1
License: Apache-2.0

Features:
- Text-to-Video (T2V) Generation
- Image-to-Video (I2V) Extension  
- Video Inpainting/Editing (VACE)
- Reference-based Style Transfer

Model Variants:
- Wan2.1-T2V-1.3B: 8GB VRAM, 480p output (recommended for cgpu)
- Wan2.1-T2V-14B: 24GB+ VRAM, 720p output (requires A100)
- Wan2.1-VACE-1.3B: Video editing/inpainting capabilities

Architecture:
    Text Prompt → Wan2.1 Model (via cgpu) → Generated Video
    
Integration with Montage AI:
- B-Roll generation for transitions
- Missing footage creation
- Style-matched video generation
- Video extension/continuation

Version: 1.1.0 - Uses shared cgpu_utils module
"""

import os
import json
import base64
import tempfile
import subprocess
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum

# Import shared cgpu utilities
from .cgpu_utils import (
    CGPUConfig,
    is_cgpu_available,
    run_cgpu_command,
    cgpu_copy_to_remote,
    parse_base64_output,
)


class WanModelSize(Enum):
    """Available Wan2.1 model sizes."""
    SMALL = "1.3B"   # 8GB VRAM, faster, lower quality
    LARGE = "14B"    # 24GB+ VRAM, slower, higher quality


class WanTaskType(Enum):
    """Wan2.1 task types."""
    TEXT_TO_VIDEO = "t2v"
    IMAGE_TO_VIDEO = "i2v"
    VIDEO_INPAINTING = "inpaint"
    VIDEO_EXTENSION = "extend"


@dataclass
class WanConfig:
    """Configuration for Wan2.1-VACE service."""
    model_size: WanModelSize = WanModelSize.SMALL
    resolution: str = "480p"  # "480p" or "720p"
    num_frames: int = 81      # ~3.4 seconds at 24fps
    fps: int = 24
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    use_cgpu: bool = True
    
    # cgpu settings
    cgpu_timeout: int = 600   # 10 minutes for generation
    
    @property
    def width(self) -> int:
        return 854 if self.resolution == "480p" else 1280
    
    @property
    def height(self) -> int:
        return 480 if self.resolution == "480p" else 720


class WanVACEService:
    """
    Wan2.1-VACE Video Generation Service.
    
    Executes Wan2.1 model on Google Colab via cgpu for:
    - B-Roll generation from text prompts
    - Video inpainting (object removal/replacement)
    - Video extension/continuation
    - Style transfer
    
    Usage:
        service = WanVACEService()
        
        # Generate B-Roll
        video_path = service.generate_broll(
            prompt="Cinematic drone shot over forest at sunset",
            duration=3.0
        )
        
        # Extend existing video
        extended = service.extend_video(
            video_path="/data/input/clip.mp4",
            prompt="Continue with same style",
            extend_seconds=2.0
        )
    """
    
    # Colab setup script for Wan2.1
    COLAB_SETUP_SCRIPT = '''
import subprocess
import sys

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
    "torch", "torchvision", "transformers", "accelerate", "diffusers>=0.28.0",
    "opencv-python", "einops", "decord"], check=True)

# Clone Wan2.1 if not exists
import os
if not os.path.exists("Wan2.1"):
    subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.1.git"], check=True)
    
os.chdir("Wan2.1")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

print("Wan2.1 setup complete!")
'''

    # Generation script template
    GENERATION_SCRIPT = '''
import torch
import os
import sys
import base64

# Add Wan2.1 to path
sys.path.insert(0, "Wan2.1")

from wan.pipeline import WanPipeline

# Configuration
PROMPT = """{prompt}"""
NEGATIVE_PROMPT = """{negative_prompt}"""
WIDTH = {width}
HEIGHT = {height}
NUM_FRAMES = {num_frames}
FPS = {fps}
GUIDANCE_SCALE = {guidance_scale}
NUM_STEPS = {num_inference_steps}
OUTPUT_PATH = "/tmp/wan_output.mp4"

# Load model
print("Loading Wan2.1 model...")
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-{model_size}",
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()  # Save VRAM

print(f"Generating video: {{PROMPT[:50]}}...")

# Generate
video = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_frames=NUM_FRAMES,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=NUM_STEPS,
).frames[0]

# Save
from diffusers.utils import export_to_video
export_to_video(video, OUTPUT_PATH, fps=FPS)

print(f"Video saved to {{OUTPUT_PATH}}")

# Output as base64 for download
with open(OUTPUT_PATH, "rb") as f:
    video_data = base64.b64encode(f.read()).decode()
    
# Print marker for parsing
print("===VIDEO_BASE64_START===")
print(video_data)
print("===VIDEO_BASE64_END===")
'''

    def __init__(self, config: Optional[WanConfig] = None):
        self.config = config or WanConfig()
        self._cgpu_setup_done = False
        self._check_environment()
    
    def _check_environment(self):
        """Check if cgpu is available and configured using shared utils."""
        self.cgpu_available = is_cgpu_available()
    
    def _run_cgpu(self, script: str, timeout: Optional[int] = None) -> str:
        """Execute Python script on cgpu cloud GPU using shared utils."""
        timeout = timeout or self.config.cgpu_timeout
        
        # Create temp script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            # Copy script to Colab using shared utils
            if not cgpu_copy_to_remote(script_path, "/tmp/wan_script.py"):
                raise RuntimeError("Failed to copy script to Colab")
            
            # Run script using shared utils
            success, stdout, stderr = run_cgpu_command(
                "python /tmp/wan_script.py",
                timeout=timeout
            )
            
            if not success:
                # Include both stdout and stderr in error for debugging
                error_msg = f"Script execution failed.\nSTDERR: {stderr}\nSTDOUT: {stdout}"
                print(f"❌ cgpu execution error:\n{error_msg}")
                raise RuntimeError(error_msg)
            
            return stdout
            
        finally:
            os.unlink(script_path)
    
    def _setup_cgpu_environment(self):
        """Setup Wan2.1 environment on cgpu if not done."""
        if self._cgpu_setup_done:
            return
        
        print("Setting up Wan2.1 on cgpu...")
        output = self._run_cgpu(self.COLAB_SETUP_SCRIPT, timeout=300)
        
        if "setup complete" in output.lower():
            self._cgpu_setup_done = True
            print("Wan2.1 environment ready.")
        else:
            raise RuntimeError(f"Failed to setup Wan2.1: {output}")
    
    def _parse_base64_output(self, output: str) -> Optional[bytes]:
        """Extract base64 video data from cgpu output using shared utility."""
        return parse_base64_output(
            output, 
            "===VIDEO_BASE64_START===",
            "===VIDEO_BASE64_END==="
        )
    
    def generate_broll(
        self,
        prompt: str,
        duration: float = 3.0,
        negative_prompt: str = "blurry, low quality, distorted, ugly",
        reference_image: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate B-Roll video from text prompt.
        
        Uses Wan2.1-T2V model to generate video matching the prompt.
        Generated at native resolution, can be upscaled with cgpu_upscaler.
        
        Args:
            prompt: Description of desired video content
            duration: Video duration in seconds (max ~3.5s for 1.3B model)
            negative_prompt: What to avoid in generation
            reference_image: Optional style reference (for I2V)
            output_path: Where to save video (auto-generated if None)
            
        Returns:
            Path to generated video file
        """
        if not self.cgpu_available:
            raise RuntimeError("cgpu GPU not available. Set CGPU_GPU_ENABLED=true")
        
        # Setup environment first time
        self._setup_cgpu_environment()
        
        # Calculate frames for duration
        num_frames = min(int(duration * self.config.fps), 81)  # Max 81 frames for 1.3B
        
        # Generate output path
        if output_path is None:
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            output_path = f"/tmp/wan_broll_{prompt_hash}.mp4"
        
        # Build generation script
        script = self.GENERATION_SCRIPT.format(
            prompt=prompt.replace('"', '\\"'),
            negative_prompt=negative_prompt.replace('"', '\\"'),
            width=self.config.width,
            height=self.config.height,
            num_frames=num_frames,
            fps=self.config.fps,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            model_size=self.config.model_size.value
        )
        
        print(f"Generating B-Roll: '{prompt[:50]}...'")
        print(f"Duration: {duration}s, Resolution: {self.config.resolution}")
        
        # Run on cgpu
        output = self._run_cgpu(script)
        
        # Parse and save video
        video_data = self._parse_base64_output(output)
        
        if video_data is None:
            raise RuntimeError(f"Failed to generate video. Output: {output[:500]}")
        
        with open(output_path, "wb") as f:
            f.write(video_data)
        
        print(f"B-Roll saved to: {output_path}")
        return output_path
    
    def extend_video(
        self,
        video_path: str,
        prompt: str,
        extend_seconds: float = 2.0,
        output_path: Optional[str] = None
    ) -> str:
        """
        Extend existing video with generated continuation.
        
        Uses last frame as I2V reference to continue the video
        with matching style and content.
        
        Args:
            video_path: Path to video to extend
            prompt: Description of continuation
            extend_seconds: How much to add
            output_path: Where to save result
            
        Returns:
            Path to extended video
        """
        # TODO: Implement I2V extension
        # 1. Extract last frame
        # 2. Use I2V mode with frame as reference
        # 3. Concatenate original + generated
        
        raise NotImplementedError("Video extension coming in next sprint")
    
    def inpaint_video(
        self,
        video_path: str,
        mask_path: str,
        prompt: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Inpaint masked regions of video.
        
        Uses Wan2.1-VACE model to fill in or replace
        masked areas with content matching the prompt.
        
        Args:
            video_path: Input video
            mask_path: Mask video/image (white = areas to inpaint)
            prompt: What to fill in
            output_path: Where to save result
            
        Returns:
            Path to inpainted video
        """
        # TODO: Implement VACE inpainting
        # 1. Load video and mask
        # 2. Use VACE model with inpaint mode
        # 3. Return result
        
        raise NotImplementedError("Video inpainting coming in next sprint")
    
    def style_transfer(
        self,
        video_path: str,
        style_reference: str,
        strength: float = 0.5,
        output_path: Optional[str] = None
    ) -> str:
        """
        Apply style from reference to video.
        
        Transfers visual style from a reference image or video
        to the input video while preserving motion.
        
        Args:
            video_path: Input video
            style_reference: Path to style reference image/video
            strength: Style transfer intensity (0-1)
            output_path: Where to save result
            
        Returns:
            Path to styled video
        """
        # TODO: Implement style transfer
        raise NotImplementedError("Style transfer coming in next sprint")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get service capabilities based on configuration."""
        return {
            "model": f"Wan2.1-T2V-{self.config.model_size.value}",
            "resolution": self.config.resolution,
            "max_duration": 81 / self.config.fps,  # ~3.4 seconds
            "cgpu_available": self.cgpu_available,
            "supported_tasks": [
                WanTaskType.TEXT_TO_VIDEO.value,
                # WanTaskType.IMAGE_TO_VIDEO.value,  # Coming soon
                # WanTaskType.VIDEO_INPAINTING.value,
            ]
        }


class WanBRollGenerator:
    """
    High-level B-Roll generator for Montage AI integration.
    
    Provides convenient methods for common B-Roll generation tasks
    within the video editing workflow.
    
    Usage:
        generator = WanBRollGenerator()
        
        # Generate transition B-Roll
        transition = generator.generate_transition(
            from_scene="city traffic",
            to_scene="peaceful forest",
            style="cinematic"
        )
        
        # Generate filler clip
        filler = generator.generate_filler(
            context="upbeat music video",
            duration=2.0
        )
    """
    
    # Prompt templates for different B-Roll types
    TEMPLATES = {
        "transition": (
            "Cinematic smooth transition shot, {style} style, "
            "transitioning from {from_scene} atmosphere to {to_scene}, "
            "high quality, professional cinematography"
        ),
        "filler": (
            "Professional {context} footage, {style} style, "
            "high quality, dynamic movement, visually interesting"
        ),
        "establishing": (
            "Establishing shot, {location}, {time_of_day}, "
            "{style} cinematography, wide angle, atmospheric"
        ),
        "detail": (
            "Close-up detail shot, {subject}, {style} style, "
            "shallow depth of field, professional lighting"
        ),
        "action": (
            "Dynamic action shot, {action}, {style} style, "
            "energetic, high quality, dramatic"
        )
    }
    
    def __init__(self, config: Optional[WanConfig] = None):
        self.service = WanVACEService(config)
    
    def generate_transition(
        self,
        from_scene: str,
        to_scene: str,
        style: str = "cinematic",
        duration: float = 2.0
    ) -> str:
        """Generate transition B-Roll between two scenes."""
        prompt = self.TEMPLATES["transition"].format(
            from_scene=from_scene,
            to_scene=to_scene,
            style=style
        )
        return self.service.generate_broll(prompt, duration)
    
    def generate_filler(
        self,
        context: str,
        style: str = "modern",
        duration: float = 2.0
    ) -> str:
        """Generate generic filler B-Roll."""
        prompt = self.TEMPLATES["filler"].format(
            context=context,
            style=style
        )
        return self.service.generate_broll(prompt, duration)
    
    def generate_establishing(
        self,
        location: str,
        time_of_day: str = "golden hour",
        style: str = "cinematic",
        duration: float = 3.0
    ) -> str:
        """Generate establishing shot."""
        prompt = self.TEMPLATES["establishing"].format(
            location=location,
            time_of_day=time_of_day,
            style=style
        )
        return self.service.generate_broll(prompt, duration)
    
    def generate_detail(
        self,
        subject: str,
        style: str = "artistic",
        duration: float = 2.0
    ) -> str:
        """Generate detail/insert shot."""
        prompt = self.TEMPLATES["detail"].format(
            subject=subject,
            style=style
        )
        return self.service.generate_broll(prompt, duration)
    
    def generate_action(
        self,
        action: str,
        style: str = "dynamic",
        duration: float = 2.0
    ) -> str:
        """Generate action shot."""
        prompt = self.TEMPLATES["action"].format(
            action=action,
            style=style
        )
        return self.service.generate_broll(prompt, duration)


# Convenience functions
def create_wan_service(model_size: str = "1.3B") -> WanVACEService:
    """Create Wan2.1-VACE service with specified model size."""
    size = WanModelSize.SMALL if model_size == "1.3B" else WanModelSize.LARGE
    config = WanConfig(model_size=size)
    return WanVACEService(config)


def generate_quick_broll(prompt: str, duration: float = 2.0) -> str:
    """Quick B-Roll generation with default settings."""
    service = WanVACEService()
    return service.generate_broll(prompt, duration)
