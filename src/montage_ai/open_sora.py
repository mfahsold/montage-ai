"""
Open-Sora 2.0 Integration - Text-to-Video Generation via cgpu

Based on: HPC-AI Tech Open-Sora
Repository: github.com/hpcaitech/Open-Sora
HuggingFace: hpcai-tech/Open-Sora-v2
License: Apache-2.0

Open-Sora is an open-source reproduction of Sora with:
- 11B parameter model
- Text-to-Video (T2V) generation
- Image-to-Video (I2V) generation
- Up to 16 seconds of video generation
- Apache-2.0 license for commercial use

Hardware Requirements:
- 256p: 1x GPU (16GB+ VRAM) - feasible on cgpu T4
- 512p: 2-4x GPUs
- 768p: 8x GPUs (not practical for cgpu)

Strategy for Montage AI:
1. Generate at 256p resolution using cgpu
2. Upscale to target resolution using Real-ESRGAN (also via cgpu)
3. This achieves 1024p output with single T4 GPU

Architecture:
    Prompt → Open-Sora (256p) → Real-ESRGAN (4x) → 1024p Video

Version: 1.1.0 - Uses shared cgpu_utils module
"""

import os
import json
import base64
import tempfile
import subprocess
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from enum import Enum

# Import shared cgpu utilities
from .cgpu_utils import (
    CGPUConfig,
    is_cgpu_available,
    run_cgpu_command,
    cgpu_copy_to_remote,
)


class OpenSoraResolution(Enum):
    """Available Open-Sora resolutions."""
    LOW = "256p"      # 1x GPU, 16GB VRAM - cgpu feasible
    MEDIUM = "512p"   # 2-4x GPUs
    HIGH = "768p"     # 8x GPUs - not practical


class OpenSoraAspectRatio(Enum):
    """Supported aspect ratios."""
    SQUARE = "1:1"
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    WIDE = "21:9"


@dataclass
class OpenSoraConfig:
    """Configuration for Open-Sora generation."""
    resolution: OpenSoraResolution = OpenSoraResolution.LOW
    aspect_ratio: OpenSoraAspectRatio = OpenSoraAspectRatio.LANDSCAPE
    num_frames: int = 51          # ~2 seconds at 24fps
    fps: int = 24
    guidance_scale: float = 7.0
    num_inference_steps: int = 50
    
    # Model settings
    model_id: str = "hpcai-tech/Open-Sora-v2"
    
    # cgpu settings
    use_cgpu: bool = True
    cgpu_timeout: int = 600       # 10 minutes for generation
    auto_upscale: bool = True     # Upscale with Real-ESRGAN after generation
    
    @property
    def width(self) -> int:
        """Calculate width based on resolution and aspect ratio."""
        base = 256 if self.resolution == OpenSoraResolution.LOW else 512
        
        if self.aspect_ratio == OpenSoraAspectRatio.SQUARE:
            return base
        elif self.aspect_ratio == OpenSoraAspectRatio.LANDSCAPE:
            return int(base * 16 / 9)
        elif self.aspect_ratio == OpenSoraAspectRatio.PORTRAIT:
            return int(base * 9 / 16)
        elif self.aspect_ratio == OpenSoraAspectRatio.WIDE:
            return int(base * 21 / 9)
        return base
    
    @property
    def height(self) -> int:
        """Calculate height based on resolution and aspect ratio."""
        base = 256 if self.resolution == OpenSoraResolution.LOW else 512
        
        if self.aspect_ratio == OpenSoraAspectRatio.SQUARE:
            return base
        elif self.aspect_ratio == OpenSoraAspectRatio.LANDSCAPE:
            return base
        elif self.aspect_ratio == OpenSoraAspectRatio.PORTRAIT:
            return int(base * 16 / 9)
        elif self.aspect_ratio == OpenSoraAspectRatio.WIDE:
            return base
        return base


class OpenSoraGenerator:
    """
    Open-Sora Video Generator.
    
    Generates video from text prompts using HPC-AI Tech's Open-Sora
    model via cgpu cloud GPU. Automatically upscales to higher resolution
    using Real-ESRGAN for production-quality output.
    
    Pipeline:
        1. Open-Sora generates 256p video on cgpu T4
        2. Real-ESRGAN upscales 4x to 1024p
        3. Optional: Post-processing (stabilization, color correction)
    
    Usage:
        generator = OpenSoraGenerator()
        
        # Generate video
        video = generator.generate(
            prompt="A majestic eagle soaring over snow-capped mountains",
            duration=3.0
        )
        
        # Generate with reference image
        video = generator.generate_from_image(
            image_path="/data/assets/reference.jpg",
            prompt="The scene comes alive with gentle motion",
            duration=2.0
        )
    """
    
    # Colab setup script for Open-Sora
    COLAB_SETUP_SCRIPT = '''
import subprocess
import sys
import os

print("Setting up Open-Sora environment...")

# Install PyTorch with CUDA
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "torch>=2.1.0", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)

# Install Open-Sora dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.36.0", "accelerate", "diffusers>=0.28.0",
    "einops", "decord", "imageio", "imageio-ffmpeg",
    "colossalai>=0.3.0"], check=True)

# Clone Open-Sora if not exists
if not os.path.exists("Open-Sora"):
    subprocess.run(["git", "clone", "https://github.com/hpcaitech/Open-Sora.git"], check=True)

os.chdir("Open-Sora")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

print("Open-Sora setup complete!")
'''

    # Generation script template
    T2V_SCRIPT = '''
import torch
import os
import sys
import base64

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Add Open-Sora to path
sys.path.insert(0, "Open-Sora")

print("Loading Open-Sora model...")

from opensora.models import STDiT
from opensora.utils.inference_utils import load_model, get_scheduler
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
import imageio

# Configuration
PROMPT = """{prompt}"""
NEGATIVE_PROMPT = """{negative_prompt}"""
WIDTH = {width}
HEIGHT = {height}
NUM_FRAMES = {num_frames}
FPS = {fps}
GUIDANCE_SCALE = {guidance_scale}
NUM_STEPS = {num_inference_steps}
OUTPUT_PATH = "/tmp/opensora_output.mp4"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {{device}}")

# Load components
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16
).to(device)

print("Loading text encoder...")
text_encoder = T5EncoderModel.from_pretrained(
    "google/t5-v1_1-xxl",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

print("Loading STDiT model...")
model = STDiT.from_pretrained(
    "{model_id}",
    torch_dtype=torch.float16
).to(device)

# Get scheduler
scheduler = get_scheduler("ddpm")

print(f"Generating video: {{PROMPT[:50]}}...")

# Encode prompt
text_inputs = tokenizer(
    PROMPT,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    text_embeddings = text_encoder(text_inputs.input_ids)[0]

# Generate latents
latent_shape = (1, 4, NUM_FRAMES // 4, HEIGHT // 8, WIDTH // 8)
latents = torch.randn(latent_shape, device=device, dtype=torch.float16)

# Denoising loop
scheduler.set_timesteps(NUM_STEPS)
for t in scheduler.timesteps:
    latent_model_input = scheduler.scale_model_input(latents, t)
    
    noise_pred = model(
        latent_model_input,
        timestep=t,
        encoder_hidden_states=text_embeddings
    )
    
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# Decode to video
print("Decoding video...")
video = vae.decode(latents / vae.config.scaling_factor).sample

# Post-process and save
video = (video / 2 + 0.5).clamp(0, 1)
video = video.permute(0, 2, 1, 3, 4).squeeze(0)  # BCTHW -> TCHW
video = (video * 255).to(torch.uint8).cpu().numpy()

print(f"Saving video to {{OUTPUT_PATH}}...")
imageio.mimwrite(OUTPUT_PATH, video, fps=FPS, quality=8)

print(f"Video saved! Shape: {{video.shape}}")

# Output as base64 for download
with open(OUTPUT_PATH, "rb") as f:
    video_data = base64.b64encode(f.read()).decode()

# Print marker for parsing
print("===VIDEO_BASE64_START===")
print(video_data)
print("===VIDEO_BASE64_END===")
'''

    # Simplified fallback script using diffusers
    DIFFUSERS_T2V_SCRIPT = '''
import torch
import base64
import os

print("Loading video generation pipeline...")

from diffusers import DiffusionPipeline
import imageio

# Configuration  
PROMPT = """{prompt}"""
NEGATIVE_PROMPT = """{negative_prompt}"""
NUM_FRAMES = {num_frames}
WIDTH = {width}
HEIGHT = {height}
FPS = {fps}
GUIDANCE_SCALE = {guidance_scale}
NUM_STEPS = {num_inference_steps}
OUTPUT_PATH = "/tmp/opensora_output.mp4"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {{device}}")

# Try loading Open-Sora or fall back to ModelScope
try:
    pipe = DiffusionPipeline.from_pretrained(
        "{model_id}",
        torch_dtype=torch.float16,
        variant="fp16"
    )
except Exception as e:
    print(f"Open-Sora not available, using ModelScope T2V: {{e}}")
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16,
        variant="fp16"
    )

pipe = pipe.to(device)
pipe.enable_model_cpu_offload()

print(f"Generating video: {{PROMPT[:50]}}...")

# Generate
output = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_frames=min(NUM_FRAMES, 24),  # ModelScope limit
    width=WIDTH,
    height=HEIGHT,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=NUM_STEPS
)

video = output.frames[0]

print(f"Saving video to {{OUTPUT_PATH}}...")
from diffusers.utils import export_to_video
export_to_video(video, OUTPUT_PATH, fps=FPS)

# Output as base64
with open(OUTPUT_PATH, "rb") as f:
    video_data = base64.b64encode(f.read()).decode()

print("===VIDEO_BASE64_START===")
print(video_data)
print("===VIDEO_BASE64_END===")
'''

    def __init__(self, config: Optional[OpenSoraConfig] = None):
        self.config = config or OpenSoraConfig()
        self._cgpu_setup_done = False
        self._check_environment()
    
    def _check_environment(self):
        """Check cgpu availability using shared utils."""
        self.cgpu_available = is_cgpu_available()
    
    def _run_cgpu(self, script: str, timeout: Optional[int] = None) -> str:
        """Execute Python script on cgpu cloud GPU using shared utils."""
        timeout = timeout or self.config.cgpu_timeout
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            # Copy to Colab using shared utils
            if not cgpu_copy_to_remote(script_path, "/tmp/opensora_script.py"):
                raise RuntimeError("Failed to copy script to Colab")
            
            # Execute using shared utils
            success, stdout, stderr = run_cgpu_command(
                "python /tmp/opensora_script.py",
                timeout=timeout
            )
            
            if not success:
                raise RuntimeError(f"Script execution failed: {stderr}")
            
            return stdout
            
        finally:
            os.unlink(script_path)
    
    def _parse_base64_output(self, output: str) -> Optional[bytes]:
        """Extract base64 video data from output."""
        start_marker = "===VIDEO_BASE64_START==="
        end_marker = "===VIDEO_BASE64_END==="
        
        if start_marker not in output or end_marker not in output:
            return None
        
        start_idx = output.index(start_marker) + len(start_marker)
        end_idx = output.index(end_marker)
        
        b64_data = output[start_idx:end_idx].strip()
        return base64.b64decode(b64_data)
    
    def _upscale_video(self, video_path: str, output_path: str) -> str:
        """Upscale video using Real-ESRGAN via cgpu."""
        try:
            from .cgpu_upscaler import upscale_with_cgpu
            return upscale_with_cgpu(video_path, output_path, scale=4)
        except ImportError:
            print("Warning: cgpu_upscaler not available, skipping upscale")
            return video_path
    
    def generate(
        self,
        prompt: str,
        duration: float = 2.0,
        negative_prompt: str = "blurry, low quality, distorted, ugly, watermark",
        output_path: Optional[str] = None,
        upscale: Optional[bool] = None
    ) -> str:
        """
        Generate video from text prompt.
        
        Args:
            prompt: Description of desired video
            duration: Video duration in seconds (max ~2s for 256p)
            negative_prompt: What to avoid
            output_path: Where to save (auto-generated if None)
            upscale: Whether to 4x upscale with Real-ESRGAN
            
        Returns:
            Path to generated video
        """
        if not self.cgpu_available:
            raise RuntimeError("cgpu GPU not available. Set CGPU_GPU_ENABLED=true")
        
        # Calculate frames
        num_frames = min(int(duration * self.config.fps), 51)  # Max 51 for T4
        
        # Generate output path
        if output_path is None:
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            output_path = f"/tmp/opensora_{prompt_hash}.mp4"
        
        # Use simpler diffusers script (more reliable on cgpu)
        script = self.DIFFUSERS_T2V_SCRIPT.format(
            prompt=prompt.replace('"', '\\"'),
            negative_prompt=negative_prompt.replace('"', '\\"'),
            width=self.config.width,
            height=self.config.height,
            num_frames=num_frames,
            fps=self.config.fps,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            model_id=self.config.model_id
        )
        
        print(f"Generating video: '{prompt[:50]}...'")
        print(f"Resolution: {self.config.width}x{self.config.height}, "
              f"Frames: {num_frames}")
        
        # Run on cgpu
        output = self._run_cgpu(script)
        
        # Parse result
        video_data = self._parse_base64_output(output)
        
        if video_data is None:
            # Check for errors in output
            if "CUDA out of memory" in output:
                raise RuntimeError("GPU out of memory. Try shorter duration.")
            raise RuntimeError(f"Failed to generate video. Output: {output[:500]}")
        
        # Save raw output
        raw_path = output_path.replace(".mp4", "_raw.mp4")
        with open(raw_path, "wb") as f:
            f.write(video_data)
        
        # Upscale if requested
        should_upscale = upscale if upscale is not None else self.config.auto_upscale
        
        if should_upscale:
            print("Upscaling with Real-ESRGAN...")
            final_path = self._upscale_video(raw_path, output_path)
            # Clean up raw file
            if os.path.exists(raw_path) and final_path != raw_path:
                os.unlink(raw_path)
            return final_path
        else:
            os.rename(raw_path, output_path)
            return output_path
    
    def generate_from_image(
        self,
        image_path: str,
        prompt: str,
        duration: float = 2.0,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate video starting from a reference image (I2V).
        
        The generated video will start with content similar to
        the reference image and animate it based on the prompt.
        
        Args:
            image_path: Path to reference image
            prompt: Description of desired motion/action
            duration: Video duration in seconds
            output_path: Where to save
            
        Returns:
            Path to generated video
        """
        # TODO: Implement I2V generation
        raise NotImplementedError("Image-to-Video coming in next sprint")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get generator capabilities."""
        return {
            "model": self.config.model_id,
            "resolution": self.config.resolution.value,
            "max_duration": 51 / self.config.fps,  # ~2.1 seconds
            "cgpu_available": self.cgpu_available,
            "auto_upscale": self.config.auto_upscale,
            "supported_aspect_ratios": [ar.value for ar in OpenSoraAspectRatio]
        }


class OpenSoraPromptEnhancer:
    """
    Enhances prompts for better Open-Sora generation results.
    
    Open-Sora works best with detailed, cinematographic prompts.
    This helper adds quality boosters and stylistic elements.
    """
    
    QUALITY_BOOSTERS = [
        "high quality",
        "4K",
        "cinematic",
        "professional cinematography",
        "detailed",
        "sharp focus"
    ]
    
    STYLE_TEMPLATES = {
        "cinematic": "cinematic lighting, movie quality, professional color grading",
        "documentary": "documentary style, natural lighting, realistic",
        "artistic": "artistic, creative composition, unique visual style",
        "commercial": "commercial quality, polished, professional",
        "vintage": "vintage film look, film grain, nostalgic",
        "modern": "modern, sleek, contemporary aesthetic"
    }
    
    @classmethod
    def enhance(
        cls,
        prompt: str,
        style: str = "cinematic",
        add_quality_boosters: bool = True
    ) -> str:
        """
        Enhance a prompt for better generation results.
        
        Args:
            prompt: Original prompt
            style: Visual style to apply
            add_quality_boosters: Whether to add quality terms
            
        Returns:
            Enhanced prompt
        """
        parts = [prompt]
        
        # Add style
        if style in cls.STYLE_TEMPLATES:
            parts.append(cls.STYLE_TEMPLATES[style])
        
        # Add quality boosters
        if add_quality_boosters:
            parts.extend(cls.QUALITY_BOOSTERS[:3])
        
        return ", ".join(parts)


# Convenience functions
def create_open_sora(resolution: str = "256p") -> OpenSoraGenerator:
    """Create Open-Sora generator with specified resolution."""
    res = OpenSoraResolution.LOW if resolution == "256p" else OpenSoraResolution.MEDIUM
    config = OpenSoraConfig(resolution=res)
    return OpenSoraGenerator(config)


def generate_quick_video(prompt: str, duration: float = 2.0) -> str:
    """Quick video generation with default settings."""
    generator = OpenSoraGenerator()
    enhanced = OpenSoraPromptEnhancer.enhance(prompt)
    return generator.generate(enhanced, duration)
