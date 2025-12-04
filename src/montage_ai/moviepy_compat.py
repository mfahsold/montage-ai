"""MoviePy 1.x/2.x compatibility layer.

Provides unified API for both MoviePy versions, centralizing all
version-specific workarounds in one place.

MoviePy 2.x changes (from 1.x):
- from moviepy.editor → from moviepy
- .subclip() → .subclipped()
- .set_audio() → .with_audio()
- .set_duration() → .with_duration()
- .set_position() → .with_position()
- .resize() → .resized()
- .crop() → .cropped()
"""

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    ColorClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

__all__ = [
    # Re-exports
    "VideoFileClip",
    "AudioFileClip",
    "ImageClip",
    "ColorClip",
    "TextClip",
    "CompositeVideoClip",
    "concatenate_videoclips",
    # Compat helpers
    "subclip",
    "set_audio",
    "set_duration",
    "set_position",
    "resize",
    "crop",
    # Dimension helpers
    "enforce_dimensions",
    "pad_to_target",
    "ensure_even_dimensions",
    "log_clip_info",
]


# =============================================================================
# Version-compatible method wrappers
# =============================================================================

def subclip(clip, start, end):
    """Extract subclip - works with MoviePy 1.x and 2.x."""
    if hasattr(clip, 'subclipped'):
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


def set_audio(clip, audio):
    """Set audio on clip - works with MoviePy 1.x and 2.x."""
    if hasattr(clip, 'with_audio'):
        return clip.with_audio(audio)
    return clip.set_audio(audio)


def set_duration(clip, duration):
    """Set duration - works with MoviePy 1.x and 2.x."""
    if hasattr(clip, 'with_duration'):
        return clip.with_duration(duration)
    return clip.set_duration(duration)


def set_position(clip, pos):
    """Set position - works with MoviePy 1.x and 2.x."""
    if hasattr(clip, 'with_position'):
        return clip.with_position(pos)
    return clip.set_position(pos)


def resize(clip, **kwargs):
    """Resize clip - works with MoviePy 1.x and 2.x.
    
    Args:
        clip: VideoFileClip or ImageClip
        **kwargs: newsize=(w,h) or width=x or height=y
    """
    if hasattr(clip, 'resized'):
        return clip.resized(**kwargs)
    return clip.resize(**kwargs)


def crop(clip, **kwargs):
    """Crop clip - works with MoviePy 1.x and 2.x.
    
    Args:
        clip: VideoFileClip
        **kwargs: x1, y1, x2, y2 coordinates
    """
    if hasattr(clip, 'cropped'):
        return clip.cropped(**kwargs)
    return clip.crop(**kwargs)


# =============================================================================
# Dimension helpers (extracted from editor.py for DRY)
# =============================================================================

def log_clip_info(label: str, clip, extra: str = ""):
    """Unified clip logging helper.
    
    Args:
        label: Description (e.g., "Input", "After resize")
        clip: VideoFileClip with .size and .duration
        extra: Optional extra info to append
    """
    w, h = clip.size
    dur = getattr(clip, 'duration', 0) or 0
    msg = f"  📐 {label}: {w}x{h} ({dur:.2f}s)"
    if extra:
        msg += f" {extra}"
    print(msg)


def ensure_even_dimensions(clip):
    """Ensure clip dimensions are even (required by h264).
    
    Returns:
        Tuple of (clip, was_modified)
    """
    w, h = clip.size
    if w % 2 == 0 and h % 2 == 0:
        return clip, False
    
    new_w = w if w % 2 == 0 else w - 1
    new_h = h if h % 2 == 0 else h - 1
    print(f"  ⚠️  Odd dimensions {w}x{h} → {new_w}x{new_h}")
    return crop(clip, x2=new_w, y2=new_h), True


def pad_to_target(clip, target_w: int, target_h: int):
    """Pad clip to target size with black background (centered).
    
    Args:
        clip: VideoFileClip smaller than target
        target_w: Target width
        target_h: Target height
        
    Returns:
        CompositeVideoClip with black padding
    """
    bg = ColorClip(
        size=(target_w, target_h),
        color=(0, 0, 0),
        duration=clip.duration
    )
    return CompositeVideoClip(
        [set_duration(bg, clip.duration), set_position(clip, ('center', 'center'))],
        size=(target_w, target_h)
    )


def enforce_dimensions(clip, target_w: int, target_h: int, verbose: bool = True):
    """Ensure clip has exact target dimensions via crop/pad.
    
    Handles:
    1. Aspect ratio correction (crop to target ratio)
    2. Scaling to target size
    3. Corrective crop/pad for off-by-one errors
    4. Even dimension enforcement
    
    Args:
        clip: VideoFileClip to process
        target_w: Exact target width
        target_h: Exact target height
        verbose: Print progress info
        
    Returns:
        Processed clip with exact dimensions
    """
    target_ratio = target_w / target_h
    w, h = clip.size
    current_ratio = w / h
    
    if verbose:
        log_clip_info("Input", clip, f"ratio={current_ratio:.3f}")
    
    # Step 1: Crop to target aspect ratio
    if current_ratio > target_ratio + 0.01:
        # Video is wider - crop width
        new_w = round(h * target_ratio)
        crop_x = (w - new_w) // 2
        if verbose:
            print(f"  ✂️ Crop width: {w} → {new_w}")
        clip = crop(clip, x1=crop_x, x2=crop_x + new_w)
    elif current_ratio < target_ratio - 0.01:
        # Video is taller - crop height
        new_h = round(w / target_ratio)
        crop_y = (h - new_h) // 2
        if verbose:
            print(f"  ✂️ Crop height: {h} → {new_h}")
        clip = crop(clip, y1=crop_y, y2=crop_y + new_h)
    
    # Step 2: Scale to target size
    if clip.size != (target_w, target_h):
        if verbose:
            print(f"  📐 Resize: {clip.size} → ({target_w}, {target_h})")
        clip = resize(clip, newsize=(target_w, target_h))
    
    # Step 3: Verify and correct (MoviePy sometimes returns off-by-one)
    actual_w, actual_h = clip.size
    if actual_w != target_w or actual_h != target_h:
        if verbose:
            print(f"  🔧 Correcting: {actual_w}x{actual_h} → {target_w}x{target_h}")
        
        if actual_w < target_w or actual_h < target_h:
            # Pad to target
            clip = pad_to_target(clip, target_w, target_h)
        else:
            # Crop to target (center)
            crop_x = (actual_w - target_w) // 2 if actual_w > target_w else 0
            crop_y = (actual_h - target_h) // 2 if actual_h > target_h else 0
            clip = crop(clip, x1=crop_x, y1=crop_y, x2=crop_x + target_w, y2=crop_y + target_h)
    
    # Step 4: Ensure even dimensions
    clip, _ = ensure_even_dimensions(clip)
    
    if verbose:
        log_clip_info("Output", clip, "✅")
    
    return clip
