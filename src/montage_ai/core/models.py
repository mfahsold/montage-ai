from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

from ..prompts import CinematographyConfig

class StyleParams(BaseModel):
    """Parameters defining the visual style."""
    name: str = "dynamic"
    params: Dict[str, Any] = Field(default_factory=dict)

class PacingConfig(BaseModel):
    speed: str = "medium"
    variation: str = "moderate"
    breathing_offset_ms: int = 40
    micro_pacing_jitter: float = 0.05

class StoryArcConfig(BaseModel):
    type: str = "three_act"
    tension_target: float = 0.5
    momentum_weight: float = 0.1

class EditingInstructions(BaseModel):
    """
    Structured instructions for the Montage Builder.
    Replaces loose dictionary passing with typed validation.
    """
    model_config = ConfigDict(extra="allow")
    
    # Audio overrides
    music_track: Optional[str] = None
    music_start: float = 0.0
    music_end: Optional[float] = None
    
    # Visual Style & AI Control
    style: StyleParams = Field(default_factory=StyleParams)
    cinematography: CinematographyConfig = Field(default_factory=CinematographyConfig)
    pacing: PacingConfig = Field(default_factory=PacingConfig)
    story_arc: StoryArcConfig = Field(default_factory=StoryArcConfig)
    
    # Content / Scripting
    script: Optional[str] = None
    semantic_query: Optional[str] = None
    
    # Advanced features
    broll_plan: Optional[List[Any]] = None
    
