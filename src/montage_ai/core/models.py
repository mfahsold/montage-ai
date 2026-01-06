from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class StyleParams(BaseModel):
    """Parameters defining the visual style."""
    name: str = "dynamic"
    params: Dict[str, Any] = Field(default_factory=dict)

class EditingInstructions(BaseModel):
    """
    Structured instructions for the Montage Builder.
    Replaces loose dictionary passing with typed validation.
    """
    # Audio overrides
    music_track: Optional[str] = None
    music_start: float = 0.0
    music_end: Optional[float] = None
    
    # Visual Style
    style: StyleParams = Field(default_factory=StyleParams)
    
    # Content / Scripting
    script: Optional[str] = None
    semantic_query: Optional[str] = None
    
    # Advanced features
    broll_plan: Optional[List[Any]] = None
    
    class Config:
        extra = "allow" # Allow extra fields for backward compatibility during migration
