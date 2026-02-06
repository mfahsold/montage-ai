"""
Montage AI Core Module

Contains the core pipeline components:
- MontageBuilder: Main orchestration class
- MontageContext: Job state container
- AnalysisCache: Persistent caching for analysis results
- EmbeddingSearchEngine: Fast similarity search for clips
"""

from .montage_builder import MontageBuilder
from .context import (
    MontageContext,
    MontageResult,
    AudioAnalysisResult,
    SceneInfo,
    OutputProfile,
    ClipMetadata,
)

from .analysis_cache import (
    AnalysisCache,
    AudioAnalysisEntry,
    SceneAnalysisEntry,
    EpisodicMemoryEntry,
    get_analysis_cache,
    reset_cache,
)

from .embedding_search import (
    EmbeddingSearchEngine,
    SearchResult,
    get_embedding_search,
    reset_embedding_search,
)

from .encoder_router import (
    EncoderRouter,
    EncoderConfig,
    EncoderTier,
    get_encoder_router,
    smart_encode,
)

__all__ = [
    # Builder
    "MontageBuilder",
    "MontageContext",
    "MontageResult",
    "AudioAnalysisResult",
    "SceneInfo",
    "OutputProfile",
    "ClipMetadata",
    # Cache
    "AnalysisCache",
    "AudioAnalysisEntry",
    "SceneAnalysisEntry",
    "EpisodicMemoryEntry",
    "get_analysis_cache",
    "reset_cache",
    # Search
    "EmbeddingSearchEngine",
    "SearchResult",
    "get_embedding_search",
    "reset_embedding_search",
    # Encoder Router
    "EncoderRouter",
    "EncoderConfig",
    "EncoderTier",
    "get_encoder_router",
    "smart_encode",
]
