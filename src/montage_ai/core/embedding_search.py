"""
Embedding Search Engine - Fast similarity search across cached clip embeddings.

2025 Tech Vision: Infrastructure for "clips like this" and smart B-roll suggestions.

Uses pre-computed embeddings from analysis cache for fast nearest-neighbor search.
No external dependencies beyond numpy (uses brute-force for small collections).

Usage:
    from montage_ai.core.embedding_search import get_embedding_search

    search = get_embedding_search()
    search.index_directory("/data/input")  # Build index from cached analyses

    # Find similar clips
    similar = search.find_similar(embedding, k=5)
    similar = search.find_similar_to_clip("/data/input/clip.mp4", k=5)

    # Search by text query
    results = search.search_by_query("beach sunset", k=5)
"""

import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import logging

from .analysis_cache import get_analysis_cache, SemanticAnalysisEntry

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SearchResult:
    """Result of similarity search."""
    clip_path: str
    score: float  # Cosine similarity 0.0-1.0
    time_point: float
    analysis: Optional[SemanticAnalysisEntry] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_path": self.clip_path,
            "score": self.score,
            "time_point": self.time_point,
            "tags": self.analysis.tags if self.analysis else [],
            "mood": self.analysis.mood if self.analysis else "",
            "description": self.analysis.description if self.analysis else "",
        }


@dataclass
class EmbeddingIndex:
    """In-memory index of clip embeddings for fast search."""
    embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    clip_paths: List[str] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    analyses: List[Optional[SemanticAnalysisEntry]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.clip_paths)

    @property
    def is_empty(self) -> bool:
        return self.size == 0


# ============================================================================
# Embedding Search Engine
# ============================================================================

class EmbeddingSearchEngine:
    """
    Fast similarity search across clip embeddings.

    Maintains an in-memory index of embeddings from cached semantic analyses.
    Supports:
    - Find similar clips given an embedding
    - Find similar clips given an existing clip
    - Text query search (requires embedding model)

    Performance notes:
    - Uses brute-force cosine similarity (fast for <10k clips)
    - Low memory footprint (~1KB per indexed clip)
    - Thread-safe read access (index is immutable after build)
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the search engine.

        Args:
            embedding_dim: Dimension of embeddings (all-MiniLM-L6-v2 = 384)
        """
        self.embedding_dim = embedding_dim
        self.index = EmbeddingIndex()
        self._model = None  # Lazy-loaded embedding model

    @property
    def is_indexed(self) -> bool:
        """Check if index has been built."""
        return not self.index.is_empty

    def index_directory(
        self,
        directory: str,
        extensions: tuple = (".mp4", ".mov", ".avi", ".mkv", ".webm"),
        force_rebuild: bool = False,
    ) -> int:
        """
        Build index from all cached semantic analyses in directory.

        Args:
            directory: Directory to scan for video files
            extensions: Video file extensions to include
            force_rebuild: Clear existing index before building

        Returns:
            Number of clips indexed
        """
        if force_rebuild:
            self.index = EmbeddingIndex()

        cache = get_analysis_cache()
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return 0

        indexed_count = 0
        embeddings_list = []
        clip_paths_list = []
        time_points_list = []
        analyses_list = []

        # Scan for video files
        for ext in extensions:
            for video_path in dir_path.glob(f"**/*{ext}"):
                # Look for semantic cache files
                for cache_file in video_path.parent.glob(f"{video_path.name}.semantic_*.json"):
                    try:
                        # Extract time point from filename
                        time_ms = int(cache_file.stem.split("_")[-1])
                        time_point = time_ms / 1000.0

                        # Load cached analysis
                        analysis = cache.load_semantic(str(video_path), time_point)
                        if analysis and analysis.caption_embedding:
                            emb = np.array(analysis.caption_embedding, dtype=np.float32)
                            if emb.shape[0] == self.embedding_dim:
                                embeddings_list.append(emb)
                                clip_paths_list.append(str(video_path))
                                time_points_list.append(time_point)
                                analyses_list.append(analysis)
                                indexed_count += 1

                    except (ValueError, OSError) as e:
                        logger.debug(f"Skipping cache file {cache_file}: {e}")
                        continue

        # Build numpy array for fast computation
        if embeddings_list:
            self.index = EmbeddingIndex(
                embeddings=np.vstack(embeddings_list),
                clip_paths=clip_paths_list,
                time_points=time_points_list,
                analyses=analyses_list,
            )

        logger.info(f"Indexed {indexed_count} clip embeddings from {directory}")
        return indexed_count

    def add_embedding(
        self,
        clip_path: str,
        embedding: np.ndarray,
        time_point: float = 0.0,
        analysis: Optional[SemanticAnalysisEntry] = None,
    ) -> bool:
        """
        Add a single embedding to the index.

        Args:
            clip_path: Path to the clip
            embedding: Embedding vector
            time_point: Time point in clip
            analysis: Optional cached analysis entry

        Returns:
            True if added successfully
        """
        if embedding.shape[0] != self.embedding_dim:
            logger.warning(f"Embedding dim mismatch: {embedding.shape[0]} != {self.embedding_dim}")
            return False

        emb = embedding.astype(np.float32).reshape(1, -1)

        if self.index.is_empty:
            self.index = EmbeddingIndex(
                embeddings=emb,
                clip_paths=[clip_path],
                time_points=[time_point],
                analyses=[analysis],
            )
        else:
            self.index.embeddings = np.vstack([self.index.embeddings, emb])
            self.index.clip_paths.append(clip_path)
            self.index.time_points.append(time_point)
            self.index.analyses.append(analysis)

        return True

    def find_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        exclude_paths: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Find k most similar clips to query embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            exclude_paths: Paths to exclude from results

        Returns:
            List of SearchResult sorted by similarity descending
        """
        if self.index.is_empty:
            return []

        exclude_set = set(exclude_paths or [])

        # Normalize query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        query_norm = query / (np.linalg.norm(query) + 1e-9)

        # Compute cosine similarities
        index_norms = np.linalg.norm(self.index.embeddings, axis=1, keepdims=True)
        normalized_index = self.index.embeddings / (index_norms + 1e-9)
        similarities = np.dot(normalized_index, query_norm.T).flatten()

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        # Build results
        results = []
        for idx in sorted_indices:
            clip_path = self.index.clip_paths[idx]
            if clip_path in exclude_set:
                continue

            results.append(SearchResult(
                clip_path=clip_path,
                score=float(similarities[idx]),
                time_point=self.index.time_points[idx],
                analysis=self.index.analyses[idx],
            ))

            if len(results) >= k:
                break

        return results

    def find_similar_to_clip(
        self,
        clip_path: str,
        k: int = 5,
        exclude_self: bool = True,
    ) -> List[SearchResult]:
        """
        Find clips similar to an existing indexed clip.

        Args:
            clip_path: Path to the reference clip
            k: Number of results to return
            exclude_self: Whether to exclude the reference clip

        Returns:
            List of SearchResult sorted by similarity descending
        """
        if self.index.is_empty:
            return []

        # Find embedding for reference clip
        ref_idx = None
        for i, path in enumerate(self.index.clip_paths):
            if path == clip_path or os.path.basename(path) == os.path.basename(clip_path):
                ref_idx = i
                break

        if ref_idx is None:
            logger.warning(f"Clip not found in index: {clip_path}")
            return []

        query_embedding = self.index.embeddings[ref_idx]
        exclude_paths = [clip_path] if exclude_self else None

        return self.find_similar(query_embedding, k=k, exclude_paths=exclude_paths)

    def search_by_query(
        self,
        query_text: str,
        k: int = 5,
    ) -> List[SearchResult]:
        """
        Search clips by text query.

        Requires sentence-transformers to be installed.

        Args:
            query_text: Natural language query
            k: Number of results to return

        Returns:
            List of SearchResult sorted by similarity descending
        """
        if self.index.is_empty:
            return []

        # Lazy load embedding model
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.warning("sentence_transformers not available for text search")
                return []

        try:
            query_embedding = self._model.encode(query_text)
            return self.find_similar(query_embedding, k=k)
        except Exception as e:
            logger.warning(f"Failed to encode query: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "indexed_clips": self.index.size,
            "embedding_dim": self.embedding_dim,
            "memory_mb": (self.index.embeddings.nbytes / 1024 / 1024) if not self.index.is_empty else 0,
            "unique_clips": len(set(self.index.clip_paths)),
        }


# ============================================================================
# Singleton Accessor
# ============================================================================

_embedding_search: Optional[EmbeddingSearchEngine] = None


def get_embedding_search() -> EmbeddingSearchEngine:
    """Get singleton EmbeddingSearchEngine instance."""
    global _embedding_search
    if _embedding_search is None:
        _embedding_search = EmbeddingSearchEngine()
    return _embedding_search


def reset_embedding_search() -> None:
    """Reset the singleton instance (for testing)."""
    global _embedding_search
    _embedding_search = None


__all__ = [
    "EmbeddingSearchEngine",
    "SearchResult",
    "EmbeddingIndex",
    "get_embedding_search",
    "reset_embedding_search",
]
