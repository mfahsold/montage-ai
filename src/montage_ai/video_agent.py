"""
VideoAgent Integration - Memory-Augmented Clip Analysis

Inspired by: ECCV 2024 Paper "VideoAgent: A Memory-Augmented Multimodal Agent 
             for Video Understanding"
Reference: github.com/YueFan1014/VideoAgent

This module provides intelligent clip analysis and selection through:
- Temporal Memory: Scene captions with embeddings for semantic search
- Object Memory: Tracked objects across video with SQL-backed storage
- 4 Core Tools: caption_retrieval, segment_localization, VQA, object_memory

Integration with Montage AI:
- Enhances footage_manager.py with semantic understanding
- Provides context-aware clip selection based on natural language queries
- Maintains memory across the editing session for coherent narratives

Architecture:
    Footage Files → VideoAgent Analysis → Memory Database
                                              ↓
    Creative Director Query → Tool Selection → Memory Query → Ranked Clips
"""

import os
import json
import sqlite3
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from .config import get_settings
from enum import Enum
import numpy as np

# Optional imports for full functionality
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class MemoryType(Enum):
    """Types of memory in the VideoAgent system."""
    TEMPORAL = "temporal"   # Time-based scene memory
    OBJECT = "object"       # Object tracking memory
    ACTION = "action"       # Action/event memory


@dataclass
class TemporalMemoryEntry:
    """
    A segment in temporal memory with caption and embedding.
    
    Temporal memory stores scene-level information that can be
    queried using natural language descriptions.
    """
    segment_id: str
    video_path: str
    start_time: float
    end_time: float
    caption: str
    embedding: Optional[List[float]] = None
    
    # Derived attributes
    scene_type: str = "action"
    energy_level: float = 0.5
    dominant_colors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "video_path": self.video_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "caption": self.caption,
            "scene_type": self.scene_type,
            "energy_level": self.energy_level,
            "duration": self.duration
        }


@dataclass
class ObjectMemoryEntry:
    """
    An object tracked across video frames.
    
    Object memory enables queries like "Find all clips with a red car"
    or "Show me scenes where person X appears".
    """
    object_id: str
    object_class: str
    first_seen: float
    last_seen: float
    video_path: str
    appearances: List[Dict] = field(default_factory=list)
    # Each appearance: {"timestamp": float, "bbox": [x,y,w,h], "confidence": float}
    
    @property
    def total_appearances(self) -> int:
        return len(self.appearances)
    
    @property
    def screen_time(self) -> float:
        """Total time object is visible."""
        return self.last_seen - self.first_seen


@dataclass 
class ActionMemoryEntry:
    """
    An action or event detected in video.
    
    Actions are higher-level semantic events like "person walking",
    "car driving", "crowd cheering".
    """
    action_id: str
    action_class: str
    start_time: float
    end_time: float
    video_path: str
    confidence: float = 0.5
    participants: List[str] = field(default_factory=list)  # Object IDs involved


class VideoAgentMemory:
    """
    SQLite-backed memory system for VideoAgent.
    
    Provides persistent storage for:
    - Temporal segments with captions
    - Tracked objects
    - Detected actions
    
    Enables fast retrieval through indexed queries and
    embedding-based similarity search.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(get_settings().paths.temp_dir / "video_agent_memory.db")
        self.db_path = db_path
        self._init_database()
        
        # Embedding model for semantic search
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                # Use lightweight model for efficiency
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass
    
    def _init_database(self):
        """Initialize SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        
        # Temporal memory table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS temporal_memory (
                segment_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                caption TEXT,
                embedding BLOB,
                scene_type TEXT DEFAULT 'action',
                energy_level REAL DEFAULT 0.5,
                dominant_colors TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Object memory table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS object_memory (
                object_id TEXT PRIMARY KEY,
                object_class TEXT NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                video_path TEXT NOT NULL,
                appearances TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Action memory table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS action_memory (
                action_id TEXT PRIMARY KEY,
                action_class TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                video_path TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                participants TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes for fast queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_video ON temporal_memory(video_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_time ON temporal_memory(start_time, end_time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_object_class ON object_memory(object_class)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_action_class ON action_memory(action_class)")
        
        conn.commit()
        conn.close()
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        if self.embedding_model is None:
            return None
        return self.embedding_model.encode(text)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    # ========================================================================
    # Temporal Memory Operations
    # ========================================================================
    
    def add_temporal_entry(self, entry: TemporalMemoryEntry) -> None:
        """Add a temporal memory entry."""
        conn = sqlite3.connect(self.db_path)
        
        # Generate embedding if model available
        embedding_blob = None
        if entry.embedding:
            embedding_blob = np.array(entry.embedding).tobytes()
        elif entry.caption and self.embedding_model:
            embedding = self._get_embedding(entry.caption)
            if embedding is not None:
                embedding_blob = embedding.tobytes()
        
        conn.execute("""
            INSERT OR REPLACE INTO temporal_memory 
            (segment_id, video_path, start_time, end_time, caption, embedding, 
             scene_type, energy_level, dominant_colors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.segment_id,
            entry.video_path,
            entry.start_time,
            entry.end_time,
            entry.caption,
            embedding_blob,
            entry.scene_type,
            entry.energy_level,
            json.dumps(entry.dominant_colors)
        ))
        
        conn.commit()
        conn.close()
    
    def search_temporal_by_caption(
        self, 
        query: str, 
        top_k: int = 5,
        video_filter: Optional[str] = None
    ) -> List[Tuple[TemporalMemoryEntry, float]]:
        """
        Search temporal memory using natural language query.
        
        Returns entries ranked by semantic similarity to query.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Fetch all entries (or filtered by video)
        if video_filter:
            cursor = conn.execute(
                "SELECT * FROM temporal_memory WHERE video_path = ?",
                (video_filter,)
            )
        else:
            cursor = conn.execute("SELECT * FROM temporal_memory")
        
        results = []
        for row in cursor.fetchall():
            entry = TemporalMemoryEntry(
                segment_id=row[0],
                video_path=row[1],
                start_time=row[2],
                end_time=row[3],
                caption=row[4],
                scene_type=row[6],
                energy_level=row[7],
                dominant_colors=json.loads(row[8]) if row[8] else []
            )
            
            # Calculate similarity score
            if query_embedding is not None and row[5]:
                stored_embedding = np.frombuffer(row[5], dtype=np.float32)
                similarity = self._cosine_similarity(query_embedding, stored_embedding)
            else:
                # Fallback to simple text matching
                query_words = set(query.lower().split())
                caption_words = set(entry.caption.lower().split()) if entry.caption else set()
                overlap = len(query_words & caption_words)
                similarity = overlap / max(len(query_words), 1)
            
            results.append((entry, similarity))
        
        conn.close()
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_temporal_at_time(
        self, 
        timestamp: float,
        video_path: Optional[str] = None
    ) -> Optional[TemporalMemoryEntry]:
        """Get temporal entry containing a specific timestamp."""
        conn = sqlite3.connect(self.db_path)
        
        if video_path:
            cursor = conn.execute("""
                SELECT * FROM temporal_memory 
                WHERE video_path = ? AND start_time <= ? AND end_time >= ?
            """, (video_path, timestamp, timestamp))
        else:
            cursor = conn.execute("""
                SELECT * FROM temporal_memory 
                WHERE start_time <= ? AND end_time >= ?
            """, (timestamp, timestamp))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return TemporalMemoryEntry(
                segment_id=row[0],
                video_path=row[1],
                start_time=row[2],
                end_time=row[3],
                caption=row[4],
                scene_type=row[6],
                energy_level=row[7],
                dominant_colors=json.loads(row[8]) if row[8] else []
            )
        return None
    
    # ========================================================================
    # Object Memory Operations
    # ========================================================================
    
    def add_object_entry(self, entry: ObjectMemoryEntry) -> None:
        """Add or update an object memory entry."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO object_memory
            (object_id, object_class, first_seen, last_seen, video_path, appearances)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry.object_id,
            entry.object_class,
            entry.first_seen,
            entry.last_seen,
            entry.video_path,
            json.dumps(entry.appearances)
        ))
        
        conn.commit()
        conn.close()
    
    def search_objects_by_class(
        self, 
        object_class: str,
        video_filter: Optional[str] = None
    ) -> List[ObjectMemoryEntry]:
        """Find all objects of a specific class."""
        conn = sqlite3.connect(self.db_path)
        
        if video_filter:
            cursor = conn.execute("""
                SELECT * FROM object_memory 
                WHERE object_class LIKE ? AND video_path = ?
            """, (f"%{object_class}%", video_filter))
        else:
            cursor = conn.execute("""
                SELECT * FROM object_memory WHERE object_class LIKE ?
            """, (f"%{object_class}%",))
        
        results = []
        for row in cursor.fetchall():
            results.append(ObjectMemoryEntry(
                object_id=row[0],
                object_class=row[1],
                first_seen=row[2],
                last_seen=row[3],
                video_path=row[4],
                appearances=json.loads(row[5]) if row[5] else []
            ))
        
        conn.close()
        return results
    
    def get_objects_at_time(
        self, 
        timestamp: float,
        video_path: Optional[str] = None
    ) -> List[ObjectMemoryEntry]:
        """Get all objects visible at a specific timestamp."""
        conn = sqlite3.connect(self.db_path)
        
        if video_path:
            cursor = conn.execute("""
                SELECT * FROM object_memory 
                WHERE video_path = ? AND first_seen <= ? AND last_seen >= ?
            """, (video_path, timestamp, timestamp))
        else:
            cursor = conn.execute("""
                SELECT * FROM object_memory 
                WHERE first_seen <= ? AND last_seen >= ?
            """, (timestamp, timestamp))
        
        results = []
        for row in cursor.fetchall():
            results.append(ObjectMemoryEntry(
                object_id=row[0],
                object_class=row[1],
                first_seen=row[2],
                last_seen=row[3],
                video_path=row[4],
                appearances=json.loads(row[5]) if row[5] else []
            ))
        
        conn.close()
        return results
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def clear_video(self, video_path: str) -> None:
        """Remove all memory entries for a specific video."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM temporal_memory WHERE video_path = ?", (video_path,))
        conn.execute("DELETE FROM object_memory WHERE video_path = ?", (video_path,))
        conn.execute("DELETE FROM action_memory WHERE video_path = ?", (video_path,))
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        conn = sqlite3.connect(self.db_path)
        
        temporal_count = conn.execute("SELECT COUNT(*) FROM temporal_memory").fetchone()[0]
        object_count = conn.execute("SELECT COUNT(*) FROM object_memory").fetchone()[0]
        action_count = conn.execute("SELECT COUNT(*) FROM action_memory").fetchone()[0]
        
        conn.close()
        
        return {
            "temporal_entries": temporal_count,
            "object_entries": object_count,
            "action_entries": action_count,
            "total": temporal_count + object_count + action_count
        }


class VideoAgentAdapter:
    """
    Adapter for integrating VideoAgent concepts into Montage AI.
    
    Provides the 4 core tools from the VideoAgent paper:
    1. caption_retrieval - Find scenes based on description
    2. segment_localization - Locate specific segments
    3. visual_question_answering - Answer questions about video
    4. object_memory_querying - Find objects across time
    
    Usage:
        adapter = VideoAgentAdapter()
        
        # Analyze footage
        adapter.analyze_video("/data/input/video.mp4")
        
        # Query for clips
        results = adapter.caption_retrieval("energetic action scene")
        
        # Find specific objects
        cars = adapter.object_memory_querying("car")
    """
    
    def __init__(
        self, 
        db_path: Optional[str] = None,
        caption_model: str = "ollama"  # or "blip", "llava"
    ):
        if db_path is None:
            db_path = str(get_settings().paths.temp_dir / "video_agent_memory.db")
        self.memory = VideoAgentMemory(db_path)
        self.caption_model = caption_model
        
        # cgpu configuration for VQA
        llm = get_settings().llm
        self.cgpu_enabled = llm.cgpu_enabled
        self.cgpu_host = llm.cgpu_host
        self.cgpu_port = llm.cgpu_port
    
    def analyze_video(
        self, 
        video_path: str,
        segment_duration: float = 5.0,
        detect_objects: bool = True,
        generate_captions: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze video and populate memory databases.
        
        This is the main preprocessing step that:
        1. Segments video into temporal chunks
        2. Generates captions for each segment
        3. Detects and tracks objects
        4. Computes embeddings for semantic search
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of each temporal segment (seconds)
            detect_objects: Whether to run object detection
            generate_captions: Whether to generate scene captions
            
        Returns:
            Analysis statistics
        """
        if not CV2_AVAILABLE:
            return {"success": False, "error": "OpenCV not available"}
        
        # Clear previous analysis for this video
        self.memory.clear_video(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": f"Could not open {video_path}"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        segments_created = 0
        objects_detected = 0
        
        # Process video in segments
        current_time = 0.0
        segment_idx = 0
        
        while current_time < duration:
            segment_end = min(current_time + segment_duration, duration)
            
            # Generate segment ID
            segment_id = f"{Path(video_path).stem}_seg_{segment_idx:04d}"
            
            # Extract keyframe for analysis
            keyframe_time = (current_time + segment_end) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, keyframe_time * 1000)
            ret, frame = cap.read()
            
            if ret:
                # Analyze frame
                analysis = self._analyze_frame(frame, keyframe_time, video_path)
                
                # Generate caption if enabled
                caption = ""
                if generate_captions:
                    caption = self._generate_caption(frame, analysis)
                
                # Create temporal memory entry
                entry = TemporalMemoryEntry(
                    segment_id=segment_id,
                    video_path=video_path,
                    start_time=current_time,
                    end_time=segment_end,
                    caption=caption,
                    scene_type=analysis.get("scene_type", "action"),
                    energy_level=analysis.get("energy_level", 0.5),
                    dominant_colors=analysis.get("dominant_colors", [])
                )
                
                self.memory.add_temporal_entry(entry)
                segments_created += 1
                
                # Detect objects if enabled
                if detect_objects:
                    detected = self._detect_objects(frame, keyframe_time, video_path)
                    objects_detected += len(detected)
            
            current_time = segment_end
            segment_idx += 1
        
        cap.release()
        
        return {
            "success": True,
            "video_path": video_path,
            "duration": duration,
            "segments_created": segments_created,
            "objects_detected": objects_detected
        }
    
    def _analyze_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float,
        video_path: str
    ) -> Dict[str, Any]:
        """Analyze a single frame for scene attributes."""
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Dominant colors (simplified)
        hist = cv2.calcHist([hsv], [0], None, [12], [0, 180])
        dominant_hue = int(np.argmax(hist) * 15)
        
        hue_names = {
            0: "red", 15: "orange", 30: "yellow", 45: "lime",
            60: "green", 75: "cyan", 90: "blue", 105: "purple",
            120: "magenta", 135: "pink", 150: "red", 165: "red"
        }
        dominant_color = hue_names.get(dominant_hue, "neutral")
        
        # Brightness/energy estimation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        
        # Motion/energy estimation (using Laplacian variance as proxy)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        energy_level = min(1.0, variance / 5000.0)  # Normalize
        
        # Scene type heuristics
        if energy_level > 0.7:
            scene_type = "action"
        elif brightness < 0.3:
            scene_type = "dramatic"
        elif brightness > 0.7:
            scene_type = "bright"
        else:
            scene_type = "neutral"
        
        return {
            "scene_type": scene_type,
            "energy_level": energy_level,
            "brightness": brightness,
            "dominant_colors": [dominant_color],
            "variance": variance
        }
    
    def _generate_caption(self, frame: np.ndarray, analysis: Dict) -> str:
        """
        Generate a caption for the frame.
        
        Uses local heuristics or external LLM (Ollama/cgpu) if available.
        """
        # Simple heuristic caption
        scene_type = analysis.get("scene_type", "scene")
        colors = analysis.get("dominant_colors", [])
        energy = analysis.get("energy_level", 0.5)
        
        energy_desc = "high-energy" if energy > 0.6 else "calm" if energy < 0.4 else "moderate"
        color_desc = colors[0] if colors else "neutral"
        
        caption = f"A {energy_desc} {scene_type} scene with {color_desc} tones"
        
        # ROADMAP (Q2 2026): Integrate Ollama/LLaVA for better captions
        # Current caption system works for MVP; ML-based captioning is future enhancement
        # if self.caption_model == "ollama":
        #     caption = self._generate_ollama_caption(frame)
        
        return caption
    
    def _detect_objects(
        self, 
        frame: np.ndarray, 
        timestamp: float,
        video_path: str
    ) -> List[ObjectMemoryEntry]:
        """
        Detect objects in frame.
        
        Placeholder for YOLO/SAM integration.
        Currently returns empty list - implement with actual detector.
        """
        # ROADMAP (Q3 2026): Integrate object detection model
        # Current face detection via MediaPipe works for MVP
        # Full object detection (YOLO/SAM/DETR) is future enhancement
        # Options:
        # - YOLO via ultralytics
        # - SAM (Segment Anything)
        # - DETR
        
        return []
    
    # ========================================================================
    # VideoAgent Tools (4 Core Tools)
    # ========================================================================
    
    def caption_retrieval(
        self, 
        query: str, 
        top_k: int = 5,
        video_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tool 1: Find scenes based on natural language description.
        
        This is the primary tool for semantic scene search.
        
        Args:
            query: Natural language description (e.g., "energetic dance scene")
            top_k: Number of results to return
            video_filter: Optional - limit search to specific video
            
        Returns:
            List of matching segments with similarity scores
        """
        results = self.memory.search_temporal_by_caption(query, top_k, video_filter)
        
        return [
            {
                **entry.to_dict(),
                "similarity_score": score
            }
            for entry, score in results
        ]
    
    def segment_localization(
        self, 
        description: str,
        video_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Tool 2: Locate a specific segment in video.
        
        More precise than caption_retrieval - returns best single match.
        
        Args:
            description: What to find
            video_path: Which video to search
            
        Returns:
            Best matching segment or None
        """
        results = self.memory.search_temporal_by_caption(
            description, 
            top_k=1, 
            video_filter=video_path
        )
        
        if results:
            entry, score = results[0]
            return {
                **entry.to_dict(),
                "confidence": score
            }
        return None
    
    def visual_question_answering(
        self, 
        question: str, 
        video_path: str,
        timestamp: Optional[float] = None
    ) -> str:
        """
        Tool 3: Answer questions about video content.
        
        Uses the memory system and optionally LLM for complex questions.
        
        Args:
            question: Natural language question
            video_path: Video to query about
            timestamp: Optional - specific time to query
            
        Returns:
            Answer string
        """
        # Get relevant context from memory
        context_segments = self.caption_retrieval(question, top_k=3, video_filter=video_path)
        
        if timestamp is not None:
            time_segment = self.memory.get_temporal_at_time(timestamp, video_path)
            if time_segment:
                context_segments.insert(0, time_segment.to_dict())
        
        # Build context string
        context = "\n".join([
            f"- {seg.get('caption', 'No caption')} "
            f"(time: {seg.get('start_time', 0):.1f}s-{seg.get('end_time', 0):.1f}s)"
            for seg in context_segments
        ])
        
        # Simple heuristic answers
        question_lower = question.lower()
        
        if "how many" in question_lower:
            return f"Based on analysis, there are {len(context_segments)} relevant segments."
        
        if "what happens" in question_lower or "describe" in question_lower:
            if context_segments:
                return f"The video shows: {context_segments[0].get('caption', 'various scenes')}"
        
        if "when" in question_lower:
            if context_segments:
                seg = context_segments[0]
                return f"This occurs around {seg.get('start_time', 0):.1f} seconds."
        
        # Default response with context
        return f"Based on video analysis:\n{context}"
    
    def object_memory_querying(
        self, 
        object_class: str,
        video_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tool 4: Find objects across video timeline.
        
        Queries the object memory for all instances of a class.
        
        Args:
            object_class: Type of object (e.g., "car", "person", "dog")
            video_filter: Optional - limit to specific video
            
        Returns:
            List of object tracking entries
        """
        objects = self.memory.search_objects_by_class(object_class, video_filter)
        
        return [
            {
                "object_id": obj.object_id,
                "class": obj.object_class,
                "first_seen": obj.first_seen,
                "last_seen": obj.last_seen,
                "screen_time": obj.screen_time,
                "appearances": obj.total_appearances,
                "video_path": obj.video_path
            }
            for obj in objects
        ]
    
    # ========================================================================
    # Integration with Footage Manager
    # ========================================================================
    
    def get_clips_for_story_phase(
        self, 
        phase: str,
        music_bpm: float,
        available_videos: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get recommended clips for a story arc phase.
        
        Maps story phases to appropriate clip characteristics:
        - intro: calm, establishing shots
        - build: increasing energy
        - climax: high energy, action
        - sustain: variety, interest
        - outro: calm, resolution
        
        Args:
            phase: Story phase name
            music_bpm: Music tempo for timing
            available_videos: List of video paths to consider
            
        Returns:
            Ranked list of suitable clips
        """
        phase_queries = {
            "intro": "calm establishing wide shot scenic",
            "build": "movement energy building action",
            "climax": "intense fast action peak energy",
            "sustain": "interesting varied dynamic",
            "outro": "calm peaceful resolution settling"
        }
        
        query = phase_queries.get(phase.lower(), "dynamic interesting")
        
        all_results = []
        for video_path in available_videos:
            results = self.caption_retrieval(query, top_k=10, video_filter=video_path)
            all_results.extend(results)
        
        # Sort by similarity and return
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return all_results[:20]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return self.memory.get_stats()


# Convenience function for quick initialization
def create_video_agent(db_path: Optional[str] = None) -> VideoAgentAdapter:
    """Create a VideoAgent adapter instance."""
    if db_path is None:
        db_path = str(get_settings().paths.temp_dir / "video_agent_memory.db")
    return VideoAgentAdapter(db_path)
