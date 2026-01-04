"""
Project Session Management

Centralized state management for creative sessions (Transcript Editor, Shorts Studio).
Implements the "Single Source of Truth" pattern for editing state.

DRY: Replaces ad-hoc file path passing with structured session objects.
KISS: Simple JSON-backed persistence.
"""

import os
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

try:
    import redis
except ImportError:
    redis = None

from ..logger import logger

SESSION_DIR = Path(os.environ.get("SESSION_DIR", "/tmp/montage_sessions"))
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Redis Configuration
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
redis_client = None

if REDIS_HOST and redis:
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}. Falling back to file storage.")
        redis_client = None

@dataclass
class ProjectAsset:
    """A media asset within a project."""
    id: str
    type: str  # 'video', 'audio', 'transcript', 'shorts_crop'
    path: str
    filename: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectSession:
    """
    Represents an active editing session.
    
    Can be a Transcript Edit, a Shorts creation, or a full Montage.
    """
    id: str
    type: str  # 'transcript', 'shorts', 'montage'
    created_at: str
    updated_at: str
    assets: Dict[str, ProjectAsset] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)  # Editor state (cuts, crops, etc.)
    
    def add_asset(self, path: str, asset_type: str, metadata: Dict[str, Any] = None) -> ProjectAsset:
        """Register a new asset to the session."""
        asset_id = str(uuid.uuid4())[:8]
        path_obj = Path(path)
        
        asset = ProjectAsset(
            id=asset_id,
            type=asset_type,
            path=str(path_obj.absolute()),
            filename=path_obj.name,
            metadata=metadata or {}
        )
        self.assets[asset_id] = asset
        self.updated_at = datetime.now().isoformat()
        self.save()
        return asset

    def get_asset(self, asset_id: str) -> Optional[ProjectAsset]:
        """Retrieve an asset by ID."""
        return self.assets.get(asset_id)
    
    def get_main_video(self) -> Optional[ProjectAsset]:
        """Helper to find the primary video asset."""
        for asset in self.assets.values():
            if asset.type == 'video':
                return asset
        return None

    def update_state(self, key: str, value: Any):
        """Update a specific state key."""
        self.state[key] = value
        self.updated_at = datetime.now().isoformat()
        self.save()

    def save(self):
        """Persist session to disk or Redis."""
        data = asdict(self)
        
        # Save to Redis if available
        if redis_client:
            try:
                redis_client.set(f"session:{self.id}", json.dumps(data))
                # Also save to disk as backup/persistence if needed, 
                # but for now we can treat Redis as primary if active.
                # We'll continue to save to disk for hybrid workflows/persistence safety.
            except Exception as e:
                logger.error(f"Redis save failed: {e}")

        # Always save to disk as fallback/persistence
        file_path = SESSION_DIR / f"{self.id}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, session_id: str) -> Optional['ProjectSession']:
        """Load session from Redis or disk."""
        data = None
        
        # Try Redis first
        if redis_client:
            try:
                cached = redis_client.get(f"session:{session_id}")
                if cached:
                    data = json.loads(cached)
            except Exception as e:
                logger.error(f"Redis load failed: {e}")

        # Fallback to disk
        if not data:
            file_path = SESSION_DIR / f"{session_id}.json"
            if not file_path.exists():
                return None
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")
                return None
            
        try:
            # Reconstruct objects
            assets = {}
            for aid, adata in data.get('assets', {}).items():
                assets[aid] = ProjectAsset(**adata)
            
            return cls(
                id=data['id'],
                type=data.get('type', 'generic'),
                created_at=data['created_at'],
                updated_at=data.get('updated_at', data['created_at']),
                assets=assets,
                state=data.get('state', {})
            )
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    @classmethod
    def create(cls, session_type: str = 'generic') -> 'ProjectSession':
        """Create a new session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
        session = cls(
            id=session_id,
            type=session_type,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        session.save()
        return session

def get_session_manager():
    """Factory for session management."""
    return ProjectSession
