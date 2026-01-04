import json
import redis
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from src.montage_ai.config import get_settings

class JobStore:
    def __init__(self):
        settings = get_settings()
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.prefix = "job:"
        self.ttl = 86400 * 7 # 7 days

    def create_job(self, job_id: str, data: Dict[str, Any]):
        key = f"{self.prefix}{job_id}"
        data['created_at'] = datetime.now().isoformat()
        self.redis.set(key, json.dumps(data), ex=self.ttl)
        # Add to sorted set for timeline
        self.redis.zadd("jobs:timeline", {job_id: datetime.now().timestamp()})

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        key = f"{self.prefix}{job_id}"
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    def update_job(self, job_id: str, updates: Dict[str, Any]):
        key = f"{self.prefix}{job_id}"
        # Simple get-update-set
        data = self.get_job(job_id)
        if data:
            data.update(updates)
            data['updated_at'] = datetime.now().isoformat()
            self.redis.set(key, json.dumps(data), ex=self.ttl)
            
            # Publish update for SSE
            self.redis.publish("job_updates", json.dumps({
                "job_id": job_id,
                "updates": updates,
                "full_data": data
            }))

    def list_jobs(self, limit: int = 50) -> Dict[str, Any]:
        # Get latest job IDs
        job_ids = self.redis.zrevrange("jobs:timeline", 0, limit - 1)
        result = {}
        for jid in job_ids:
            job = self.get_job(jid)
            if job:
                result[jid] = job
        return result
