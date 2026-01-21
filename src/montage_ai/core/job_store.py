import json
import redis
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from montage_ai.config import get_settings

class JobStore:
    def __init__(self):
        settings = get_settings()
        redis_host = settings.session.redis_host or 'localhost'
        redis_port = settings.session.redis_port
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

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update a job record and return True on success, False on failure.

        This method is idempotent and will swallow transient storage errors; callers
        should use `update_job_with_retry` if they require stronger guarantees.
        """
        key = f"{self.prefix}{job_id}"
        try:
            data = self.get_job(job_id)
            if data:
                data.update(updates)
                data['updated_at'] = datetime.now().isoformat()
                self.redis.set(key, json.dumps(data), ex=self.ttl)
                try:
                    self.redis.publish("job_updates", json.dumps({
                        "job_id": job_id,
                        "updates": updates,
                        "full_data": data
                    }))
                except Exception:
                    # publish is best-effort
                    pass
                return True
            return False
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            return False
        except Exception:
            return False

    def update_job_with_retry(self, job_id: str, updates: Dict[str, Any], retries: int = 5, backoff_base: float = 0.05) -> bool:
        """Retrying wrapper around `update_job` with exponential backoff.

        Returns True when persisted, False otherwise.
        Emits telemetry events (best-effort) for observability.
        """
        try:
            from montage_ai import telemetry
        except Exception:
            telemetry = None

        attempt = 0
        while attempt < retries:
            attempt += 1
            ok = self.update_job(job_id, updates)
            if telemetry:
                try:
                    telemetry.record_event("jobstore_update_attempt", {"job_id": job_id, "attempt": attempt, "success": bool(ok)})
                except Exception:
                    pass
            if ok:
                return True
            # exponential backoff
            try:
                time.sleep(backoff_base * (2 ** (attempt - 1)))
            except Exception:
                pass

        if telemetry:
            try:
                telemetry.record_event("jobstore_update_failed", {"job_id": job_id, "attempts": attempt})
            except Exception:
                pass
        return False

    def list_jobs(self, limit: int = 50) -> Dict[str, Any]:
        # Get latest job IDs
        job_ids = self.redis.zrevrange("jobs:timeline", 0, limit - 1)
        result = {}
        for jid in job_ids:
            job = self.get_job(jid)
            if job:
                result[jid] = job
        return result
