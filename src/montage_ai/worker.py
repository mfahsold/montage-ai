import os
import sys
import time
import redis
import json
from rq import Worker, Queue
from montage_ai.config import get_settings
from montage_ai.logger import logger

# Listen to both queues (preview jobs on 'default', heavy/standard jobs on 'heavy')
listen = ['default', 'heavy']

def start_worker():
    """
    Production-ready RQ worker with Redis Streams pattern.
    Based on fluxibri_core proven architecture.
    """
    settings = get_settings()
    redis_host = settings.session.redis_host or 'localhost'
    redis_port = settings.session.redis_port
    
    logger.info(f"🚀 Starting Montage AI RQ Worker")
    logger.info(f"📡 Redis: {redis_host}:{redis_port}")
    logger.info(f"📋 Queues: {listen}")
    
    # Create Redis connection with retry logic
    max_retries = 5
    retry_delay = 2
    
    conn = None
    for attempt in range(max_retries):
        try:
            conn = redis.Redis(
                host=redis_host,
                port=redis_port,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            conn.ping()
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
            break
        except (redis.ConnectionError, redis.TimeoutError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️  Redis connection attempt {attempt + 1}/{max_retries} failed: {e}")
                logger.info(f"⏳ Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"❌ Failed to connect to Redis after {max_retries} attempts")
                sys.exit(1)
    
    # Create queues with connection
    queues = [Queue(name, connection=conn) for name in listen]
    
    # Create worker with connection
    # Use hostname (pod name in k8s) for unique worker name
    import socket
    worker_name = os.environ.get("WORKER_NAME", f"montage-{socket.gethostname()}")
    worker = Worker(
        queues,
        connection=conn,
        name=worker_name
    )
    
    logger.info(f"🎬 Worker ready: {worker.name}")
    logger.info(f"⏱️  Listening for jobs...")
    
    # Start working (blocking)
    worker.work(
        with_scheduler=True,
        max_jobs=1000,  # Restart after 1000 jobs to prevent memory leaks
    )

if __name__ == '__main__':
    start_worker()
