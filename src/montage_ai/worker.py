import os
import redis
from rq import Worker, Queue, Connection
from montage_ai.config import get_settings

listen = ['default']

def start_worker():
    settings = get_settings()
    # Use centralized session config (fallback to localhost if unset)
    redis_host = settings.session.redis_host or 'localhost'
    redis_port = settings.session.redis_port

    conn = redis.Redis(host=redis_host, port=redis_port)
    
    print(f"Starting RQ worker listening on {listen}...")
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()

if __name__ == '__main__':
    start_worker()
