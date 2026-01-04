import os
import redis
from rq import Worker, Queue, Connection
from src.montage_ai.config import get_settings

listen = ['default']

def start_worker():
    settings = get_settings()
    # Use REDIS_URL or construct from host/port
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    
    conn = redis.Redis(host=redis_host, port=redis_port)
    
    print(f"Starting RQ worker listening on {listen}...")
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()

if __name__ == '__main__':
    start_worker()
