#!/usr/bin/env python3
"""Submit jobs using specifically the 'raw' video files."""
import sys
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from redis import Redis
from rq import Queue
import time

# Redis endpoint (override via env)
cluster_ns = os.environ.get("CLUSTER_NAMESPACE", "montage-ai")
redis_host = os.environ.get("REDIS_HOST", f"redis.{cluster_ns}.svc.cluster.local")
redis_port = int(os.environ.get("REDIS_PORT", "6379"))

def submit_raw_job():
    print(f"Connecting to Redis at {redis_host}...")
    try:
        redis_conn = Redis(host=redis_host, port=redis_port, decode_responses=False)
        redis_conn.ping()
        print("✅ Connected to Redis")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("Trying with port-forward...")
        redis_conn = Redis(host='localhost', port=redis_port, decode_responses=False)
    
    q = Queue('default', connection=redis_conn)
    
    # Define the 3 raw video files
    video_files = [
        'raw_8ogEEWOt6N8.mp4',
        'raw_AE2P0GkCvi8.mp4',
        'raw_F-CoZthcAJQ.mp4'
    ]
    
    style = 'dynamic'
    job_id = f'raw_trailer_{int(time.time())}'
    
    print(f"\n📤 Submitting RAW trailer job (style: {style})...")
    from montage_ai.tasks import run_montage
    job = q.enqueue(
        run_montage,
        args=(job_id, style),
        kwargs={
            'options': {
                'prompt': 'Create a high-energy trailer using the three raw clips.',
                'stabilize': False,
                'upscale': False,
                'enhance': True,
                'video_files': video_files,
                'quality_profile': 'preview'  # Use preview for faster turnaround
            }
        },
        timeout='20m',
        result_ttl=500
    )
    
    print(f"   ✅ Job ID: {job.id}")
    print(f"\n✅ Submitted RAW job with {len(video_files)} files.")
    
    # Check queue status
    print(f"\n📊 Queue status:")
    print(f"   Queued: {len(q)}")
    print(f"   Started: {len(q.started_job_registry)}")

if __name__ == '__main__':
    submit_raw_job()
