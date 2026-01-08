#!/usr/bin/env python3
"""Submit test jobs to RQ directly"""
import sys
sys.path.insert(0, '/home/codeai/montage-ai/src')

from redis import Redis
from rq import Queue
import time

redis_host = 'redis.montage-ai.svc.cluster.local'

def submit_jobs():
    print(f"Connecting to Redis at {redis_host}...")
    try:
        redis_conn = Redis(host=redis_host, port=6379, decode_responses=False)
        redis_conn.ping()
        print("✅ Connected to Redis")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("Trying with port-forward...")
        redis_conn = Redis(host='localhost', port=6379, decode_responses=False)
    
    q = Queue('default', connection=redis_conn)
    
    styles = ['dynamic', 'hitchcock', 'action', 'wes_anderson']
    job_ids = []
    
    for style in styles:
        print(f"\n📤 Submitting {style} job...")
        from montage_ai.tasks import run_montage
        job = q.enqueue(
            run_montage,
            args=(f'test_{style}_{int(time.time())}', style),
            kwargs={
                'options': {
                    'prompt': '',
                    'stabilize': False,
                    'upscale': False,
                    'enhance': True
                }
            },
            timeout='10m',
            result_ttl=500
        )
        job_ids.append(job.id)
        print(f"   ✅ Job ID: {job.id}")
        time.sleep(1)
    
    print(f"\n✅ Submitted {len(job_ids)} jobs")
    print(f"   Job IDs: {', '.join(job_ids)}")
    
    # Check queue status
    print(f"\n📊 Queue status:")
    print(f"   Queued: {len(q)}")
    print(f"   Started: {len(q.started_job_registry)}")
    print(f"   Finished: {len(q.finished_job_registry)}")
    print(f"   Failed: {len(q.failed_job_registry)}")

if __name__ == '__main__':
    submit_jobs()
