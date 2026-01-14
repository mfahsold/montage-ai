"""
Cluster Orchestration Validator

Tests if the Montage AI Orchestrator can successfully:
1. Connect to Kubernetes API
2. Submit a Dummy Job
3. Monitor for completion
4. Cleanup

Usage:
    python scripts/validate_cluster_orchestration.py
"""

import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from montage_ai.cluster.job_submitter import JobSubmitter
from montage_ai.logger import logger

def validate():
    logger.info("🧪 Validating Cluster Orchestration...")
    
    submitter = JobSubmitter()
    
    # Check connectivity
    try:
        nodes = submitter._kubectl("get", "nodes")
        if nodes.returncode != 0:
            logger.error(f"❌ Cannot connect to K8s: {nodes.stderr}")
            return False
        logger.info("✅ K8s Connectivity: OK")
    except Exception as e:
        logger.error(f"❌ Exception during connectivity check: {e}")
        return False

    # Submit a dummy job
    job_id = f"test-orchestration-{int(time.time())}"
    command = ["echo", "Orchestration test successful"]
    
    logger.info(f"🚀 Submitting test job: {job_id}")
    try:
        job_spec = submitter.submit_generic_job(
            job_id=job_id,
            command=command,
            parallelism=1,
            component="test-orchestration"
        )
        
        # Wait for completion
        success = submitter.wait_for_job(job_spec.name, timeout_seconds=120)
        
        if success:
            logger.info("✅ Test Job completed successfully")
        else:
            logger.error("❌ Test Job failed or timed out")
            
        # Cleanup
        logger.info(f"🧹 Deleting test job {job_id}")
        submitter.delete_job(job_spec.name)
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Exception during job execution: {e}")
        return False

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
