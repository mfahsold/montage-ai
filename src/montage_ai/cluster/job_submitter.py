"""
K8s Job Submitter - Submit and monitor distributed jobs

Provides programmatic interface for:
- Creating distributed scene detection jobs
- Submitting GPU encoding jobs
- Monitoring job progress
- Collecting job results

Usage:
    from montage_ai.cluster import JobSubmitter

    submitter = JobSubmitter()

    # Submit distributed scene detection
    job = submitter.submit_scene_detection(
        video_paths=["/data/input/video1.mp4", "/data/input/video2.mp4"],
        parallelism=4
    )

    # Wait for completion
    submitter.wait_for_job(job.name)

    # Get results
    scenes = submitter.get_scene_results(job.name)
"""

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..logger import logger


@dataclass
class JobSpec:
    """Specification for a distributed job."""
    name: str
    namespace: str = "montage-ai"
    parallelism: int = 4
    completions: int = 4
    image: str = os.environ.get("IMAGE_FULL", os.environ.get("MONTAGE_IMAGE", "ghcr.io/mfahsold/montage-ai:latest"))
    env: Dict[str, str] = None
    resources: Dict[str, Any] = None


@dataclass
class JobStatus:
    """Status of a distributed job."""
    name: str
    active: int
    succeeded: int
    failed: int
    completion_time: Optional[str]
    conditions: List[Dict[str, str]]

    @property
    def is_complete(self) -> bool:
        return self.succeeded >= 1 or self.failed >= 1

    @property
    def is_successful(self) -> bool:
        return self.succeeded >= 1 and self.failed == 0


class JobSubmitter:
    """
    Submit and manage distributed Kubernetes jobs.

    Requires kubectl access to the cluster.
    """

    def __init__(self, namespace: str = "montage-ai", kubeconfig: Optional[str] = None):
        self.namespace = namespace
        self.kubeconfig = kubeconfig or os.environ.get("KUBECONFIG")

    def _kubectl(self, *args, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run kubectl command."""
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        cmd.extend(["-n", self.namespace])
        cmd.extend(args)

        return subprocess.run(cmd, capture_output=capture_output, text=True)

    def _generate_job_id(self, prefix: str, *args) -> str:
        """Generate unique job ID based on inputs."""
        content = "-".join(str(a) for a in args)
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = int(time.time()) % 100000
        return f"{prefix}-{timestamp}-{hash_suffix}"

    def submit_scene_detection(
        self,
        video_paths: List[str],
        parallelism: int = 4,
        threshold: float = 27.0,
        output_dir: str = "/data/output/scene_cache"
    ) -> JobSpec:
        """
        Submit distributed scene detection job.

        Args:
            video_paths: List of video paths to analyze
            parallelism: Number of parallel workers
            threshold: Scene detection threshold
            output_dir: Output directory for results

        Returns:
            JobSpec with job details
        """
        job_id = self._generate_job_id("scene-detect", *video_paths)

        # Create ConfigMap with job config
        config_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: scene-detect-config-{job_id}
  namespace: {self.namespace}
data:
  video_paths: "{','.join(video_paths)}"
  threshold: "{threshold}"
  output_dir: "{output_dir}"
  job_id: "{job_id}"
"""

        # Create Job manifest
        image_var = os.environ.get("IMAGE_FULL", os.environ.get("MONTAGE_IMAGE", "ghcr.io/mfahsold/montage-ai:latest"))
        job_yaml = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_id}
  namespace: {self.namespace}
  labels:
    app.kubernetes.io/name: montage-ai
    app.kubernetes.io/component: scene-detection
    montage-ai.fluxibri.dev/job-id: "{job_id}"
spec:
  completionMode: Indexed
  completions: {parallelism}
  parallelism: {parallelism}
  backoffLimit: 2
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app.kubernetes.io/component: scene-detection
                topologyKey: kubernetes.io/hostname
      containers:
        - name: scene-detect
          image: {image_var}
          imagePullPolicy: IfNotPresent
          command:
            - python
            - -m
            - montage_ai.cluster.distributed_scene_detection
            - --shard-mode
          env:
            - name: SHARD_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
            - name: SHARD_COUNT
              value: "{parallelism}"
            - name: VIDEO_PATH
              value: "{video_paths[0] if len(video_paths) == 1 else '/data/input'}"
            - name: VIDEOS
              value: "{','.join(video_paths)}"
            - name: OUTPUT_DIR
              value: "{output_dir}"
            - name: JOB_ID
              value: "{job_id}"
            - name: PYTHONUNBUFFERED
              value: "1"
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          volumeMounts:
            - name: input
              mountPath: /data/input
              readOnly: true
            - name: output
              mountPath: /data/output
      volumes:
        - name: input
          persistentVolumeClaim:
            claimName: montage-input
        - name: output
          persistentVolumeClaim:
            claimName: montage-output
"""

        # Apply ConfigMap
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=config_yaml,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Failed to create ConfigMap: {result.stderr}")
            raise RuntimeError(f"ConfigMap creation failed: {result.stderr}")

        # Apply Job
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=job_yaml,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Failed to create Job: {result.stderr}")
            raise RuntimeError(f"Job creation failed: {result.stderr}")

        logger.info(f"✅ Submitted distributed scene detection job: {job_id}")
        logger.info(f"   Parallelism: {parallelism}, Videos: {len(video_paths)}")

        return JobSpec(
            name=job_id,
            namespace=self.namespace,
            parallelism=parallelism,
            completions=parallelism,
            env={"JOB_ID": job_id, "VIDEOS": ",".join(video_paths)}
        )

    def get_job_status(self, job_name: str) -> Optional[JobStatus]:
        """Get status of a job."""
        result = self._kubectl(
            "get", "job", job_name,
            "-o", "json"
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        status = data.get("status", {})

        return JobStatus(
            name=job_name,
            active=status.get("active", 0),
            succeeded=status.get("succeeded", 0),
            failed=status.get("failed", 0),
            completion_time=status.get("completionTime"),
            conditions=status.get("conditions", [])
        )

    def wait_for_job(
        self,
        job_name: str,
        timeout: int = 3600,
        poll_interval: int = 10
    ) -> JobStatus:
        """
        Wait for job completion.

        Args:
            job_name: Name of the job
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Final JobStatus
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_name)
            if status is None:
                raise RuntimeError(f"Job {job_name} not found")

            if status.is_complete:
                if status.is_successful:
                    logger.info(f"✅ Job {job_name} completed successfully")
                else:
                    logger.warning(f"⚠️ Job {job_name} failed")
                return status

            logger.info(
                f"⏳ Job {job_name}: {status.succeeded} succeeded, "
                f"{status.active} active, {status.failed} failed"
            )
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_name} did not complete within {timeout}s")

    def get_scene_results(self, job_id: str, output_dir: str = "/data/output/scene_cache") -> List[dict]:
        """
        Get aggregated scene detection results.

        Args:
            job_id: Job ID
            output_dir: Output directory where results are stored

        Returns:
            List of detected scenes
        """
        # Check for aggregated results file
        result_path = Path(output_dir) / f"scenes_{job_id}.json"

        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
                return data.get("scenes", [])

        # If not aggregated yet, trigger aggregation
        from .distributed_scene_detection import aggregate_shard_results
        return aggregate_shard_results(output_dir, job_id)

    def submit_gpu_encoding(
        self,
        input_path: str,
        output_path: str,
        codec: str = "h264_amf",
        quality: str = "standard"
    ) -> JobSpec:
        """
        Submit GPU encoding job.

        Routes to best available GPU node.
        """
        job_id = self._generate_job_id("encode", input_path)

        job_yaml = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_id}
  namespace: {self.namespace}
  labels:
    app.kubernetes.io/name: montage-ai
    app.kubernetes.io/component: encoder
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                  - key: amd.com/gpu
                    operator: In
                    values: ["present", "true"]
            - weight: 50
              preference:
                matchExpressions:
                  - key: accelerator
                    operator: In
                    values: ["nvidia"]
      containers:
        - name: encoder
          image: {image_var}
          command:
            - ffmpeg
            - -i
            - "{input_path}"
            - -c:v
            - "{codec}"
            - -preset
            - fast
            - -y
            - "{output_path}"
          env:
            - name: FFMPEG_HWACCEL
              value: "auto"
          resources:
            requests:
              cpu: "4"
              memory: "8Gi"
            limits:
              cpu: "8"
              memory: "24Gi"
          volumeMounts:
            - name: output
              mountPath: /data/output
      volumes:
        - name: output
          persistentVolumeClaim:
            claimName: montage-output
"""

        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=job_yaml,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Job creation failed: {result.stderr}")

        logger.info(f"✅ Submitted GPU encoding job: {job_id}")
        return JobSpec(name=job_id, namespace=self.namespace)

    def list_jobs(self, label_selector: str = "app.kubernetes.io/name=montage-ai") -> List[Dict]:
        """List all montage-ai jobs."""
        result = self._kubectl(
            "get", "jobs",
            "-l", label_selector,
            "-o", "json"
        )

        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        return data.get("items", [])

    def delete_job(self, job_name: str, cascade: bool = True) -> bool:
        """Delete a job and its pods."""
        args = ["delete", "job", job_name]
        if cascade:
            args.append("--cascade=foreground")

        result = self._kubectl(*args)
        return result.returncode == 0

    def wait_for_job(self, job_name: str, timeout_seconds: int = 3600, poll_interval: int = 10) -> bool:
        """Wait for a job to complete."""
        logger.info(f"⏳ Waiting for job {job_name} to complete (timeout: {timeout_seconds}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            status = self.get_job_status(job_name)
            if status is None:
                logger.warning(f"⚠️ Job {job_name} not found.")
                return False
                
            if status.is_complete:
                if status.is_successful:
                    logger.info(f"✅ Job {job_name} succeeded.")
                    return True
                else:
                    logger.error(f"❌ Job {job_name} failed.")
                    return False
            
            time.sleep(poll_interval)
            
        logger.error(f"⌛ Job {job_name} timed out after {timeout_seconds}s.")
        return False

    def get_job_status(self, job_name: str) -> Optional[JobStatus]:
        """Get the current status of a job."""
        result = self._kubectl("get", "job", job_name, "-o", "json")
        if result.returncode != 0:
            return None
            
        data = json.loads(result.stdout)
        status = data.get("status", {})
        
        return JobStatus(
            name=job_name,
            active=status.get("active", 0),
            succeeded=status.get("succeeded", 0),
            failed=status.get("failed", 0),
            completion_time=status.get("completionTime"),
            conditions=status.get("conditions", [])
        )

    def is_job_successful(self, job_name: str) -> bool:
        """Check if a job has completed successfully."""
        status = self.get_job_status(job_name)
        return status is not None and status.is_successful

    def submit_generic_job(
        self,
        job_id: str,
        command: List[str],
        parallelism: int = 1,
        completions: Optional[int] = None,
        component: str = "generic",
        env: Optional[Dict[str, str]] = None,
        resources: Optional[Dict[str, Any]] = None
    ) -> JobSpec:
        """
        Submit a generic completion-indexed Job.
        """
        if completions is None:
            completions = parallelism
            
        image_var = os.environ.get("IMAGE_FULL", os.environ.get("MONTAGE_IMAGE", "ghcr.io/mfahsold/montage-ai:latest"))
        namespace = self.namespace
        
        # Build env vars
        env_section = ""
        if env:
            for key, value in env.items():
                env_section += f"""
            - name: {key}
              value: "{value}" """

        # Kubernetes Job Manifest
        job_yaml = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_id}
  namespace: {namespace}
  labels:
    app.kubernetes.io/name: montage-ai
    app.kubernetes.io/component: {component}
    montage-ai.fluxibri.dev/job-id: "{job_id}"
spec:
  completionMode: Indexed
  completions: {completions}
  parallelism: {parallelism}
  backoffLimit: 2
  ttlSecondsAfterFinished: 1800
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: worker
          image: {image_var}
          imagePullPolicy: IfNotPresent
          command: {json.dumps(command)}
          env:
            - name: SHARD_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
            - name: SHARD_COUNT
              value: "{completions}"
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: CLUSTER_MODE
              value: "true" {env_section}
          resources:
            requests:
              cpu: "{resources.get('cpu', '2') if resources else '2'}"
              memory: "{resources.get('memory', '4Gi') if resources else '4Gi'}"
            limits:
              cpu: "{resources.get('limit_cpu', '4') if resources else '4'}"
              memory: "{resources.get('limit_memory', '8Gi') if resources else '8Gi'}"
          volumeMounts:
            - name: input
              mountPath: /data/input
              readOnly: true
            - name: output
              mountPath: /data/output
      volumes:
        - name: input
          persistentVolumeClaim:
            claimName: montage-input
        - name: output
          persistentVolumeClaim:
            claimName: montage-output
"""

        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=job_yaml,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Job creation failed: {result.stderr}")

        logger.info(f"✅ Submitted {component} job: {job_id}")
        return JobSpec(name=job_id, namespace=namespace, parallelism=parallelism, completions=completions)
