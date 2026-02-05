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
import re
import subprocess
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from ..logger import logger
from ..config import get_settings

# Use lazy-initialized settings
settings = get_settings()


@dataclass
class JobSpec:
    """Specification for a distributed job."""
    name: str
    namespace: str = settings.cluster.namespace
    parallelism: int = 4
    completions: int = 4
    image: str = settings.cluster.image_full
    env: Dict[str, str] = None
    resources: Dict[str, Any] = None
    image_pull_secret: Optional[str] = settings.cluster.image_pull_secret


@dataclass
class JobStatus:
    """Status of a distributed job."""
    name: str
    active: int
    succeeded: int
    failed: int
    completion_time: Optional[str]
    conditions: List[Dict[str, str]]
    is_not_found: bool = False

    @property
    def is_complete(self) -> bool:
        """Check if job has finished (success or failure)."""
        if self.is_not_found:
            return False
        return (self.succeeded >= 1 or self.failed >= 1)

    @property
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        if self.is_not_found:
            return False
        return self.succeeded >= 1 and self.failed == 0


class JobSubmitter:
    """
    Submit and manage distributed Kubernetes jobs using the official K8s client.
    """

    def __init__(self, namespace: Optional[str] = None, kubeconfig: Optional[str] = None):
        self.namespace = namespace or settings.cluster.namespace
        self.kubeconfig = kubeconfig or os.environ.get("KUBECONFIG")
        self.image = settings.cluster.image_full
        
        # Initialize K8s client
        try:
            if self.kubeconfig:
                config.load_kube_config(config_file=self.kubeconfig)
            else:
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()
            
            self.batch_v1 = client.BatchV1Api()
            self.core_v1 = client.CoreV1Api()
            # Custom API for CRDs (JobSet)
            try:
                self.custom_api = client.CustomObjectsApi()
            except Exception:
                self.custom_api = None
            logger.info(f"✅ Kubernetes client initialized (Namespace: {self.namespace})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kubernetes client: {e}")
            self.batch_v1 = None
            self.core_v1 = None
            self.custom_api = None

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
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:12]
        timestamp = int(time.time()) % 1000000
        return f"{prefix}-{timestamp}-{hash_suffix}"

    def _sanitize_k8s_name(self, name: str, max_len: int = 63) -> str:
        """Make a string safe for Kubernetes resource names (RFC 1123)."""
        safe = re.sub(r"[^a-z0-9.-]+", "-", name.lower())
        safe = safe.strip("-.")
        if not safe:
            safe = f"job-{int(time.time())}"
        if len(safe) > max_len:
            safe = safe[:max_len].rstrip("-.")
        return safe

    def submit_scene_detection(
        self,
        video_paths: List[str],
        parallelism: int = 4,
        threshold: float = 27.0,
        output_dir: str = "/data/output/scene_cache"
    ) -> JobSpec:
        """
        Submit distributed scene detection job using K8s API.
        """
        job_id = self._generate_job_id("scene-detect", *video_paths)
        
        # Create ConfigMap via API
        config_map_name = f"scene-detect-config-{job_id}"
        cm_body = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(name=config_map_name, namespace=self.namespace),
            data={
                "video_paths": ",".join(video_paths),
                "threshold": str(threshold),
                "output_dir": output_dir,
                "job_id": job_id
            }
        )
        
        try:
            self.core_v1.create_namespaced_config_map(self.namespace, cm_body)
        except ApiException as e:
            if getattr(e, "status", 0) != 409: # Ignore already exists
                logger.error(f"Failed to create ConfigMap: {e}")
                raise

        # Build Job Spec using the newly refactored submit_generic_job
        env = {
            "VIDEO_PATH": video_paths[0] if len(video_paths) == 1 else "/data/input",
            "VIDEOS": ",".join(video_paths),
            "OUTPUT_DIR": output_dir,
            "JOB_ID": job_id,
            "SCENE_DETECT_CONFIG": config_map_name
        }
        
        return self.submit_generic_job(
            job_id=job_id,
            command=["python", "-m", "montage_ai.cluster.distributed_scene_detection", "--shard-mode"],
            parallelism=parallelism,
            component="scene-detection",
            env=env,
            tier="medium"
        )

    def _supports_jobset(self) -> bool:
        """Detect whether JobSet CRD is available in the cluster."""
        if not getattr(settings.cluster, "use_jobset", False):
            return False
        if not self.custom_api:
            return False
        try:
            # Try to list jobsets in the namespace as a capability probe
            self.custom_api.list_namespaced_custom_object(group="batch.jobset.sigs.k8s.io", version="v1alpha1", namespace=self.namespace, plural="jobsets", limit=1)
            return True
        except ApiException as e:
            logger.info("JobSet CRD not available or accessible: %s", e)
            return False
        except Exception as e:
            logger.info("Unexpected error checking JobSet CRD: %s", e)
            return False

    def get_job_status(self, job_name: str) -> Optional[JobStatus]:
        """Get status of a job using K8s API."""
        if not self.batch_v1:
            # Fallback to kubectl if client failed
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

        try:
            job = self.batch_v1.read_namespaced_job_status(job_name, self.namespace)
            status = job.status
            return JobStatus(
                name=job_name,
                active=status.active or 0,
                succeeded=status.succeeded or 0,
                failed=status.failed or 0,
                completion_time=status.completion_time.isoformat() if status.completion_time else None,
                conditions=[{
                    "type": c.type,
                    "status": c.status,
                    "reason": c.reason,
                    "message": c.message
                } for c in (status.conditions or [])]
            )
        except ApiException as e:
            if e.status == 404:
                return JobStatus(name=job_name, active=0, succeeded=0, failed=0, completion_time=None, conditions=[], is_not_found=True)
            logger.error(f"Error getting job status: {e}")
            return None

    def is_job_successful(self, job_name: str) -> bool:
        """Check if a job completed successfully."""
        status = self.get_job_status(job_name)
        if not status:
            return False
        return status.is_successful

    def list_jobs(self, label_selector: str = "app.kubernetes.io/name=montage-ai") -> List[Dict]:
        """List all montage-ai jobs."""
        if not self.batch_v1:
            result = self._kubectl("get", "jobs", "-l", label_selector, "-o", "json")
            if result.returncode != 0:
                return []
            data = json.loads(result.stdout)
            return data.get("items", [])

        try:
            jobs = self.batch_v1.list_namespaced_job(self.namespace, label_selector=label_selector)
            return [j.to_dict() for j in jobs.items]
        except ApiException as e:
            logger.error(f"Error listing jobs: {e}")
            return []

    def delete_job(self, job_name: str, cascade: bool = True) -> bool:
        """Delete a job and its pods."""
        if not self.batch_v1:
            args = ["delete", "job", job_name]
            if cascade:
                args.append("--cascade=foreground")
            result = self._kubectl(*args)
            return result.returncode == 0

        try:
            # Use background propagation for cleaner deletion usually, but foreground ensures it's gone
            body = client.V1DeleteOptions(propagation_policy='Foreground' if cascade else 'Background')
            self.batch_v1.delete_namespaced_job(job_name, self.namespace, body=body)
            return True
        except ApiException as e:
            if e.status == 404:
                return True
            logger.error(f"Error deleting job: {e}")
            return False

    def wait_for_job(self, job_name: str, timeout_seconds: int = 3600, poll_interval: int = 5) -> bool:
        """Wait for a job to complete."""
        logger.info(f"⏳ Waiting for job {job_name} to complete (timeout: {timeout_seconds}s)...")
        
        start_time = time.time()
        not_found_count = 0
        max_not_found = 3  # Allow for some K8s API propagation delay

        while time.time() - start_time < timeout_seconds:
            status = self.get_job_status(job_name)
            
            if status is None:
                not_found_count += 1
                if not_found_count >= max_not_found:
                    logger.warning(f"⚠️ Job {job_name} not found after {not_found_count} attempts.")
                    return False
                logger.info(f"  (Job {job_name} not found yet, retrying... {not_found_count}/{max_not_found})")
                time.sleep(2)
                continue
            
            # Reset not_found if we found it once
            not_found_count = 0

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

    def submit_generic_job(
        self,
        job_id: str,
        command: List[str],
        parallelism: int = 1,
        completions: Optional[int] = None,
        component: str = "generic",
        env: Optional[Dict[str, str]] = None,
        resources: Optional[Dict[str, Any]] = None,
        tier: str = "small",
        image_pull_secret: Optional[str] = None,
        node_selector: Optional[Dict[str, str]] = None,
        affinity: Optional[client.V1Affinity] = None
    ) -> JobSpec:
        """
        Submit a generic completion-indexed Job using K8s API.
        """
        if completions is None:
            completions = parallelism

        job_name = self._sanitize_k8s_name(job_id)
            
        pull_secret = image_pull_secret or settings.cluster.image_pull_secret

        # Architecture targeting (default to amd64 as per latest tag)
        if node_selector is None:
            node_selector = {"kubernetes.io/arch": "amd64"}
        
        # Build Job Manifest using library objects (SOTA approach)
        env_vars = [
            client.V1EnvVar(name="SHARD_INDEX", value_from=client.V1EnvVarSource(
                field_ref=client.V1ObjectFieldSelector(field_path="metadata.annotations['batch.kubernetes.io/job-completion-index']")
            )),
            client.V1EnvVar(name="SHARD_COUNT", value=str(completions)),
            client.V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
            # Unified deployment mode (preferred) + legacy CLUSTER_MODE for backwards compat
            client.V1EnvVar(name="DEPLOYMENT_MODE", value="cluster"),
            client.V1EnvVar(name="CLUSTER_MODE", value="true"),
        ]
        
        if env:
            for key, value in env.items():
                env_vars.append(client.V1EnvVar(name=key, value=str(value)))

        # Resource limits/requests (Use tier-based defaults if not specfied)
        if not resources:
            tier_config = settings.cluster.tiers.get(tier, settings.cluster.tiers["small"])
            req_cpu = tier_config["requests"]["cpu"]
            req_mem = tier_config["requests"]["memory"]
            lim_cpu = tier_config["limits"]["cpu"]
            lim_mem = tier_config["limits"]["memory"]
        else:
            req_cpu = resources.get("requests_cpu", "1")
            req_mem = resources.get("requests_memory", "2Gi")
            lim_cpu = resources.get("limits_cpu", "2")
            lim_mem = resources.get("limits_memory", "4Gi")

        pvc_input = settings.cluster.pvc_input or ""
        pvc_output = settings.cluster.pvc_output or ""
        pvc_music = settings.cluster.pvc_music or ""
        pvc_assets = settings.cluster.pvc_assets or ""
        use_split_pvcs = any([pvc_input, pvc_output, pvc_music, pvc_assets])

        volume_mounts = []
        volumes = []

        if use_split_pvcs:
            input_claim = pvc_input or settings.cluster.pvc_name
            output_claim = pvc_output or settings.cluster.pvc_name
            volumes.append(client.V1Volume(
                name="input",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=input_claim
                )
            ))
            volumes.append(client.V1Volume(
                name="output",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=output_claim
                )
            ))
            volume_mounts.extend([
                client.V1VolumeMount(name="input", mount_path="/data/input", read_only=True),
                client.V1VolumeMount(name="output", mount_path="/data/output")
            ])

            if pvc_music:
                volumes.append(client.V1Volume(
                    name="music",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=pvc_music
                    )
                ))
                volume_mounts.append(
                    client.V1VolumeMount(name="music", mount_path="/data/music", read_only=True)
                )

            if pvc_assets:
                volumes.append(client.V1Volume(
                    name="assets",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=pvc_assets
                    )
                ))
                volume_mounts.append(
                    client.V1VolumeMount(name="assets", mount_path="/data/assets", read_only=True)
                )
        else:
            volumes.append(client.V1Volume(
                name="data",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=settings.cluster.pvc_name
                )
            ))
            volume_mounts.extend([
                client.V1VolumeMount(name="data", mount_path="/data/input", sub_path="input", read_only=True),
                client.V1VolumeMount(name="data", mount_path="/data/output", sub_path="output")
            ])

        container = client.V1Container(
            name="worker",
            image=self.image,
            image_pull_policy="IfNotPresent",
            command=command,
            env=env_vars,
            resources=client.V1ResourceRequirements(
                requests={"cpu": req_cpu, "memory": req_mem},
                limits={"cpu": lim_cpu, "memory": lim_mem}
            ),
            volume_mounts=volume_mounts
        )

        pod_spec = client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            volumes=volumes,
            node_selector=node_selector,
            affinity=affinity
        )
        
        if pull_secret:
            pod_spec.image_pull_secrets = [client.V1LocalObjectReference(name=pull_secret)]

        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                labels={
                    "app.kubernetes.io/name": "montage-ai",
                    "app.kubernetes.io/component": component,
                    "fluxibri.ai/project": "montage-ai",
                    "fluxibri.ai/tier": tier,
                    "fluxibri.ai/architecture": node_selector.get("kubernetes.io/arch", "amd64"),
                    "montage-ai.fluxibri.dev/job-id": job_id
                }
            ),
            spec=client.V1JobSpec(
                completion_mode="Indexed",
                completions=completions,
                parallelism=parallelism,
                backoff_limit=2,
                ttl_seconds_after_finished=1800,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels={
                        "app.kubernetes.io/name": "montage-ai",
                        "app.kubernetes.io/component": component,
                        "fluxibri.ai/project": "montage-ai",
                        "fluxibri.ai/tier": tier
                    }),
                    spec=pod_spec
                )
            )
        )

        # If JobSet is available and desired, prefer JobSet for coordinated distributed workloads
        if self._supports_jobset():
            jobset = {
                "apiVersion": "batch.jobset.sigs.k8s.io/v1alpha1",
                "kind": "JobSet",
                "metadata": {
                    "name": job_name,
                    "namespace": self.namespace,
                    "labels": job.metadata.labels
                },
                "spec": {
                    "completions": completions,
                    "parallelism": parallelism,
                    "template": {
                        "metadata": {"labels": job.spec.template.metadata.labels},
                        "spec": job.spec.template.spec.to_dict()
                    }
                }
            }
            try:
                self.custom_api.create_namespaced_custom_object(
                    group="batch.jobset.sigs.k8s.io",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="jobsets",
                    body=jobset
                )
                logger.info(f"Successfully submitted JobSet for {component}: {job_name}")
                return JobSpec(name=job_name, namespace=self.namespace, parallelism=parallelism, completions=completions)
            except ApiException as e:
                logger.warning(f"JobSet submission failed, falling back to Job: {e}")
                # fall through to Job creation

        try:
            self.batch_v1.create_namespaced_job(self.namespace, job)
            logger.info(f"Successfully submitted {component} job via API: {job_name}")
            return JobSpec(name=job_name, namespace=self.namespace, parallelism=parallelism, completions=completions)
        except ApiException as e:
            logger.error(f"Failed to submit job: {e.body}")
            raise RuntimeError(f"Job creation failed: {e}")

    def submit_gpu_encoding(
        self,
        input_path: str,
        output_path: str,
        codec: str = "h264_amf",
        quality: str = "standard"
    ) -> JobSpec:
        """
        Submit GPU encoding job to specialized heavy nodes via K8s API.
        """
        job_id = self._generate_job_id("encode", input_path)
        
        # Build node affinity for GPUs (prefer nodes with GPUs)
        affinity = client.V1Affinity(
            node_affinity=client.V1NodeAffinity(
                preferred_during_scheduling_ignored_during_execution=[
                    client.V1PreferredSchedulingTerm(
                        weight=100,
                        preference=client.V1NodeSelectorTerm(
                            match_expressions=[
                                client.V1NodeSelectorRequirement(
                                    key="fluxibri.ai/gpu-enabled",
                                    operator="In",
                                    values=["true", "present"]
                                )
                            ]
                        )
                    )
                ]
            )
        )

        return self.submit_generic_job(
            job_id=job_id,
            command=[
                "python", "-m", "montage_ai.core.render_engine",
                "--render-distributed", input_path, output_path,
                "--codec", codec, "--quality", quality
            ],
            parallelism=1,
            component="encoder",
            tier="gpu",
            affinity=affinity
        )
