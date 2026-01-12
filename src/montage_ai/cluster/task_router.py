"""
Task Router - Intelligent Job Distribution

Routes montage-ai tasks to optimal nodes based on:
- Hardware capabilities (GPU, CPU, memory)
- Current node load
- Task requirements (resolution, complexity)

Usage:
    from montage_ai.cluster import TaskRouter, TaskType

    router = TaskRouter()

    # Route a single task
    node = router.route_task(TaskType.GPU_ENCODING, resolution=(3840, 2160))

    # Create distributed jobs
    jobs = router.create_parallel_jobs(
        task=TaskType.CPU_SCENE_DETECTION,
        video_paths=["/data/input/video1.mp4", "/data/input/video2.mp4"],
        resolution=(3840, 2160)
    )
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from .node_capabilities import (
    ClusterManager,
    NodeCapability,
    TaskType,
    get_cluster_manager,
)
from ..logger import logger


@dataclass
class DistributedJob:
    """Represents a job to be executed on a specific node."""
    node: str
    task_type: TaskType
    items: List[str]  # Video paths, segment IDs, etc.
    shard_index: int
    shard_count: int
    env_vars: Dict[str, str]

    def to_k8s_env(self) -> List[Dict[str, str]]:
        """Convert to Kubernetes env var format."""
        envs = [
            {"name": "SHARD_INDEX", "value": str(self.shard_index)},
            {"name": "SHARD_COUNT", "value": str(self.shard_count)},
            {"name": "TARGET_NODE", "value": self.node},
            {"name": "TASK_TYPE", "value": self.task_type.name},
        ]
        for key, value in self.env_vars.items():
            envs.append({"name": key, "value": str(value)})
        return envs


class TaskRouter:
    """
    Routes tasks to optimal cluster nodes.

    Considers:
    - Node hardware capabilities
    - Task requirements
    - Load balancing
    """

    def __init__(self, cluster: Optional[ClusterManager] = None):
        self.cluster = cluster or get_cluster_manager()

    def route_task(
        self,
        task: TaskType,
        resolution: Tuple[int, int] = (1920, 1080),
        prefer_node: Optional[str] = None
    ) -> Optional[NodeCapability]:
        """
        Route a single task to the best available node.

        Args:
            task: Type of task to route
            resolution: Video resolution
            prefer_node: Preferred node name (if available and capable)

        Returns:
            Best node for the task, or None if no capable nodes
        """
        if prefer_node and prefer_node in self.cluster._nodes:
            node = self.cluster._nodes[prefer_node]
            if node.can_handle_task(task, resolution):
                return node

        return self.cluster.get_best_node_for_task(task, resolution)

    def create_parallel_jobs(
        self,
        task: TaskType,
        items: List[str],
        resolution: Tuple[int, int] = (1920, 1080),
        max_shards: Optional[int] = None,
        extra_env: Optional[Dict[str, str]] = None
    ) -> List[DistributedJob]:
        """
        Create parallel jobs distributed across capable nodes.

        Args:
            task: Type of task to distribute
            items: List of items to process (video paths, segment IDs, etc.)
            resolution: Video resolution for capability matching
            max_shards: Maximum number of parallel shards
            extra_env: Additional environment variables

        Returns:
            List of DistributedJob objects
        """
        if not items:
            return []

        # Get distribution
        distribution = self.cluster.get_parallel_distribution(
            task, len(items), resolution
        )

        if not distribution:
            logger.warning(f"No nodes capable of task {task.name}")
            return []

        # Limit shards if requested
        if max_shards and len(distribution) > max_shards:
            # Keep only top N nodes by priority
            nodes = self.cluster.get_nodes_for_task(task, resolution, max_nodes=max_shards)
            distribution = {n.name: [] for n in nodes}
            for i, item_idx in enumerate(range(len(items))):
                node = nodes[i % len(nodes)]
                distribution[node.name].append(item_idx)

        # Create jobs
        jobs = []
        shard_index = 0
        shard_count = len([d for d in distribution.values() if d])

        for node_name, item_indices in distribution.items():
            if not item_indices:
                continue

            job_items = [items[i] for i in item_indices]

            env_vars = extra_env.copy() if extra_env else {}
            env_vars["ITEMS"] = ",".join(job_items)
            env_vars["ITEM_COUNT"] = str(len(job_items))

            # Add node-specific encoder info
            node = self.cluster._nodes.get(node_name)
            if node:
                if node.encoder:
                    env_vars["FFMPEG_ENCODER"] = node.encoder
                if node.hwaccel:
                    env_vars["FFMPEG_HWACCEL"] = node.hwaccel

            jobs.append(DistributedJob(
                node=node_name,
                task_type=task,
                items=job_items,
                shard_index=shard_index,
                shard_count=shard_count,
                env_vars=env_vars
            ))
            shard_index += 1

        logger.info(
            f"Created {len(jobs)} parallel jobs for {task.name} "
            f"({len(items)} items across {shard_count} nodes)"
        )

        return jobs

    def get_encoding_node(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        prefer_gpu: bool = True
    ) -> Tuple[Optional[NodeCapability], str]:
        """
        Get the best node for video encoding.

        Returns:
            (node, encoder_name) tuple
        """
        if prefer_gpu:
            node = self.route_task(TaskType.GPU_ENCODING, resolution)
            if node and node.encoder:
                return node, node.encoder

        # Fallback to CPU encoding
        node = self.route_task(TaskType.FINAL_RENDER, resolution)
        return node, "libx264"

    def print_routing_plan(
        self,
        task: TaskType,
        items: List[str],
        resolution: Tuple[int, int] = (1920, 1080)
    ):
        """Print a human-readable routing plan."""
        jobs = self.create_parallel_jobs(task, items, resolution)

        print(f"\n{'='*60}")
        print(f"ROUTING PLAN: {task.name}")
        print(f"{'='*60}")
        print(f"Items: {len(items)}")
        print(f"Resolution: {resolution[0]}x{resolution[1]}")
        print(f"Shards: {len(jobs)}")
        print()

        for job in jobs:
            node = self.cluster._nodes.get(job.node)
            gpu_info = f" ({node.gpu_type.value})" if node and node.is_gpu_node else ""
            print(f"  Shard {job.shard_index}: {job.node}{gpu_info}")
            print(f"    Items: {len(job.items)}")
            if node:
                print(f"    Resources: {node.cpu_cores} CPU, {node.memory_gb}GB RAM")
        print()


if __name__ == "__main__":
    # Test routing
    router = TaskRouter()

    test_videos = [f"/data/input/video_{i}.mp4" for i in range(10)]

    print("\n=== Scene Detection Routing (4K) ===")
    router.print_routing_plan(TaskType.CPU_SCENE_DETECTION, test_videos, (3840, 2160))

    print("\n=== GPU Encoding Routing (4K) ===")
    router.print_routing_plan(TaskType.GPU_ENCODING, test_videos[:3], (3840, 2160))

    print("\n=== Final Render Routing ===")
    node, encoder = router.get_encoding_node((3840, 2160))
    if node:
        print(f"Best encoding node: {node.name} with {encoder}")
