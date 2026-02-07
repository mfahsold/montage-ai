#!/usr/bin/env bash
# Minimal helper for reading deploy/k3s/config-global.yaml.
# Uses python3 if available; falls back to a basic parser for limited keys.

config_global_default_path() {
  if [ -n "${MONTAGE_CONFIG_GLOBAL:-}" ]; then
    echo "$MONTAGE_CONFIG_GLOBAL"
    return 0
  fi

  local lib_dir
  lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local repo_root
  repo_root="$(cd "$lib_dir/../../.." && pwd)"
  echo "$repo_root/deploy/k3s/config-global.yaml"
}

config_global_export() {
  local config_path="${1:-$(config_global_default_path)}"

  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi

  python3 - "$config_path" <<'PY'
import shlex
import sys

path = sys.argv[1]
text = ""
try:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
except Exception:
    print("CONFIG_GLOBAL_ERROR='config-global.yaml not readable'")
    sys.exit(0)

def clean(value):
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'")

registry_url = ""
registry_host = ""
registry_port = ""
cluster_namespace = ""
cluster_domain = ""
montage_hostname = ""
cluster_allow_mixed_arch = ""
cluster_arch_selector_key = ""
cluster_node_selector = ""
cluster_parallelism = ""
cluster_render_tier = ""
scene_detect_tier = ""
worker_min_replicas = ""
worker_max_replicas = ""
worker_queue_scale_threshold = ""
worker_queue_list_name = ""
max_scene_workers = ""
max_parallel_jobs = ""
worker_cpu_request = ""
worker_cpu_limit = ""
worker_memory_request = ""
worker_memory_limit = ""
ffmpeg_threads = ""
image_name = ""
image_tag = ""
image_full = ""
storage_class_default = ""
storage_class_nfs = ""
nfs_server = ""
nfs_path = ""
pvc_input = ""
pvc_output = ""
pvc_music = ""
pvc_assets = ""

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

if yaml is not None:
    try:
        parsed = yaml.safe_load(text) or {}
        registry = parsed.get("registry", {}) if isinstance(parsed, dict) else {}
        cluster = parsed.get("cluster", {}) if isinstance(parsed, dict) else {}
        storage = parsed.get("storage", {}) if isinstance(parsed, dict) else {}
        images = parsed.get("images", {}) if isinstance(parsed, dict) else {}
        montage_ai = images.get("montage_ai", {}) if isinstance(images, dict) else {}

        registry_url = clean(registry.get("url", ""))
        registry_host = clean(registry.get("host", ""))
        registry_port = clean(registry.get("port", ""))
        cluster_namespace = clean(cluster.get("namespace", ""))
        cluster_domain = clean(cluster.get("clusterDomain", ""))
        cluster_allow_mixed_arch = clean(cluster.get("allowMixedArch", ""))
        cluster_arch_selector_key = clean(cluster.get("archSelectorKey", ""))
        cluster_node_selector = clean(cluster.get("nodeSelector", ""))
        cluster_parallelism = clean(cluster.get("parallelism", ""))
        cluster_render_tier = clean(cluster.get("renderTier", ""))
        scene_detect_tier = clean(cluster.get("sceneDetectTier", ""))
        worker_min_replicas = clean(cluster.get("workerMinReplicas", ""))
        worker_max_replicas = clean(cluster.get("workerMaxReplicas", ""))
        worker_queue_scale_threshold = clean(cluster.get("workerQueueScaleThreshold", ""))
        worker_queue_list_name = clean(cluster.get("workerQueueListName", ""))
        max_scene_workers = clean(cluster.get("maxSceneWorkers", ""))
        max_parallel_jobs = clean(cluster.get("maxParallelJobs", ""))
        worker_resources = cluster.get("workerResources", {}) if isinstance(cluster, dict) else {}
        worker_cpu_request = clean(worker_resources.get("cpuRequest", ""))
        worker_cpu_limit = clean(worker_resources.get("cpuLimit", ""))
        worker_memory_request = clean(worker_resources.get("memoryRequest", ""))
        worker_memory_limit = clean(worker_resources.get("memoryLimit", ""))
        ffmpeg_threads = clean(cluster.get("ffmpegThreads", ""))
        cluster_hostnames = cluster.get("hostnames", {}) if isinstance(cluster, dict) else {}
        montage_hostname = clean(cluster_hostnames.get("montage", ""))

        image_name = clean(montage_ai.get("name", ""))
        image_tag = clean(montage_ai.get("tag", ""))

        storage_classes = storage.get("classes", {}) if isinstance(storage, dict) else {}
        storage_class_default = clean(storage_classes.get("default", ""))
        storage_class_nfs = clean(storage_classes.get("nfs", ""))

        storage_nfs = storage.get("nfs", {}) if isinstance(storage, dict) else {}
        nfs_server = clean(storage_nfs.get("server", ""))
        nfs_path = clean(storage_nfs.get("path", ""))
        storage_pvc = storage.get("pvc", {}) if isinstance(storage, dict) else {}
        pvc_input = clean(storage_pvc.get("input", ""))
        pvc_output = clean(storage_pvc.get("output", ""))
        pvc_music = clean(storage_pvc.get("music", ""))
        pvc_assets = clean(storage_pvc.get("assets", ""))
    except Exception:
        pass

if registry_url and image_name and image_tag:
    image_full = f"{registry_url}/{image_name}:{image_tag}"

if not any([
    registry_url,
    registry_host,
    registry_port,
    cluster_namespace,
    cluster_domain,
    montage_hostname,
    cluster_parallelism,
    cluster_render_tier,
    scene_detect_tier,
    worker_min_replicas,
    worker_max_replicas,
    worker_queue_scale_threshold,
    worker_queue_list_name,
    max_scene_workers,
    max_parallel_jobs,
    worker_cpu_request,
    worker_cpu_limit,
    worker_memory_request,
    worker_memory_limit,
    ffmpeg_threads,
    image_name,
    image_tag,
    image_full,
    storage_class_default,
    storage_class_nfs,
    nfs_server,
    nfs_path,
]):
    section = None
    subsection = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            section = None
            subsection = None
            if stripped.startswith("registry:"):
                section = "registry"
            elif stripped.startswith("cluster:"):
                section = "cluster"
            elif stripped.startswith("storage:"):
                section = "storage"
            elif stripped.startswith("images:"):
                section = "images"
            continue
        if section == "registry" and indent == 2 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = clean(value)
            if key == "url":
                registry_url = value
            elif key == "host":
                registry_host = value
            elif key == "port":
                registry_port = value
        elif section == "cluster" and indent == 2 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = clean(value)
            if key == "namespace":
                cluster_namespace = value
            elif key == "clusterDomain":
                cluster_domain = value
            elif key == "allowMixedArch":
                cluster_allow_mixed_arch = value
            elif key == "archSelectorKey":
                cluster_arch_selector_key = value
            elif key == "nodeSelector":
                cluster_node_selector = value
            elif key == "parallelism":
                cluster_parallelism = value
            elif key == "renderTier":
                cluster_render_tier = value
            elif key == "sceneDetectTier":
                scene_detect_tier = value
            elif key == "workerMinReplicas":
                worker_min_replicas = value
            elif key == "workerMaxReplicas":
                worker_max_replicas = value
            elif key == "workerQueueScaleThreshold":
                worker_queue_scale_threshold = value
            elif key == "workerQueueListName":
                worker_queue_list_name = value
            elif key == "maxSceneWorkers":
                max_scene_workers = value
            elif key == "maxParallelJobs":
                max_parallel_jobs = value
            elif key == "ffmpegThreads":
                ffmpeg_threads = value
            elif key == "hostnames":
                subsection = "cluster_hostnames"
                continue
            elif key == "workerResources":
                subsection = "cluster_worker_resources"
                continue
        elif section == "cluster" and subsection == "cluster_hostnames" and indent == 4 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = clean(value)
            if key == "montage":
                montage_hostname = value
        elif section == "cluster" and subsection == "cluster_worker_resources" and indent == 4 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = clean(value)
            if key == "cpuRequest":
                worker_cpu_request = value
            elif key == "cpuLimit":
                worker_cpu_limit = value
            elif key == "memoryRequest":
                worker_memory_request = value
            elif key == "memoryLimit":
                worker_memory_limit = value
        elif section == "storage":
            if indent == 2 and stripped.startswith("classes:"):
                subsection = "storage_classes"
                continue
            if indent == 2 and stripped.startswith("nfs:"):
                subsection = "storage_nfs"
                continue
            if indent == 2 and stripped.startswith("pvc:"):
                subsection = "storage_pvc"
                continue
            if subsection == "storage_classes" and indent == 4 and ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = clean(value)
                if key == "default":
                    storage_class_default = value
                elif key == "nfs":
                    storage_class_nfs = value
            if subsection == "storage_nfs" and indent == 4 and ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = clean(value)
                if key == "server":
                    nfs_server = value
                elif key == "path":
                    nfs_path = value
            if subsection == "storage_pvc" and indent == 4 and ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = clean(value)
                if key == "input":
                    pvc_input = value
                elif key == "output":
                    pvc_output = value
                elif key == "music":
                    pvc_music = value
                elif key == "assets":
                    pvc_assets = value
        elif section == "images":
            if indent == 2 and stripped.startswith("montage_ai:"):
                subsection = "montage_ai"
                continue
            if subsection == "montage_ai" and indent == 4 and ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = clean(value)
                if key == "name":
                    image_name = value
                elif key == "tag":
                    image_tag = value


def emit(key, value):
    print(f"{key}={shlex.quote(clean(value))}")

emit("REGISTRY_URL", registry_url)
emit("REGISTRY_HOST", registry_host)
emit("REGISTRY_PORT", registry_port)
emit("CLUSTER_NAMESPACE", cluster_namespace)
emit("K3S_CLUSTER_DOMAIN", cluster_domain)
emit("MONTAGE_HOSTNAME", montage_hostname)
emit("CLUSTER_ALLOW_MIXED_ARCH", cluster_allow_mixed_arch)
emit("CLUSTER_ARCH_SELECTOR_KEY", cluster_arch_selector_key)
emit("CLUSTER_NODE_SELECTOR", cluster_node_selector)
emit("CLUSTER_PARALLELISM", cluster_parallelism)
emit("CLUSTER_RENDER_TIER", cluster_render_tier)
emit("SCENE_DETECT_TIER", scene_detect_tier)
emit("WORKER_MIN_REPLICAS", worker_min_replicas)
emit("WORKER_MAX_REPLICAS", worker_max_replicas)
emit("WORKER_QUEUE_SCALE_THRESHOLD", worker_queue_scale_threshold)
emit("WORKER_QUEUE_LIST_NAME", worker_queue_list_name)
emit("MAX_SCENE_WORKERS", max_scene_workers)
emit("MAX_PARALLEL_JOBS", max_parallel_jobs)
emit("WORKER_CPU_REQUEST", worker_cpu_request)
emit("WORKER_CPU_LIMIT", worker_cpu_limit)
emit("WORKER_MEMORY_REQUEST", worker_memory_request)
emit("WORKER_MEMORY_LIMIT", worker_memory_limit)
emit("FFMPEG_THREADS", ffmpeg_threads)
emit("IMAGE_NAME", image_name)
emit("IMAGE_TAG", image_tag)
emit("IMAGE_FULL", image_full)
emit("STORAGE_CLASS_DEFAULT", storage_class_default)
emit("STORAGE_CLASS_NFS", storage_class_nfs)
emit("NFS_SERVER", nfs_server)
emit("NFS_PATH", nfs_path)
emit("PVC_INPUT_NAME", pvc_input)
emit("PVC_OUTPUT_NAME", pvc_output)
emit("PVC_MUSIC_NAME", pvc_music)
emit("PVC_ASSETS_NAME", pvc_assets)
PY
}
