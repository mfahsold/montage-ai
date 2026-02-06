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
            elif key == "hostnames":
                subsection = "cluster_hostnames"
                continue
        elif section == "cluster" and subsection == "cluster_hostnames" and indent == 4 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = clean(value)
            if key == "montage":
                montage_hostname = value
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
