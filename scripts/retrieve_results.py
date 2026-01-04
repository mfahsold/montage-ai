#!/usr/bin/env python3
"""
Retrieve results from the Kubernetes cluster automatically.
Finds the latest project and video file in the output PVC and copies them locally.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

NAMESPACE = "montage-ai"
LOCAL_OUTPUT_DIR = Path("data/output")
REMOTE_OUTPUT_DIR = "/data/output"
POD_SELECTOR = "app=data-helper" # Or we can find a specific pod

def run_kubectl(args, check=True, capture_output=True):
    """Run a kubectl command."""
    cmd = ["kubectl", "-n", NAMESPACE] + args
    result = subprocess.run(cmd, check=check, capture_output=capture_output, text=True)
    return result.stdout.strip()

def find_helper_pod():
    """Find a pod that mounts the output PVC, or create one."""
    pod_name = "data-helper"
    
    # Check if exists
    try:
        run_kubectl(["get", "pod", pod_name], check=True, capture_output=True)
        # Check if ready
        run_kubectl(["wait", "--for=condition=Ready", "pod/" + pod_name, "--timeout=5s"], check=True)
        return pod_name
    except subprocess.CalledProcessError:
        pass

    print(f"🚀 Creating {pod_name} pod...")
    # Create the pod
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": NAMESPACE},
        "spec": {
            "containers": [{
                "name": "helper",
                "image": "busybox",
                "command": ["sleep", "infinity"],
                "volumeMounts": [{"name": "out", "mountPath": REMOTE_OUTPUT_DIR}]
            }],
            "volumes": [{"name": "out", "persistentVolumeClaim": {"claimName": "montage-ai-output-nfs"}}],
            "restartPolicy": "Never"
        }
    }
    
    # Apply manifest via stdin
    subprocess.run(["kubectl", "apply", "-f", "-"], input=json.dumps(manifest), text=True, check=True, capture_output=True)
    
    print("⏳ Waiting for pod to be ready...")
    run_kubectl(["wait", "--for=condition=Ready", "pod/" + pod_name, "--timeout=60s"])
    return pod_name

def list_remote_files(pod_name):
    """List files in the remote output directory."""
    # Use ls -1tF to get filenames sorted by time, with type indicators
    try:
        output = run_kubectl(["exec", pod_name, "--", "ls", "-1tF", REMOTE_OUTPUT_DIR])
    except subprocess.CalledProcessError:
        print("⚠️  Failed to list remote files.")
        return []

    files = []
    for line in output.splitlines():
        name = line.strip()
        if not name:
            continue
            
        is_dir = name.endswith('/')
        clean_name = name.rstrip('/')
        
        files.append({
            "name": clean_name,
            "is_dir": is_dir
        })
    return files

def copy_file(pod_name, remote_path, local_path):
    """Copy a file from the pod to local."""
    print(f"⬇️  Copying {remote_path} -> {local_path}...")
    
    # Ensure local dir exists
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Method 1: kubectl cp (try first)
    try:
        run_kubectl(["cp", f"{pod_name}:{remote_path}", str(local_path)])
        return True
    except subprocess.CalledProcessError:
        print("⚠️  kubectl cp failed, trying cat redirection...")
    
    # Method 2: Tar pipe (more robust for binary)
    # kubectl exec ... -- tar cf - /remote/path | tar xf - -C /local/dir
    try:
        print("   Trying tar pipe...")
        # We need to strip the leading slash from remote path for tar to work nicely or handle it
        # tar -C / -cf - data/output/file
        
        remote_dir = os.path.dirname(remote_path)
        remote_file = os.path.basename(remote_path)
        
        cmd_remote = ["kubectl", "exec", "-n", NAMESPACE, pod_name, "--", "tar", "-C", remote_dir, "-cf", "-", remote_file]
        cmd_local = ["tar", "xf", "-", "-C", str(local_path.parent)]
        
        p1 = subprocess.Popen(cmd_remote, stdout=subprocess.PIPE)
        p2 = subprocess.run(cmd_local, stdin=p1.stdout, check=True)
        p1.wait()
        
        if p1.returncode == 0 and p2.returncode == 0:
            return True
    except Exception as e:
        print(f"⚠️  Tar pipe failed: {e}")

    # Method 3: Base64 encoding (robust for binary)
    try:
        print("   Trying base64 transfer...")
        # kubectl exec ... -- base64 /path | base64 -d > local
        
        cmd_remote = ["kubectl", "exec", "-n", NAMESPACE, pod_name, "--", "base64", remote_path]
        cmd_decode = ["base64", "-d"]
        
        # Open local file for writing
        with open(local_path, "wb") as f:
            p1 = subprocess.Popen(cmd_remote, stdout=subprocess.PIPE)
            p2 = subprocess.run(cmd_decode, stdin=p1.stdout, stdout=f, check=True)
            p1.wait()
            
        if p1.returncode == 0 and p2.returncode == 0:
            return True
    except Exception as e:
        print(f"⚠️  Base64 transfer failed: {e}")

    # Method 4: cat redirection (last resort)
    try:
        print("   Trying cat redirection...")
        with open(local_path, "wb") as f:
            subprocess.run(
                ["kubectl", "exec", "-n", NAMESPACE, pod_name, "--", "cat", remote_path],
                stdout=f,
                check=True
            )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Copy failed: {e}")
        return False

def copy_directory(pod_name, remote_path, local_path):
    """Copy a directory from the pod to local."""
    print(f"⬇️  Copying directory {remote_path} -> {local_path}...")
    # kubectl cp -r works for directories usually
    try:
        run_kubectl(["cp", f"{pod_name}:{remote_path}", str(local_path)])
        return True
    except subprocess.CalledProcessError:
        print("⚠️  kubectl cp -r failed, trying manual recursive copy...")
        # This would be complex to implement fully with cat, so we might just fail or try tar
        return False

def main():
    print("🔍 Finding helper pod...")
    pod_name = find_helper_pod()
    print(f"✅ Using pod: {pod_name}")
    
    print("📂 Listing remote files...")
    files = list_remote_files(pod_name)
    
    # Find latest project folder and mp4
    latest_project = None
    latest_video = None
    
    for f in files:
        if f["is_dir"] and f["name"].endswith("_PROJECT") and not latest_project:
            latest_project = f
        if not f["is_dir"] and f["name"].endswith(".mp4") and not latest_video:
            latest_video = f
            
    if not latest_project and not latest_video:
        print("❌ No results found.")
        sys.exit(1)
        
    if latest_project:
        print(f"📦 Found project: {latest_project['name']}")
        local_proj = LOCAL_OUTPUT_DIR / latest_project['name']
        if not local_proj.exists():
            if copy_directory(pod_name, f"{REMOTE_OUTPUT_DIR}/{latest_project['name']}", local_proj):
                if local_proj.exists() and any(local_proj.iterdir()):
                    print("✅ Project folder verification successful!")
                else:
                    print("❌ Project folder verification failed (empty or missing)!")
        else:
            print("   (Already exists locally)")
    else:
        print("⚠️  No project folder found for the latest run.")
            
    if latest_video:
        print(f"🎬 Found video: {latest_video['name']}")
        local_vid = LOCAL_OUTPUT_DIR / latest_video['name']
        
        if not local_vid.exists():
            copy_file(pod_name, f"{REMOTE_OUTPUT_DIR}/{latest_video['name']}", local_vid)
            if local_vid.exists():
                print("✅ Verification successful!")
            else:
                print("❌ Verification failed!")
        else:
            print("   (Already exists locally)")

if __name__ == "__main__":
    main()
