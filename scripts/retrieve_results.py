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
    """Find a pod that mounts the output PVC."""
    # First try data-helper
    try:
        pod = run_kubectl(["get", "pod", "data-helper", "-o", "jsonpath={.metadata.name}"], check=False)
        if pod:
            return pod
    except:
        pass
    
    # Fallback to any pod with the mount
    # This is a simplification; in a real scenario we'd query pods with the PVC mounted
    return "data-helper"

def list_remote_files(pod_name):
    """List files in the remote output directory."""
    # Use ls -lt to sort by time (newest first)
    # Alpine/Busybox supports --full-time but not --time-style
    try:
        output = run_kubectl(["exec", pod_name, "--", "ls", "-lt", "--full-time", REMOTE_OUTPUT_DIR])
    except subprocess.CalledProcessError:
        # Fallback to simple ls -lt if --full-time fails
        output = run_kubectl(["exec", pod_name, "--", "ls", "-lt", REMOTE_OUTPUT_DIR])

    files = []
    for line in output.splitlines()[1:]: # Skip total line
        parts = line.split()
        if len(parts) < 8:
            continue
        
        # Busybox ls -lt --full-time output:
        # -rw-r--r--    1 root     root      31069405 2026-01-03 17:31:41.000000000 +0000 filename
        # Standard ls -lt output:
        # -rw-r--r--    1 root     root      31069405 Jan  3 17:31 filename
        
        is_dir = parts[0].startswith('d')
        try:
            size = int(parts[4])
        except ValueError:
            continue # Skip if size is not an int
            
        # Name is usually the last part, but spaces in filenames make it tricky.
        # We need to find where the date ends.
        # Heuristic: find the first part after size that looks like a date/time or month
        
        # Simple approach: assume name starts after the time column
        # In full-time: 5=date, 6=time, 7=timezone, 8+=name
        # In standard: 5=Month, 6=Day, 7=Time/Year, 8+=name
        
        name_start_idx = 8
        if "full-time" not in output and len(parts) >= 8:
             # Standard ls output usually has 8 columns before name if we count month/day/time separately
             # -rw-r--r-- 1 user group size Mon Day Time Name
             pass
             
        name = " ".join(parts[name_start_idx:])
        
        # Fix for standard ls where it might be fewer columns if group is missing or whatever
        # Let's just take the last part if it doesn't have spaces, or join from index 8
        # Actually, let's just look for the known extensions or _PROJECT suffix
        
        # Better parsing:
        # The filename is everything after the date/time.
        # If we use --full-time, we have a strict format.
        
        files.append({
            "name": name,
            "size": size,
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
            copy_directory(pod_name, f"{REMOTE_OUTPUT_DIR}/{latest_project['name']}", local_proj)
        else:
            print("   (Already exists locally)")
            
    if latest_video:
        print(f"🎬 Found video: {latest_video['name']} ({latest_video['size'] / 1024 / 1024:.2f} MB)")
        local_vid = LOCAL_OUTPUT_DIR / latest_video['name']
        
        should_copy = True
        if local_vid.exists():
            local_size = local_vid.stat().st_size
            if local_size == latest_video['size']:
                print("   (Already exists locally and size matches)")
                should_copy = False
            else:
                print(f"   (Local size mismatch: {local_size} vs {latest_video['size']})")
        
        if should_copy:
            copy_file(pod_name, f"{REMOTE_OUTPUT_DIR}/{latest_video['name']}", local_vid)
            
            # Verify
            if local_vid.exists() and local_vid.stat().st_size == latest_video['size']:
                print("✅ Verification successful!")
            else:
                print("❌ Verification failed!")

if __name__ == "__main__":
    main()
