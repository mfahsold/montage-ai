#!/usr/bin/env python3
import subprocess
import time
import os
import sys
from pathlib import Path

# Configuration
NAMESPACE = "montage-ai"
DOWNLOAD_DIR = Path.home() / "Downloads"
CHECK_INTERVAL = 15  # Seconds

class K8sSyncDownloader:
    def __init__(self):
        self.download_dir = DOWNLOAD_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def get_running_pod(self):
        """Find a pod that has the output volume mounted."""
        # Try web pod first
        cmd = ["kubectl", "get", "pods", "-n", NAMESPACE, "-l", "app.kubernetes.io/component=web-ui", "--field-selector=status.phase=Running", "-o", "jsonpath={.items[0].metadata.name}"]
        try:
            pod_name = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            if pod_name:
                return pod_name
        except subprocess.CalledProcessError:
            pass
        
        # Try any running pod
        cmd = ["kubectl", "get", "pods", "-n", NAMESPACE, "--field-selector=status.phase=Running", "-o", "jsonpath={.items[0].metadata.name}"]
        try:
            pods = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip().split()
            if pods:
                return pods[0]
        except subprocess.CalledProcessError:
            return None

    def download_file(self, pod_name, remote_path):
        filename = os.path.basename(remote_path)
        local_path = self.download_dir / filename
        
        print(f"📥 Downloading {filename} to {local_path}...")
        
        cmd = ["kubectl", "cp", f"{NAMESPACE}/{pod_name}:{remote_path}", str(local_path)]
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Success: {local_path}")
            # Try desktop notification
            try:
                subprocess.run(["notify-send", "Montage AI", f"New video downloaded: {filename}"], stderr=subprocess.DEVNULL)
            except:
                pass
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {remote_path}: {e}")
            return False

    def sync(self):
        pod_name = self.get_running_pod()
        if not pod_name:
            print("⚠️ No running pod found (waiting for Web UI or Job to be active)...")
            return

        try:
            # List files in /data/output
            output = subprocess.check_output(["kubectl", "exec", "-n", NAMESPACE, pod_name, "--", "ls", "-1", "/data/output/"], stderr=subprocess.DEVNULL).decode()
            remote_files = output.splitlines()
            
            for filename in remote_files:
                # We only care about final montage videos and monitoring reports
                if not (filename.endswith(".mp4") or filename.endswith(".json") or filename.endswith(".log")):
                    continue
                
                # Specifically target montage files and keep things tidy
                if "montage" in filename or "render.log" in filename:
                    local_path = self.download_dir / filename
                    if not local_path.exists():
                        print(f"✨ New file detected on cluster: {filename}")
                        self.download_file(pod_name, f"/data/output/{filename}")
        except Exception as e:
            print(f"Error during sync: {e}")

    def run(self):
        print(f"🚀 K8s Auto-Downloader (Sync Mode) started.")
        print(f"👀 Watching Cluster Output -> {self.download_dir}")
        print("Press Ctrl+C to stop.")
        
        while True:
            self.sync()
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    downloader = K8sSyncDownloader()
    try:
        downloader.run()
    except KeyboardInterrupt:
        print("\nStopping downloader...")
        sys.exit(0)
