#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_NAMESPACE = os.environ.get("MONTAGE_NAMESPACE", "montage-ai")
DEFAULT_DOWNLOAD_DIR = Path(os.environ.get("MONTAGE_DOWNLOAD_DIR", str(Path.home() / "Downloads")))
DEFAULT_INTERVAL = int(os.environ.get("MONTAGE_DOWNLOAD_INTERVAL", "15"))
DEFAULT_MIN_SIZE_MB = float(os.environ.get("MONTAGE_MIN_OUTPUT_MB", "1"))
DEFAULT_EXTENSIONS = os.environ.get("MONTAGE_OUTPUT_EXTENSIONS", "mp4,otio,json,log").split(",")


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()


class K8sSyncDownloader:
    def __init__(
        self,
        namespace: str,
        download_dir: Path,
        interval: int,
        job_id: str | None,
        min_size_mb: float,
        extensions: list[str],
        stable_rounds: int,
        once: bool,
    ):
        self.namespace = namespace
        self.download_dir = download_dir
        self.interval = interval
        self.job_id = job_id
        self.min_size_bytes = int(min_size_mb * 1024 * 1024)
        self.extensions = {ext.strip().lstrip(".") for ext in extensions if ext.strip()}
        self.stable_rounds = max(1, stable_rounds)
        self.once = once
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.download_dir / ".montage_downloaded.json"
        self.downloaded = self._load_state()
        self.last_sizes: dict[str, int] = {}
        self.stable_counts: dict[str, int] = {}

    def _load_state(self) -> set[str]:
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                if isinstance(data, list):
                    return set(str(name) for name in data)
        except Exception:
            pass
        return set()

    def _save_state(self) -> None:
        try:
            self.state_path.write_text(json.dumps(sorted(self.downloaded), indent=2))
        except Exception:
            pass

    def get_running_pod(self) -> str | None:
        """Find a running pod that has the output volume mounted."""
        selectors = [
            "app.kubernetes.io/component=worker",
            "app.kubernetes.io/component=web-ui",
        ]
        for selector in selectors:
            cmd = [
                "kubectl",
                "get",
                "pods",
                "-n",
                self.namespace,
                "-l",
                selector,
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ]
            try:
                pod_name = _run(cmd)
                if pod_name:
                    return pod_name
            except subprocess.CalledProcessError:
                continue

        cmd = [
            "kubectl",
            "get",
            "pods",
            "-n",
            self.namespace,
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ]
        try:
            pod_name = _run(cmd)
            return pod_name if pod_name else None
        except subprocess.CalledProcessError:
            return None

    def _list_remote_files(self, pod_name: str) -> dict[str, int]:
        script = (
            'for f in /data/output/*; do '
            '[ -f "$f" ] || continue; '
            'name=$(basename "$f"); '
            'size=$(stat -c %s "$f" 2>/dev/null || echo 0); '
            'printf "%s|%s\\n" "$name" "$size"; '
            'done'
        )
        cmd = [
            "kubectl",
            "exec",
            "-n",
            self.namespace,
            pod_name,
            "--",
            "/bin/sh",
            "-lc",
            script,
        ]
        output = _run(cmd)
        results: dict[str, int] = {}
        for line in output.splitlines():
            if "|" not in line:
                continue
            name, size_str = line.split("|", 1)
            name = name.strip()
            if not name:
                continue
            try:
                size = int(size_str.strip())
            except ValueError:
                size = 0
            results[name] = size
        return results

    def _should_consider(self, filename: str, size: int) -> bool:
        if size < self.min_size_bytes:
            return False
        if ".tmp" in filename or filename.endswith(".part"):
            return False
        if self.job_id and self.job_id not in filename:
            return False
        if self.extensions:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext not in self.extensions:
                return False
        return True

    def _local_up_to_date(self, filename: str, size: int) -> bool:
        local_path = self.download_dir / filename
        if not local_path.exists():
            return False
        try:
            return local_path.stat().st_size >= size
        except OSError:
            return False

    def download_file(self, pod_name: str, filename: str) -> bool:
        local_path = self.download_dir / filename
        temp_path = local_path.with_suffix(local_path.suffix + ".part")
        if temp_path.exists():
            temp_path.unlink()
        print(f"📥 Downloading {filename} to {local_path}...")
        cmd = [
            "kubectl",
            "cp",
            f"{self.namespace}/{pod_name}:/data/output/{filename}",
            str(temp_path),
        ]
        try:
            subprocess.run(cmd, check=True)
            temp_path.replace(local_path)
            print(f"✅ Success: {local_path}")
            try:
                subprocess.run(["notify-send", "Montage AI", f"New video downloaded: {filename}"], stderr=subprocess.DEVNULL)
            except Exception:
                pass
            return True
        except subprocess.CalledProcessError as exc:
            print(f"❌ Failed to download {filename}: {exc}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def sync(self) -> None:
        pod_name = self.get_running_pod()
        if not pod_name:
            print("⚠️ No running pod found (waiting for Web UI or Job to be active)...")
            return

        try:
            remote_files = self._list_remote_files(pod_name)
            for filename, size in remote_files.items():
                if not self._should_consider(filename, size):
                    continue

                last_size = self.last_sizes.get(filename)
                if last_size is not None and size == last_size:
                    self.stable_counts[filename] = self.stable_counts.get(filename, 0) + 1
                else:
                    self.stable_counts[filename] = 0
                self.last_sizes[filename] = size

                if self._local_up_to_date(filename, size):
                    self.downloaded.add(filename)
                    continue
                if filename in self.downloaded:
                    continue
                if self.stable_counts.get(filename, 0) < self.stable_rounds - 1:
                    continue

                print(f"✨ New file detected on cluster: {filename} ({size / (1024 * 1024):.1f} MB)")
                if self.download_file(pod_name, filename):
                    self.downloaded.add(filename)
                    self._save_state()
        except Exception as exc:
            print(f"Error during sync: {exc}")

    def run(self) -> None:
        print("🚀 K8s Auto-Downloader (Sync Mode) started.")
        print(f"👀 Watching Cluster Output -> {self.download_dir}")
        if self.job_id:
            print(f"🔎 Filtering for job id: {self.job_id}")
        print("Press Ctrl+C to stop.")

        while True:
            self.sync()
            if self.once:
                break
            time.sleep(self.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-download montage outputs from a K8s cluster.")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Kubernetes namespace.")
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR), help="Local directory for downloads.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Polling interval in seconds.")
    parser.add_argument("--job-id", help="Filter by job id substring.")
    parser.add_argument("--min-size-mb", type=float, default=DEFAULT_MIN_SIZE_MB, help="Minimum file size to download.")
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS), help="Comma-separated extensions to download.")
    parser.add_argument("--stable-rounds", type=int, default=2, help="Consecutive size checks before download.")
    parser.add_argument("--once", action="store_true", help="Run a single sync cycle and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    downloader = K8sSyncDownloader(
        namespace=args.namespace,
        download_dir=Path(args.download_dir).expanduser(),
        interval=args.interval,
        job_id=args.job_id,
        min_size_mb=args.min_size_mb,
        extensions=[ext.strip() for ext in args.extensions.split(",") if ext.strip()],
        stable_rounds=args.stable_rounds,
        once=args.once,
    )
    try:
        downloader.run()
    except KeyboardInterrupt:
        print("\nStopping downloader...")
        sys.exit(0)
