#!/usr/bin/env python3
"""
Download job artifacts from Montage AI (CLI utility).

Features:
- Downloads video, timeline exports (OTIO/EDL), proxies, and logs
- Supports both local and remote (API) sources
- Auto-organizes into project folder
- Progress indication

Usage:
    # Download from API (cluster/remote)
    python scripts/download_job.py --job-id 20260112_114010 --api http://<MONTAGE_API_HOST>

    # Download from local output dir
    python scripts/download_job.py --job-id 20260112_114010 --local /data/output

    # Download as ZIP
    python scripts/download_job.py --job-id 20260112_114010 --api http://<MONTAGE_API_HOST> --zip

    # Specify output directory
    python scripts/download_job.py --job-id 20260112_114010 --output ./my_project
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ANSI colors for CLI output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def log(msg: str, color: str = Colors.RESET):
    """Print colored log message."""
    print(f"{color}{msg}{Colors.RESET}")


def collect_local_artifacts(job_id: str, output_dir: Path) -> Dict[str, List[Path]]:
    """Collect artifacts from local filesystem."""
    artifacts = {
        "video": [],
        "timeline": [],
        "proxies": [],
        "logs": [],
    }

    patterns = {
        "video": [f"*{job_id}*.mp4"],
        "timeline": [f"*{job_id}*.otio", f"*{job_id}*.edl", f"*{job_id}*.xml"],
        "logs": [f"render_{job_id}.log", f"*{job_id}*_RECIPE_CARD.md", f"decisions_{job_id}.json"],
    }

    for category, globs in patterns.items():
        for pattern in globs:
            for path in output_dir.glob(pattern):
                if path.is_file():
                    artifacts[category].append(path)

    # Check for proxies
    proxy_dir = output_dir / "proxies"
    if proxy_dir.exists():
        for proxy in proxy_dir.glob("*.mp4"):
            artifacts["proxies"].append(proxy)
        for proxy in proxy_dir.glob("*.mov"):
            artifacts["proxies"].append(proxy)

    return artifacts


def download_local(job_id: str, source_dir: Path, dest_dir: Path, as_zip: bool = False) -> bool:
    """Download/copy artifacts from local filesystem."""
    log(f"\n{Colors.BOLD}Collecting artifacts for job {job_id}...{Colors.RESET}")

    artifacts = collect_local_artifacts(job_id, source_dir)

    total_files = sum(len(files) for files in artifacts.values())
    if total_files == 0:
        log(f"No artifacts found for job {job_id}", Colors.YELLOW)
        return False

    # Create destination
    dest_dir.mkdir(parents=True, exist_ok=True)

    if as_zip:
        # Create ZIP
        zip_path = dest_dir / f"montage_{job_id}.zip"
        log(f"Creating ZIP: {zip_path}", Colors.CYAN)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for category, files in artifacts.items():
                folder = "" if category == "video" else f"{category}/"
                for f in files:
                    zf.write(f, f"{folder}{f.name}")
                    log(f"  + {folder}{f.name}", Colors.GREEN)

        log(f"\n{Colors.BOLD}ZIP created: {zip_path}{Colors.RESET}", Colors.GREEN)
    else:
        # Copy files
        copied = 0
        for category, files in artifacts.items():
            if not files:
                continue

            # Create category subfolder (except video in root)
            if category == "video":
                cat_dir = dest_dir
            else:
                cat_dir = dest_dir / category
                cat_dir.mkdir(exist_ok=True)

            for f in files:
                dest = cat_dir / f.name
                shutil.copy2(f, dest)
                log(f"  {Colors.GREEN}[{category}]{Colors.RESET} {f.name}")
                copied += 1

        log(f"\n{Colors.BOLD}Downloaded {copied} files to {dest_dir}{Colors.RESET}", Colors.GREEN)

    return True


def download_from_api(job_id: str, api_base: str, dest_dir: Path, as_zip: bool = False) -> bool:
    """Download artifacts from API endpoint."""
    if not REQUESTS_AVAILABLE:
        log("Error: requests library required for API downloads", Colors.RED)
        log("Install with: pip install requests", Colors.YELLOW)
        return False

    log(f"\n{Colors.BOLD}Fetching artifacts for job {job_id} from {api_base}...{Colors.RESET}")

    # Ensure api_base ends without slash
    api_base = api_base.rstrip('/')

    # Get job status first
    try:
        resp = requests.get(f"{api_base}/api/jobs/{job_id}", timeout=10)
        if resp.status_code == 404:
            log(f"Job {job_id} not found", Colors.RED)
            return False
        resp.raise_for_status()
        job = resp.json()

        if job.get("status") not in ("completed", "success", "finished"):
            log(f"Job not completed (status: {job.get('status')})", Colors.YELLOW)
            return False
    except requests.RequestException as e:
        log(f"API error: {e}", Colors.RED)
        return False

    # Create destination
    dest_dir.mkdir(parents=True, exist_ok=True)

    if as_zip:
        # Download ZIP directly
        zip_url = f"{api_base}/api/jobs/{job_id}/download/zip"
        zip_path = dest_dir / f"montage_{job_id}.zip"

        log(f"Downloading ZIP from {zip_url}...", Colors.CYAN)

        try:
            with requests.get(zip_url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                downloaded = 0

                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded * 100 / total)
                            print(f"\r  Downloading... {pct}%", end='', flush=True)
                print()

            log(f"\n{Colors.BOLD}ZIP downloaded: {zip_path}{Colors.RESET}", Colors.GREEN)
            return True

        except requests.RequestException as e:
            log(f"ZIP download failed ({e}); falling back to file-by-file download.", Colors.YELLOW)

    # Get artifacts list
    try:
        resp = requests.get(f"{api_base}/api/jobs/{job_id}/artifacts", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log(f"Failed to get artifacts list: {e}", Colors.RED)
        return False

    # Download individual files
    downloaded = 0
    for category, files in data.get("artifacts", {}).items():
        if not files:
            continue

        # Create category subfolder (except video in root)
        if category == "video":
            cat_dir = dest_dir
        else:
            cat_dir = dest_dir / category
            cat_dir.mkdir(exist_ok=True)

        for file_info in files:
            url = f"{api_base}{file_info['download_url']}"
            dest_path = cat_dir / file_info['name']

            try:
                with requests.get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                log(f"  {Colors.GREEN}[{category}]{Colors.RESET} {file_info['name']} ({file_info['size_mb']} MB)")
                downloaded += 1

            except requests.RequestException as e:
                log(f"  {Colors.RED}[FAILED]{Colors.RESET} {file_info['name']}: {e}")

    log(f"\n{Colors.BOLD}Downloaded {downloaded} files to {dest_dir}{Colors.RESET}", Colors.GREEN)
    return downloaded > 0


def main():
    parser = argparse.ArgumentParser(
        description="Download Montage AI job artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from remote API
  python scripts/download_job.py --job-id 20260112_114010 --api http://<MONTAGE_API_HOST>

  # Download from local output dir
  python scripts/download_job.py --job-id 20260112_114010 --local /data/output

    # Download as ZIP
    python scripts/download_job.py --job-id 20260112_114010 --api http://<MONTAGE_API_HOST> --zip
        """
    )

    parser.add_argument("--job-id", "-j", required=True, help="Job ID to download")
    parser.add_argument("--output", "-o", default="./downloads", help="Output directory (default: ./downloads)")
    parser.add_argument("--zip", "-z", action="store_true", help="Download as single ZIP file")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--api", "-a", help="API base URL (e.g., http://<MONTAGE_API_HOST>)")
    source.add_argument("--local", "-l", help="Local output directory path")

    args = parser.parse_args()

    job_id = args.job_id
    dest_dir = Path(args.output) / job_id
    as_zip = args.zip

    if args.api:
        success = download_from_api(job_id, args.api, dest_dir, as_zip)
    else:
        source_dir = Path(args.local)
        if not source_dir.exists():
            log(f"Source directory not found: {source_dir}", Colors.RED)
            sys.exit(1)
        success = download_local(job_id, source_dir, dest_dir, as_zip)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
