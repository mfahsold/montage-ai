#!/usr/bin/env python3
"""
Prepare public-domain trailer assets (NASA footage + audio).

Downloads a small set of NASA video clips and a short NASA audio segment.
All NASA media is public domain (U.S. Government work).
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

import requests


NASA_SEARCH_URL = "https://images-api.nasa.gov/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (MontageAI/1.0)"}

DEFAULT_VIDEO_QUERIES = [
    "earth",
    "rocket launch",
    "iss",
    "mars rover",
]
DEFAULT_AUDIO_QUERY = "NASA podcast"


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")


def _fetch_json(url: str, params: Optional[dict] = None) -> dict:
    resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _head_size_mb(url: str) -> Optional[float]:
    try:
        resp = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=15)
        if resp.status_code >= 400:
            return None
        size = resp.headers.get("Content-Length")
        if size is None:
            return None
        return int(size) / (1024 * 1024)
    except Exception:
        return None


def _pick_best_asset(assets: Iterable[str], max_mb: float, prefer: List[str]) -> Optional[str]:
    mp4s = [u for u in assets if u.endswith(".mp4")]
    if not mp4s:
        return None

    ranked = []
    for tag in prefer:
        ranked.extend([u for u in mp4s if tag in u])
    ranked.extend([u for u in mp4s if u not in ranked])

    for url in ranked:
        size_mb = _head_size_mb(url)
        if size_mb is None or size_mb <= max_mb:
            return url
    return None


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def _nasa_items(query: str, media_type: str, limit: int) -> List[dict]:
    data = _fetch_json(NASA_SEARCH_URL, params={"q": query, "media_type": media_type})
    items = data.get("collection", {}).get("items", [])
    return items[:limit]


def download_videos(output_dir: Path, limit: int, max_mb: float, queries: List[str]) -> List[Path]:
    downloaded: List[Path] = []
    prefer_tags = ["~medium", "~preview", "~mobile"]

    for query in queries:
        if len(downloaded) >= limit:
            break

        for item in _nasa_items(query, "video", limit=10):
            if len(downloaded) >= limit:
                break

            href = item.get("href")
            if not href:
                continue
            try:
                assets = _fetch_json(href)
            except Exception:
                continue

            best = _pick_best_asset(assets, max_mb=max_mb, prefer=prefer_tags)
            if not best:
                continue

            filename = best.split("/")[-1]
            dest = output_dir / filename
            if dest.exists():
                downloaded.append(dest)
                continue

            print(f"Downloading video: {best}")
            _download(best, dest)
            downloaded.append(dest)

    return downloaded


def download_audio(output_dir: Path, max_mb: float, query: str) -> Optional[Path]:
    prefer_tags = ["~128k", "~64k", "~orig"]

    for item in _nasa_items(query, "audio", limit=8):
        href = item.get("href")
        if not href:
            continue
        try:
            assets = _fetch_json(href)
        except Exception:
            continue

        mp3s = [u for u in assets if u.endswith(".mp3")]
        if not mp3s:
            continue

        ranked = []
        for tag in prefer_tags:
            ranked.extend([u for u in mp3s if tag in u])
        ranked.extend([u for u in mp3s if u not in ranked])

        for url in ranked:
            size_mb = _head_size_mb(url)
            if size_mb is not None and size_mb > max_mb:
                continue
            filename = url.split("/")[-1]
            dest = output_dir / filename
            if dest.exists():
                return dest
            print(f"Downloading audio: {url}")
            _download(url, dest)
            return dest

    return None


def trim_audio(input_path: Path, output_path: Path, duration: int) -> Optional[Path]:
    if not shutil.which("ffmpeg"):
        return None
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NASA footage and audio for trailer demo.")
    parser.add_argument("--video-dir", default="data/input", help="Destination for video clips")
    parser.add_argument("--audio-dir", default="data/music", help="Destination for audio")
    parser.add_argument("--videos", type=int, default=4, help="Number of video clips to download")
    parser.add_argument("--max-video-mb", type=float, default=200.0, help="Max size per video clip (MB)")
    parser.add_argument("--max-audio-mb", type=float, default=200.0, help="Max size for audio file (MB)")
    parser.add_argument("--duration", type=int, default=30, help="Trim audio duration (seconds)")
    parser.add_argument("--query", action="append", default=[], help="NASA video search query (repeatable)")
    parser.add_argument("--audio-query", default=DEFAULT_AUDIO_QUERY, help="NASA audio search query")

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    queries = args.query or DEFAULT_VIDEO_QUERIES
    print("Searching NASA public-domain media...")
    print(f"   Video queries: {', '.join(queries)}")
    print(f"   Audio query: {args.audio_query}")

    videos = download_videos(video_dir, limit=args.videos, max_mb=args.max_video_mb, queries=queries)
    audio = download_audio(audio_dir, max_mb=args.max_audio_mb, query=args.audio_query)

    if audio:
        trimmed = audio_dir / "voiceover_trimmed.wav"
        try:
            trimmed_path = trim_audio(audio, trimmed, duration=args.duration)
            if trimmed_path:
                print(f"Trimmed audio saved: {trimmed_path}")
        except Exception as exc:
            print(f"Warning: could not trim audio: {exc}")

    print("\nDownload complete")
    print(f"   Videos: {len(videos)}")
    for clip in videos:
        print(f"   - {clip}")
    if audio:
        print(f"   Audio: {audio}")
    else:
        print("   Audio: (none)")

    print("\nNote: NASA media is public domain (U.S. Government work).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
