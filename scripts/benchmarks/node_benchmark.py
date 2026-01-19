#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Dict, Optional


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _cpu_benchmark() -> float:
    payload_mb = int(os.environ.get("MONTAGE_BENCH_CPU_MB", "8"))
    iterations = int(os.environ.get("MONTAGE_BENCH_CPU_ITERS", "20"))
    payload = os.urandom(payload_mb * 1024 * 1024)
    start = _now_ms()
    digest = b""
    for _ in range(iterations):
        digest = hashlib.sha256(payload).digest()
        payload = digest + payload[len(digest):]
    end = _now_ms()
    if not digest:
        raise RuntimeError("CPU benchmark did not produce output")
    return end - start


def _io_benchmark() -> float:
    total_mb = int(os.environ.get("MONTAGE_BENCH_IO_MB", "16"))
    chunk_mb = int(os.environ.get("MONTAGE_BENCH_IO_CHUNK_MB", "4"))
    chunk = os.urandom(chunk_mb * 1024 * 1024)
    chunks = max(1, total_mb // chunk_mb)

    fd, path = tempfile.mkstemp(prefix="montage-bench-", suffix=".bin")
    os.close(fd)
    start = _now_ms()
    try:
        with open(path, "wb") as f:
            for _ in range(chunks):
                f.write(chunk)
        with open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    end = _now_ms()
    return end - start


def _ffmpeg_benchmark() -> Optional[float]:
    if not shutil.which("ffmpeg"):
        return None

    width = int(os.environ.get("MONTAGE_BENCH_FFMPEG_W", "1280"))
    height = int(os.environ.get("MONTAGE_BENCH_FFMPEG_H", "720"))
    fps = int(os.environ.get("MONTAGE_BENCH_FFMPEG_FPS", "30"))
    seconds = float(os.environ.get("MONTAGE_BENCH_FFMPEG_SECONDS", "2.5"))
    threads = int(os.environ.get("MONTAGE_BENCH_FFMPEG_THREADS", "2"))

    fd, output_path = tempfile.mkstemp(prefix="montage-bench-", suffix=".mp4")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}",
        "-t",
        str(seconds),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-threads",
        str(max(1, threads)),
        output_path,
    ]

    start = _now_ms()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass
    end = _now_ms()
    return end - start


def _safe_run(label: str, fn) -> Dict[str, Optional[float]]:
    try:
        return {"value": fn(), "error": None}
    except Exception as exc:
        return {"value": None, "error": f"{label} failed: {exc}"}


def main() -> None:
    node_name = os.environ.get("NODE_NAME") or socket.gethostname()
    results = {
        "node": node_name,
        "host": socket.gethostname(),
        "timestamp": int(time.time()),
    }

    cpu_result = _safe_run("cpu", _cpu_benchmark)
    io_result = _safe_run("io", _io_benchmark)
    ffmpeg_result = _safe_run("ffmpeg", _ffmpeg_benchmark)

    results["cpu_ms"] = cpu_result["value"]
    results["io_ms"] = io_result["value"]
    results["ffmpeg_ms"] = ffmpeg_result["value"]

    errors = [r for r in (cpu_result["error"], io_result["error"], ffmpeg_result["error"]) if r]
    if errors:
        results["errors"] = errors

    total_parts = [v for v in (results["cpu_ms"], results["io_ms"], results["ffmpeg_ms"]) if isinstance(v, (int, float))]
    results["total_ms"] = round(sum(total_parts), 2) if total_parts else None

    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
