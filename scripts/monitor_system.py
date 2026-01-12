#!/usr/bin/env python3
"""Lightweight system monitor that watches a PID and records CPU/memory/GPU metrics.
Usage: python scripts/monitor_system.py --pid <pid> --outfile /path/to/out.json
"""
import argparse
import json
import time
from datetime import datetime, timezone

try:
    import psutil
except Exception:
    psutil = None

import subprocess


def sample_gpu():
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        if smi.returncode == 0 and smi.stdout.strip():
            parts = [p.strip() for p in smi.stdout.strip().split(',')]
            return {"util": parts[0], "mem_used": parts[1] if len(parts) > 1 else None}
    except Exception:
        return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--outfile", type=str, required=True)
    p.add_argument("--interval", type=float, default=5.0)
    args = p.parse_args()

    data = []
    pid = args.pid
    interval = args.interval

    while True:
        ts = datetime.now(timezone.utc).isoformat()
        if psutil:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()._asdict()
        else:
            cpu = None
            mem = None

        gpu = sample_gpu()
        data.append({"ts": ts, "cpu": cpu, "mem": mem, "gpu": gpu})

        # Write incremental
        try:
            with open(args.outfile, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

        # Check process
        try:
            proc = psutil.Process(pid) if psutil else None
            alive = proc.is_running() if proc else True
            if not alive:
                break
        except Exception:
            # Process does not exist
            break

        time.sleep(interval)

    # Final write
    try:
        with open(args.outfile, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


if __name__ == '__main__':
    main()
