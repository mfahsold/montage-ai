#!/usr/bin/env python3
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "benchmark_results" / "cluster_benchmark.json"


def _run(cmd: List[str], input_data: Optional[str] = None) -> str:
    result = subprocess.run(
        cmd,
        input=input_data,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def _sanitize_name(name: str) -> str:
    clean = re.sub(r"[^a-z0-9-]+", "-", name.lower())
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean[:40] if clean else "node"


def _get_nodes() -> List[str]:
    raw = _run(["kubectl", "get", "nodes", "-o", "json"])
    data = json.loads(raw)
    nodes = []
    for item in data.get("items", []):
        labels = item.get("metadata", {}).get("labels", {})
        if "node-role.kubernetes.io/control-plane" in labels or "node-role.kubernetes.io/master" in labels:
            continue
        ready = False
        for condition in item.get("status", {}).get("conditions", []):
            if condition.get("type") == "Ready":
                ready = condition.get("status") == "True"
                break
        if not ready:
            continue
        name = item.get("metadata", {}).get("name")
        if name:
            nodes.append(name)
    return nodes


def _get_default_image(namespace: str) -> Optional[str]:
    try:
        return _run([
            "kubectl",
            "-n",
            namespace,
            "get",
            "deployment",
            "montage-ai-web",
            "-o",
            "jsonpath={.spec.template.spec.containers[0].image}",
        ])
    except RuntimeError:
        return None


def _job_manifest(namespace: str, job_name: str, node_name: str, image: str) -> Dict[str, object]:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {"app": "montage-ai", "component": "benchmark"},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 300,
            "template": {
                "metadata": {"labels": {"app": "montage-ai", "component": "benchmark"}},
                "spec": {
                    "restartPolicy": "Never",
                    "nodeName": node_name,
                    "containers": [
                        {
                            "name": "benchmark",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python3", "/app/scripts/benchmarks/node_benchmark.py"],
                            "env": [
                                {
                                    "name": "NODE_NAME",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "spec.nodeName"}
                                    },
                                }
                            ],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "512Mi"},
                                "limits": {"cpu": "1000m", "memory": "1Gi"},
                            },
                        }
                    ],
                },
            },
        },
    }


def _wait_for_job(namespace: str, job_name: str, timeout: str) -> None:
    _run(
        [
            "kubectl",
            "-n",
            namespace,
            "wait",
            "--for=condition=complete",
            f"job/{job_name}",
            f"--timeout={timeout}",
        ]
    )


def _get_job_logs(namespace: str, job_name: str) -> str:
    return _run(["kubectl", "-n", namespace, "logs", f"job/{job_name}"])


def _delete_job(namespace: str, job_name: str) -> None:
    try:
        _run(["kubectl", "-n", namespace, "delete", "job", job_name, "--ignore-not-found=true"])
    except RuntimeError:
        pass


def _parse_json_log(logs: str) -> Optional[Dict[str, object]]:
    result = None
    for line in logs.splitlines():
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
    return result


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run per-node benchmark jobs and label nodes.")
    parser.add_argument("--namespace", default=os.environ.get("MONTAGE_BENCH_NAMESPACE", "montage-ai"))
    parser.add_argument("--image", default=os.environ.get("MONTAGE_BENCH_IMAGE"))
    parser.add_argument("--timeout", default=os.environ.get("MONTAGE_BENCH_TIMEOUT", "600s"))
    parser.add_argument("--job-prefix", default=os.environ.get("MONTAGE_BENCH_PREFIX", "montage-bench"))
    parser.add_argument("--no-label", action="store_true", help="Do not label nodes after benchmarks.")
    parser.add_argument("--keep-jobs", action="store_true", help="Keep benchmark jobs for inspection.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    namespace = args.namespace
    image = args.image or _get_default_image(namespace)
    if not image:
        raise SystemExit("Unable to determine benchmark image. Set --image or MONTAGE_BENCH_IMAGE.")

    nodes = _get_nodes()
    if not nodes:
        raise SystemExit("No worker nodes found for benchmarking.")

    results: Dict[str, Dict[str, object]] = {}
    job_names: List[str] = []
    for node in nodes:
        suffix = int(time.time()) % 100000
        job_name = f"{args.job_prefix}-{_sanitize_name(node)}-{suffix}"
        job_names.append(job_name)

        manifest = _job_manifest(namespace, job_name, node, image)
        _run(["kubectl", "apply", "-f", "-"], input_data=json.dumps(manifest))

        try:
            _wait_for_job(namespace, job_name, args.timeout)
        except RuntimeError as exc:
            results[node] = {"error": str(exc)}
            if not args.keep_jobs:
                _delete_job(namespace, job_name)
            continue

        logs = _get_job_logs(namespace, job_name)
        parsed = _parse_json_log(logs)
        if parsed:
            results[node] = parsed
        else:
            results[node] = {"error": "No JSON output from benchmark job"}

        if not args.keep_jobs:
            _delete_job(namespace, job_name)

    totals = [
        data.get("total_ms")
        for data in results.values()
        if isinstance(data, dict) and isinstance(data.get("total_ms"), (int, float))
    ]
    median_total = statistics.median(totals) if totals else None

    scores = {}
    if median_total:
        for node, data in results.items():
            total_ms = data.get("total_ms") if isinstance(data, dict) else None
            if isinstance(total_ms, (int, float)) and total_ms > 0:
                score = _clamp(median_total / total_ms, 0.5, 2.0)
            else:
                score = 1.0
            scores[node] = round(score, 2)

    if scores and not args.no_label:
        for node, score in scores.items():
            _run(
                [
                    "kubectl",
                    "label",
                    "node",
                    node,
                    f"montage-ai/bench-score={score}",
                    "--overwrite",
                ]
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "namespace": namespace,
                "image": image,
                "median_total_ms": median_total,
                "scores": scores,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"median_total_ms": median_total, "scores": scores}, sort_keys=True))


if __name__ == "__main__":
    main()
