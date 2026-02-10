"""
Montage AI - Deployment Verification

Single-command system validation that checks all prerequisites,
hardware, storage, and optional backends.

Usage:
    python -m montage_ai.verify_deployment
    ./montage-ai.sh verify-deployment
    ./montage-ai.sh verify-deployment --format json
    ./montage-ai.sh verify-deployment --verbose
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Check:
    """A single verification check result."""
    name: str
    status: str  # "ok", "warn", "fail", "info"
    detail: str = ""

    @property
    def icon(self) -> str:
        return {"ok": "\u2705", "warn": "\u26a0\ufe0f", "fail": "\u274c", "info": "\u2139\ufe0f"}.get(
            self.status, "?"
        )


@dataclass
class Section:
    """A group of related checks."""
    name: str
    icon: str
    checks: List[Check] = field(default_factory=list)


def _run(cmd: List[str], timeout: int = 10) -> Optional[str]:
    """Run a command, return stdout or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _get_version(cmd: List[str], pattern: Optional[str] = None) -> Optional[str]:
    """Get version string from a command."""
    out = _run(cmd)
    if out is None:
        return None
    if pattern:
        import re
        match = re.search(pattern, out)
        return match.group(1) if match else out.split()[0]
    return out.strip().split("\n")[0]


def _check_system() -> Section:
    """Check system-level prerequisites."""
    section = Section(name="SYSTEM", icon="\U0001f4cb")

    # Python
    py_ver = sys.version.split()[0]
    section.checks.append(Check("Python", "ok", py_ver))

    # FFmpeg
    ffmpeg_out = _get_version(["ffmpeg", "-version"], r"ffmpeg version (\S+)")
    if ffmpeg_out:
        section.checks.append(Check("FFmpeg", "ok", ffmpeg_out))
    else:
        section.checks.append(Check("FFmpeg", "fail", "not found"))

    # Docker
    docker_ver = _get_version(["docker", "--version"], r"(\d+\.\d+\.\d+)")
    if docker_ver:
        section.checks.append(Check("Docker", "ok", docker_ver))
    else:
        section.checks.append(Check("Docker", "warn", "not found (needed for Docker workflow)"))

    # Docker Compose
    compose_out = _run(["docker", "compose", "version"])
    if compose_out:
        import re
        match = re.search(r"(\d+\.\d+\.\d+)", compose_out)
        ver = match.group(1) if match else "available"
        section.checks.append(Check("Docker Compose", "ok", f"v{ver}"))
    else:
        section.checks.append(Check("Docker Compose", "warn", "not found"))

    # Disk space
    try:
        usage = shutil.disk_usage("/")
        free_gb = usage.free / (1024**3)
        if free_gb >= 10:
            section.checks.append(Check("Disk", "ok", f"{free_gb:.0f} GB available"))
        elif free_gb >= 5:
            section.checks.append(Check("Disk", "warn", f"{free_gb:.0f} GB available (10 GB+ recommended)"))
        else:
            section.checks.append(Check("Disk", "fail", f"{free_gb:.0f} GB available (need 5 GB+)"))
    except OSError:
        section.checks.append(Check("Disk", "warn", "could not check"))

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 * 1024)
                    if gb >= 16:
                        section.checks.append(Check("RAM", "ok", f"{gb:.0f} GB"))
                    elif gb >= 8:
                        section.checks.append(Check("RAM", "warn", f"{gb:.0f} GB (16 GB recommended)"))
                    else:
                        section.checks.append(Check("RAM", "fail", f"{gb:.0f} GB (8 GB minimum)"))
                    break
    except (OSError, ValueError):
        # macOS fallback
        sysctl_out = _run(["sysctl", "-n", "hw.memsize"])
        if sysctl_out:
            gb = int(sysctl_out) / (1024**3)
            status = "ok" if gb >= 16 else "warn"
            section.checks.append(Check("RAM", status, f"{gb:.0f} GB"))

    return section


def _check_gpu() -> Section:
    """Check GPU acceleration availability."""
    section = Section(name="GPU ACCELERATION", icon="\U0001f3ae")

    # NVIDIA
    nvidia_out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if nvidia_out:
        section.checks.append(Check("NVIDIA GPU", "ok", nvidia_out.split("\n")[0]))
    else:
        section.checks.append(Check("NVIDIA GPU", "info", "not detected (optional)"))

    # VAAPI
    if os.path.exists("/dev/dri/renderD128"):
        if os.access("/dev/dri/renderD128", os.R_OK):
            section.checks.append(Check("VAAPI", "ok", "/dev/dri/renderD128 readable"))
        else:
            section.checks.append(
                Check("VAAPI", "warn", "/dev/dri/renderD128 exists but not readable (check group permissions)")
            )
    else:
        section.checks.append(Check("VAAPI", "info", "no DRI device found"))

    # Active encoder (via hardware module)
    try:
        from .core.hardware import get_best_hwaccel
        best = get_best_hwaccel()
        if best.is_gpu:
            section.checks.append(Check("Active encoder", "ok", f"{best.type.upper()} ({best.encoder})"))
        else:
            section.checks.append(Check("Active encoder", "info", f"CPU ({best.encoder})"))
    except Exception as exc:
        section.checks.append(Check("Active encoder", "warn", f"probe failed: {exc}"))

    return section


def _check_storage() -> Section:
    """Check data directories."""
    section = Section(name="STORAGE", icon="\U0001f4c1")

    data_root = Path(os.environ.get("DATA_DIR", "/data"))
    if not data_root.exists():
        data_root = Path("data")

    write_required = {"output"}
    for subdir in ("input", "output", "music", "assets"):
        path = data_root / subdir
        if path.is_dir():
            try:
                usage = shutil.disk_usage(str(path))
                size_gb = usage.used / (1024**3)
                if subdir in write_required:
                    accessible = os.access(str(path), os.W_OK)
                    flag = "not writable"
                else:
                    accessible = os.access(str(path), os.R_OK)
                    flag = "not readable"
                status = "ok" if accessible else "warn"
                detail = f"{size_gb:.1f} GB"
                if not accessible:
                    detail += f" ({flag})"
                section.checks.append(Check(str(path), status, detail))
            except OSError:
                section.checks.append(Check(str(path), "ok", "exists"))
        else:
            section.checks.append(Check(str(path), "warn", "missing (run ./scripts/setup.sh)"))

    return section


def _check_llm() -> Section:
    """Check AI/LLM backend availability."""
    section = Section(name="AI/LLM BACKENDS", icon="\U0001f916")

    # OpenAI-compatible
    openai_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if openai_base:
        section.checks.append(Check("OpenAI API", "ok", openai_base))
    else:
        section.checks.append(Check("OpenAI API", "info", "OPENAI_API_BASE not configured (optional)"))

    # Ollama
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    if ollama_host:
        section.checks.append(Check("Ollama", "ok", ollama_host))
    else:
        section.checks.append(Check("Ollama", "info", "OLLAMA_HOST not set (optional)"))

    # Google/Gemini
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if google_key:
        section.checks.append(Check("Google Gemini", "ok", "API key configured"))
    else:
        section.checks.append(Check("Google Gemini", "info", "GOOGLE_API_KEY not set (optional)"))

    # cgpu
    if shutil.which("cgpu"):
        section.checks.append(Check("cgpu", "ok", "installed"))
    else:
        section.checks.append(Check("cgpu", "info", "not installed (optional)"))

    return section


def _check_docker() -> Section:
    """Check Docker runtime."""
    section = Section(name="DOCKER", icon="\U0001f433")

    # Daemon running
    docker_ps = _run(["docker", "ps", "-q"])
    if docker_ps is not None:
        section.checks.append(Check("Docker daemon", "ok", "running"))
    else:
        section.checks.append(Check("Docker daemon", "fail", "not running or not accessible"))
        return section

    # Compose v2
    compose_out = _run(["docker", "compose", "version"])
    if compose_out:
        section.checks.append(Check("docker compose v2", "ok", "available"))
    else:
        section.checks.append(Check("docker compose v2", "fail", "not available"))

    # Image built
    images_out = _run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "--filter", "reference=montage-ai"])
    if images_out:
        section.checks.append(Check("Image", "ok", f"montage-ai ({images_out.split(chr(10))[0]})"))
    else:
        section.checks.append(Check("Image", "warn", "montage-ai image not built (run: docker compose build)"))

    return section


def _check_kubernetes() -> Section:
    """Check Kubernetes connectivity (if kubectl available)."""
    section = Section(name="KUBERNETES", icon="\u2638\ufe0f")

    if not shutil.which("kubectl"):
        section.checks.append(Check("kubectl", "info", "not installed (K8s deployment optional)"))
        return section

    section.checks.append(Check("kubectl", "ok", "installed"))

    # Cluster connectivity
    cluster_info = _run(["kubectl", "cluster-info"], timeout=5)
    if cluster_info:
        section.checks.append(Check("Cluster", "ok", "connected"))
    else:
        section.checks.append(Check("Cluster", "warn", "not connected"))
        return section

    # StorageClass
    sc_out = _run(["kubectl", "get", "storageclass", "-o", "jsonpath={.items[*].metadata.name}"])
    if sc_out:
        classes = sc_out.strip().split()
        section.checks.append(Check("StorageClass", "ok", ", ".join(classes)))
    else:
        section.checks.append(Check("StorageClass", "warn", "none found"))

    # config-global.yaml placeholder check
    config_path = Path("deploy/k3s/config-global.yaml")
    if config_path.exists():
        content = config_path.read_text()
        if "<" in content and ">" in content:
            section.checks.append(Check("config-global.yaml", "warn", "contains unresolved <...> placeholders"))
        else:
            section.checks.append(Check("config-global.yaml", "ok", "valid (no placeholders)"))
    else:
        section.checks.append(Check("config-global.yaml", "info", "not created yet"))

    # kustomize
    if shutil.which("kustomize"):
        section.checks.append(Check("kustomize", "ok", "installed"))
    else:
        section.checks.append(Check("kustomize", "warn", "not installed (needed for K8s deployment)"))

    return section


def _check_codecs() -> Section:
    """Check codec/encoder support."""
    section = Section(name="CODEC SUPPORT", icon="\U0001f4ca")

    encoders_out = _run(["ffmpeg", "-encoders", "-hide_banner"])
    if encoders_out is None:
        section.checks.append(Check("FFmpeg encoders", "fail", "ffmpeg not available"))
        return section

    codec_checks = [
        ("H.264", "libx264"),
        ("H.265", "libx265"),
        ("ProRes", "prores"),
    ]

    for label, encoder in codec_checks:
        if encoder in encoders_out:
            section.checks.append(Check(label, "ok", encoder))
        else:
            section.checks.append(Check(label, "warn", f"{encoder} not available"))

    return section


def run_verification(
    json_format: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run all verification checks and return results."""
    sections = [
        _check_system(),
        _check_gpu(),
        _check_storage(),
        _check_llm(),
        _check_docker(),
        _check_kubernetes(),
        _check_codecs(),
    ]

    # Determine overall status
    all_checks = [c for s in sections for c in s.checks]
    has_fail = any(c.status == "fail" for c in all_checks)
    has_warn = any(c.status == "warn" for c in all_checks)

    if has_fail:
        overall = "ISSUES FOUND"
    elif has_warn:
        overall = "READY (with warnings)"
    else:
        overall = "READY FOR PRODUCTION"

    result = {
        "overall": overall,
        "sections": [
            {
                "name": s.name,
                "checks": [asdict(c) for c in s.checks],
            }
            for s in sections
        ],
    }

    if json_format:
        return result

    # Text output
    print()
    print("Montage AI - Deployment Verification Report")
    print("\u2550" * 47)

    for s in sections:
        print(f"\n{s.icon} {s.name}")
        for c in s.checks:
            if not verbose and c.status == "info":
                # In non-verbose, skip purely informational items
                # unless they're the only check in the section
                if len(s.checks) > 1:
                    continue
            detail = f" ({c.detail})" if c.detail else ""
            print(f"  {c.icon} {c.name}{detail}")

    print()
    if has_fail:
        print("\u274c " + overall)
    elif has_warn:
        print("\u26a0\ufe0f  " + overall)
    else:
        print("\u2705 " + overall)
    print()

    return result


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Montage AI Deployment Verification")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all checks including informational")
    args = parser.parse_args()

    json_format = args.format == "json"
    result = run_verification(json_format=json_format, verbose=args.verbose)

    if json_format:
        print(json.dumps(result, indent=2))

    # Exit code: 1 if any failures
    all_checks = [
        c
        for s in result.get("sections", [])
        for c in (s if isinstance(s, list) else s.get("checks", []))
    ]
    has_fail = any(
        (c.get("status") if isinstance(c, dict) else getattr(c, "status", "")) == "fail"
        for c in all_checks
    )
    sys.exit(1 if has_fail else 0)


if __name__ == "__main__":
    main()
