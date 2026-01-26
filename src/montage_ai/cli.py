"""
Montage AI - Unified CLI Entry Point

This module provides a unified command-line interface for Montage AI,
replacing the legacy procedural entry point in editor.py.
"""

import os
import sys
import json
import time
import click
import logging
import zipfile
from pathlib import Path
from typing import Optional, Any, Dict, Iterable

import requests

from .logger import logger
from .config import get_settings
from .core.hardware import hwaccel_probe

# Setting up basic logging for the CLI
logging.basicConfig(level=logging.INFO, format='%(message)s')

@click.group()
@click.version_option(package_name="montage-ai")
def cli():
    """Montage AI - Post-Production Assistant."""
    pass

@cli.command(name="check-hw")
@click.option("--json", "json_format", is_flag=True, help="Output results in JSON format")
@click.option("--verbose", "-v", is_flag=True, help="Include detailed error diagnostics")
def check_hw(json_format: bool, verbose: bool):
    """
    Diagnose available hardware accelerators (NVENC, VAAPI, QSV).
    
    Performs functional probes to ensure drivers and permissions are correctly
    configured for video encoding.
    """
    if not json_format:
        click.echo("🔍 Probing hardware accelerators...")
    
    report = hwaccel_probe()
    
    if json_format:
        click.echo(json.dumps(report, indent=2))
        return

    # Visual reporting
    best = report.get("best_available", {})
    best_type = best.get("type", "cpu")
    
    if best.get("is_gpu"):
        click.secho(f"\n✅ Hardware acceleration is available ({best_type.upper()})", fg="green", bold=True)
    else:
        click.secho(f"\n⚠️ Hardware acceleration NOT found. Falling back to CPU.", fg="yellow", bold=True)
        
    click.echo("-" * 40)
    
    # Detail table-like output
    accelerators = report.get("accelerators", [])
    for acc in accelerators:
        acc_type = acc.get("type", "unknown")
        available = acc.get("available", False)
        functional = acc.get("functional", False)
        
        if not available:
            status = "missing"
            icon = "⚪"
            color = "white"
        elif not functional:
            status = "BROKEN"
            icon = "❌"
            color = "red"
        else:
            status = "working"
            icon = "✅"
            color = "green"
        
        click.echo(f"{icon} ", nl=False)
        click.secho(f"{acc_type.upper():<12}", fg="blue", bold=True, nl=False)
        click.secho(f" : {status}", fg=color)
        
        if verbose and acc.get("error"):
            click.echo(f"   Diagnostic: {acc.get('error')}")
        elif status == "BROKEN" and not verbose:
            click.echo("   (Run with -v for diagnostics)")

    if best_type == "cpu" and os.getuid() != 0:
        click.echo("\n💡 Tip: If you have the hardware but it's not detected,")
        click.echo("   ensure you have the right drivers and group permissions (e.g. 'render' or 'video').")

@cli.command(name="run")
@click.argument("prompt", required=False)
@click.option("--style", "-s", help="Editing style (e.g. hitchcock, dynamic, mtv)")
@click.option("--output", "-o", help="Output file path")
@click.pass_context
def run_montage(ctx, prompt: Optional[str], style: Optional[str], output: Optional[str]):
    """Run the montage creation pipeline (Legacy facade)."""
    # This currently delegates back to editor.py's logic
    # In the future, we will move more logic here or common/core
    from .editor import main as editor_main
    
    # Inject values into environment or config as handled by editor.py
    if style:
        os.environ["CUT_STYLE"] = style
    if prompt:
        os.environ["CREATIVE_PROMPT"] = prompt
    if output:
        os.environ["OUTPUT_PATH"] = output
        
    # editor_main() uses sys.argv indirectly via its global configs
    # For now, just call it.
    editor_main()

def _normalize_api_base(api_base: str) -> str:
    base = (api_base or "").strip()
    if not base:
        raise click.ClickException("API base URL is empty.")
    return base.rstrip("/")


def _build_api_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/api") and path.startswith("/api/"):
        path = path[4:]
    return f"{base}{path}"


def _parse_option_pairs(pairs: Iterable[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.ClickException(f"Invalid --option '{pair}'. Use key=value.")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise click.ClickException("Option key cannot be empty.")
        value: Any
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        if key.startswith("options."):
            opt_key = key[len("options."):]
            if not opt_key:
                raise click.ClickException("Option key cannot end with 'options.'.")
            parsed.setdefault("options", {})[opt_key] = value
        else:
            parsed[key] = value
    return parsed


def _load_payload(payload: Optional[str], payload_file: Optional[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if payload_file:
        try:
            with open(payload_file, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise click.ClickException(f"Failed to read payload file: {exc}") from exc
        if not isinstance(loaded, dict):
            raise click.ClickException("Payload file must contain a JSON object.")
        data.update(loaded)
    if payload:
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON payload: {exc}") from exc
        if not isinstance(loaded, dict):
            raise click.ClickException("Payload must be a JSON object.")
        data.update(loaded)
    return data


def _request_json(api_base: str, method: str, path: str, timeout: int, **kwargs: Any) -> Any:
    url = _build_api_url(api_base, path)
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise click.ClickException(f"Failed to reach {url}: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text.strip()
        raise click.ClickException(f"{response.status_code} {response.reason}: {detail}")

    try:
        return response.json()
    except ValueError:
        return response.text


def _download_file(api_base: str, url_path: str, dest_path: Path, timeout: int) -> None:
    url = _build_api_url(api_base, url_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        if response.status_code >= 400:
            raise click.ClickException(f"{response.status_code} {response.reason}: {response.text}")
        with dest_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _download_job_artifacts(
    api_base: str,
    job_id: str,
    dest_root: Path,
    timeout: int,
    create_zip: bool,
    log: callable,
) -> Path:
    artifacts = _request_json(
        api_base,
        "GET",
        f"/api/jobs/{job_id}/artifacts",
        timeout,
    )
    if not isinstance(artifacts, dict):
        raise click.ClickException("Unexpected artifacts response from API.")

    artifact_map = artifacts.get("artifacts", {})
    if not any(artifact_map.values()):
        raise click.ClickException(f"No artifacts found for job {job_id}.")

    job_dir = dest_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths: list[Path] = []

    for category, files in artifact_map.items():
        if not files:
            continue
        if category == "video":
            category_dir = job_dir
        else:
            category_dir = job_dir / category
            category_dir.mkdir(exist_ok=True)
        for file_info in files:
            name = file_info.get("name")
            url_path = file_info.get("download_url")
            if not name or not url_path:
                continue
            dest_path = category_dir / name
            log(f"Downloading {name}...", err=True)
            _download_file(api_base, url_path, dest_path, timeout)
            downloaded_paths.append(dest_path)

    if create_zip:
        zip_path = job_dir / f"montage_{job_id}.zip"
        log(f"Creating ZIP: {zip_path}", err=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in downloaded_paths:
                arcname = path.relative_to(job_dir)
                zf.write(path, arcname.as_posix())
        return zip_path

    return job_dir


def _wait_for_job(
    api_base: str,
    job_id: str,
    request_timeout: int,
    wait_timeout: int,
    poll_interval: int,
    log: callable,
) -> Dict[str, Any]:
    start = time.time()
    last_status = None

    while True:
        job = _request_json(
            api_base,
            "GET",
            f"/api/jobs/{job_id}",
            request_timeout,
        )
        if not isinstance(job, dict):
            raise click.ClickException("Unexpected job status response from API.")

        status = job.get("status", "unknown")
        phase = job.get("phase", {}).get("label") or job.get("phase", {}).get("name", "")

        if status != last_status:
            message = f"Job {job_id}: {status}"
            if phase:
                message = f"{message} ({phase})"
            log(message, err=True)
            last_status = status

        if status in {"completed", "success", "finished"}:
            return job
        if status in {"failed", "cancelled", "error"}:
            raise click.ClickException(f"Job {job_id} ended with status: {status}")

        if poll_interval > 0:
            time.sleep(poll_interval)
        if wait_timeout > 0 and (time.time() - start) > wait_timeout:
            raise click.ClickException(f"Timed out waiting for job {job_id}.")


@cli.group()
@click.option("--api-base", "--api", default=None, envvar="MONTAGE_API_BASE", help="API base URL (no /api suffix required).")
@click.option("--timeout", default=20, show_default=True, help="HTTP timeout in seconds.")
@click.pass_context
def jobs(ctx: click.Context, api_base: Optional[str], timeout: int):
    """Submit and manage jobs via the Montage API."""
    if not api_base:
        api_base = os.environ.get("BACKEND_API_URL") or "http://localhost:8080"
    ctx.ensure_object(dict)
    ctx.obj["api_base"] = _normalize_api_base(api_base)
    ctx.obj["timeout"] = timeout


@jobs.command("submit")
@click.option("--style", help="Editing style (required unless payload provides it).")
@click.option("--prompt", help="Creative prompt for the editor.")
@click.option("--quality-profile", help="Quality profile (preview/standard/high/etc).")
@click.option("--preset", help="Preset name (e.g. fast for preview).")
@click.option("--option", "options", multiple=True, help="Extra key=value fields (use options.<key>=... for nested options).")
@click.option("--payload", help="Raw JSON payload string to merge.")
@click.option("--payload-file", type=click.Path(exists=True, dir_okay=False), help="Path to JSON payload file.")
@click.option("--output", type=click.Choice(["id", "json"], case_sensitive=False), default="id", show_default=True)
@click.option("--download", is_flag=True, help="Wait for completion and download artifacts to this machine.")
@click.option("--download-dir", type=click.Path(file_okay=False), default="./downloads", show_default=True, help="Directory to store downloaded artifacts.")
@click.option("--download-zip", is_flag=True, help="Create a ZIP from downloaded artifacts.")
@click.option("--poll-interval", default=10, show_default=True, help="Polling interval (seconds) when waiting for completion.")
@click.option("--wait-timeout", default=0, show_default=True, help="Max seconds to wait (0 = no timeout).")
@click.pass_context
def jobs_submit(
    ctx: click.Context,
    style: Optional[str],
    prompt: Optional[str],
    quality_profile: Optional[str],
    preset: Optional[str],
    options: Iterable[str],
    payload: Optional[str],
    payload_file: Optional[str],
    output: str,
    download: bool,
    download_dir: str,
    download_zip: bool,
    poll_interval: int,
    wait_timeout: int,
):
    """Submit a job via POST /api/jobs."""
    data = _load_payload(payload, payload_file)
    data.update(_parse_option_pairs(options))

    if style:
        data["style"] = style
    if prompt:
        data["prompt"] = prompt
    if quality_profile:
        data["quality_profile"] = quality_profile
    if preset:
        data["preset"] = preset

    if not data.get("style"):
        raise click.ClickException("Missing required field: style")

    result = _request_json(
        ctx.obj["api_base"],
        "POST",
        "/api/jobs",
        ctx.obj["timeout"],
        json=data,
    )

    job_id = result.get("id") if isinstance(result, dict) else None
    if output == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(job_id or json.dumps(result, indent=2))

    if download:
        if not job_id:
            raise click.ClickException("API response did not include job id.")
        download_root = Path(download_dir).expanduser()
        _wait_for_job(
            ctx.obj["api_base"],
            job_id,
            ctx.obj["timeout"],
            wait_timeout,
            poll_interval,
            click.echo,
        )
        downloaded_path = _download_job_artifacts(
            ctx.obj["api_base"],
            job_id,
            download_root,
            ctx.obj["timeout"],
            download_zip,
            click.echo,
        )
        click.echo(f"Downloaded artifacts to: {downloaded_path}", err=True)


@jobs.command("status")
@click.argument("job_id")
@click.option("--output", type=click.Choice(["json", "status"], case_sensitive=False), default="json", show_default=True)
@click.pass_context
def jobs_status(ctx: click.Context, job_id: str, output: str):
    """Fetch job status via GET /api/jobs/<id>."""
    result = _request_json(
        ctx.obj["api_base"],
        "GET",
        f"/api/jobs/{job_id}",
        ctx.obj["timeout"],
    )
    if output == "status" and isinstance(result, dict):
        click.echo(result.get("status", "unknown"))
        return
    click.echo(json.dumps(result, indent=2) if isinstance(result, dict) else result)


@jobs.command("list")
@click.pass_context
def jobs_list(ctx: click.Context):
    """List jobs via GET /api/jobs."""
    result = _request_json(
        ctx.obj["api_base"],
        "GET",
        "/api/jobs",
        ctx.obj["timeout"],
    )
    click.echo(json.dumps(result, indent=2) if isinstance(result, dict) else result)


@jobs.command("cancel")
@click.argument("job_id")
@click.pass_context
def jobs_cancel(ctx: click.Context, job_id: str):
    """Cancel a job via POST /api/jobs/<id>/cancel."""
    result = _request_json(
        ctx.obj["api_base"],
        "POST",
        f"/api/jobs/{job_id}/cancel",
        ctx.obj["timeout"],
    )
    click.echo(json.dumps(result, indent=2) if isinstance(result, dict) else result)

if __name__ == "__main__":
    cli()
