"""
Montage AI - Unified CLI Entry Point

This module provides a unified command-line interface for Montage AI,
replacing the legacy procedural entry point in editor.py.
"""

import os
import sys
import json
import click
import logging
from pathlib import Path
from typing import Optional

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

if __name__ == "__main__":
    cli()
