import click
import os
import subprocess
import sys
import time
from typing import Optional

# Lazy load rich to improve startup time and reduce memory footprint
_console = None

def get_console():
    global _console
    if _console is None:
        from rich.console import Console
        _console = Console()
    return _console

STYLES = {
    "dynamic": "Position-aware pacing (intro→build→climax→outro)",
    "hitchcock": "Suspense - slow build, fast climax",
    "mtv": "Fast 1-2 beat cuts, high energy",
    "action": "Michael Bay rapid cuts",
    "documentary": "Natural, observational",
    "minimalist": "Long contemplative takes",
    "wes_anderson": "Symmetrical, whimsical"
}

@click.group()
def cli():
    """Montage AI - Intelligent Video Post-Production Assistant"""
    pass

@cli.command()
@click.argument("style", required=False)
@click.option("--stabilize/--no-stabilize", default=False, help="Enable video stabilization")
@click.option("--upscale/--no-upscale", default=False, help="Enable AI upscaling")
@click.option("--cgpu/--no-cgpu", default=False, help="Enable cgpu/Gemini for Creative Director")
@click.option("--variants", default=1, help="Number of variants to generate")
@click.option("--docker/--local", default=True, help="Run in Docker container")
def run(style: Optional[str], stabilize: bool, upscale: bool, cgpu: bool, variants: int, docker: bool):
    """Create a video montage."""
    console = get_console()
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    # Interactive style selection if not provided
    if not style:
        console.print(Panel.fit("🎬 Welcome to Montage AI", style="bold blue"))
        console.print("Select a style for your montage:")
        for key, desc in STYLES.items():
            console.print(f"  [bold cyan]{key}[/]: {desc}")
        style = click.prompt("Style", type=click.Choice(list(STYLES.keys())), default="dynamic")

    console.print(f"\n🚀 Starting Montage AI with style: [bold green]{style}[/]")
    
    # Use DRY env mapper
    from .env_mapper import map_options_to_env
    options = {
        "stabilize": stabilize,
        "upscale": upscale,
        "cgpu": cgpu
    }
    env_vars = map_options_to_env(style, options)
    
    if docker:
        cmd = ["docker", "compose", "run", "--rm", "montage-ai", "./montage-ai.sh", "run", style]
        if stabilize: cmd.append("--stabilize")
        if upscale: cmd.append("--upscale") # Note: script might need update to handle this flag if not present
        if cgpu: cmd.append("--cgpu")
        
        # We'll use the existing shell script inside docker for now to keep compatibility
        # But we wrap it nicely
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            progress.add_task(description="Initializing Docker environment...", total=None)
            # In a real scenario we might stream output, but for now let's just run it
            # subprocess.run(cmd, env=env_vars) 
            # Actually, we want to see the output.
            pass
        
        subprocess.run(cmd, env=env_vars)
        
    else:
        # Local run (dev mode)
        console.print("[yellow]Running locally (dev mode)...[/]")
        # Here we would call the python module directly
        # python -m montage_ai.editor ...
        pass

@cli.command()
def web():
    """Start the Web UI."""
    console = get_console()
    console.print("🚀 Starting Web UI...")
    console.print("   Open [link=http://localhost:8080]http://localhost:8080[/link] in your browser")
    subprocess.run(["docker", "compose", "-f", "docker-compose.web.yml", "up"])

@cli.command()
def list():
    """List available styles."""
    console = get_console()
    from rich.table import Table
    
    table = Table(title="Available Styles")
    table.add_column("Style", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")

    for key, desc in STYLES.items():
        table.add_row(key, desc)

    console.print(table)


if __name__ == "__main__":
    cli()
