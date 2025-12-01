#!/usr/bin/env python3
"""
Montage AI - Command Line Interface

Usage:
    montage-ai create [options] <input_dir> <music_file>
    montage-ai analyze <input_dir>
    montage-ai styles
    montage-ai --help

Examples:
    # Create montage with natural language prompt
    montage-ai create ./videos ./music.mp3 --prompt "Edit like Hitchcock"
    
    # Create with specific style template
    montage-ai create ./videos ./music.mp3 --style mtv
    
    # Analyze footage without creating montage
    montage-ai analyze ./videos
    
    # List available styles
    montage-ai styles
"""

import os
import sys
import argparse
from typing import Optional


def create_montage(
    input_dir: str,
    music_path: str,
    output_dir: str = "./output",
    creative_prompt: Optional[str] = None,
    style: Optional[str] = None,
    variants: int = 1,
    stabilize: bool = False,
    upscale: bool = False,
    enhance: bool = True,
    export_timeline: bool = False,
    verbose: bool = True
):
    """Create a video montage."""
    from .editor import create_montage as run_montage
    
    # Set environment variables for editor
    os.environ["INPUT_DIR"] = input_dir
    os.environ["MUSIC_DIR"] = os.path.dirname(music_path)
    os.environ["OUTPUT_DIR"] = output_dir
    os.environ["VERBOSE"] = "true" if verbose else "false"
    os.environ["NUM_VARIANTS"] = str(variants)
    os.environ["STABILIZE"] = "true" if stabilize else "false"
    os.environ["UPSCALE"] = "true" if upscale else "false"
    os.environ["ENHANCE"] = "true" if enhance else "false"
    os.environ["EXPORT_TIMELINE"] = "true" if export_timeline else "false"
    
    if creative_prompt:
        os.environ["CREATIVE_PROMPT"] = creative_prompt
    elif style:
        os.environ["CREATIVE_PROMPT"] = style
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the montage creation
    for variant in range(1, variants + 1):
        run_montage(variant_id=variant)


def analyze_footage(input_dir: str, verbose: bool = True):
    """Analyze footage without creating montage."""
    from .footage_analyzer import DeepFootageAnalyzer
    import glob
    
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not video_files:
        print(f"❌ No video files found in {input_dir}")
        return
    
    print(f"\n🔬 Deep Footage Analysis")
    print(f"   Found {len(video_files)} video files\n")
    
    analyzer = DeepFootageAnalyzer(sample_frames=10, verbose=verbose)
    
    for video_path in video_files:
        analyzer.analyze_clip(video_path)
    
    analyzer.print_footage_summary()
    
    # Export to JSON
    output_file = os.path.join(input_dir, "footage_analysis.json")
    analyzer.export_analysis(output_file)
    print(f"\n✅ Analysis exported to {output_file}")


def list_styles():
    """List available style templates."""
    from .style_templates import STYLE_TEMPLATES
    
    print("\n🎬 Available Style Templates")
    print("=" * 60)
    
    for name, template in STYLE_TEMPLATES.items():
        print(f"\n  {name}")
        print(f"  └─ {template['description']}")
        
        params = template.get('params', {})
        pacing = params.get('pacing', {})
        effects = params.get('effects', {})
        
        speed = pacing.get('speed', 'dynamic')
        color = effects.get('color_grading', 'neutral')
        
        print(f"     Speed: {speed} | Color: {color}")
    
    print("\n" + "=" * 60)
    print("Usage: montage-ai create ./videos ./music.mp3 --style <name>")
    print("   Or: montage-ai create ./videos ./music.mp3 --prompt 'your description'")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="montage-ai",
        description="AI-powered automatic video montage creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  montage-ai create ./videos ./music.mp3 --prompt "Edit like Hitchcock"
  montage-ai create ./videos ./music.mp3 --style mtv --variants 3
  montage-ai analyze ./videos
  montage-ai styles
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a video montage")
    create_parser.add_argument("input_dir", help="Directory containing video files")
    create_parser.add_argument("music_path", help="Path to music file")
    create_parser.add_argument("-o", "--output", default="./output", help="Output directory")
    create_parser.add_argument("-p", "--prompt", help="Natural language creative prompt")
    create_parser.add_argument("-s", "--style", help="Style template name")
    create_parser.add_argument("-n", "--variants", type=int, default=1, help="Number of variants")
    create_parser.add_argument("--stabilize", action="store_true", help="Enable stabilization")
    create_parser.add_argument("--upscale", action="store_true", help="Enable AI upscaling")
    create_parser.add_argument("--no-enhance", action="store_true", help="Disable enhancement")
    create_parser.add_argument("--timeline", action="store_true", help="Export timeline (OTIO/EDL)")
    create_parser.add_argument("-q", "--quiet", action="store_true", help="Less output")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze footage")
    analyze_parser.add_argument("input_dir", help="Directory containing video files")
    analyze_parser.add_argument("-q", "--quiet", action="store_true", help="Less output")
    
    # Styles command
    subparsers.add_parser("styles", help="List available style templates")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_montage(
            input_dir=args.input_dir,
            music_path=args.music_path,
            output_dir=args.output,
            creative_prompt=args.prompt,
            style=args.style,
            variants=args.variants,
            stabilize=args.stabilize,
            upscale=args.upscale,
            enhance=not args.no_enhance,
            export_timeline=args.timeline,
            verbose=not args.quiet
        )
    elif args.command == "analyze":
        analyze_footage(args.input_dir, verbose=not args.quiet)
    elif args.command == "styles":
        list_styles()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
