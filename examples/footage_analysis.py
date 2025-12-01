"""
Example: Footage Analysis

Shows how to use the deep footage analyzer.
"""

from montage_ai.footage_analyzer import DeepFootageAnalyzer
import glob
import os

# Initialize analyzer
analyzer = DeepFootageAnalyzer(
    sample_frames=10,  # Frames per clip to analyze
    verbose=True       # Print detailed results
)

# Analyze all videos in a directory
video_dir = "./data/input"
video_files = glob.glob(os.path.join(video_dir, "*.mp4"))

for video_path in video_files:
    analysis = analyzer.analyze_clip(video_path)
    
    print(f"\n{analysis.source_file}")
    print(f"  Shot Type: {analysis.narrative.shot_type}")
    print(f"  Energy: {analysis.narrative.energy_level:.2f}")
    print(f"  Best for: {', '.join(analysis.best_used_for)}")

# Print comprehensive summary
analyzer.print_footage_summary()

# Export to JSON
analyzer.export_analysis("./data/output/footage_analysis.json")
