"""
Example: Basic Montage Creation

Demonstrates the simplest usage of Montage AI.
"""

from montage_ai import MontageEditor

# Create editor instance
editor = MontageEditor()

# Create a montage with natural language prompt
editor.create_montage(
    video_dir="./data/input",
    music_path="./data/music/track.mp3",
    output_dir="./data/output",
    creative_prompt="Create a cinematic gallery video with elegant transitions"
)
