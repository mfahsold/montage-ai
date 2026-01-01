#!/usr/bin/env python3
"""
Evaluation Harness for Storytelling Engine.
Verifies that the solver produces a timeline that adheres to the requested StoryArc.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from montage_ai.storytelling import StoryArc, TensionProvider, StorySolver

def evaluate_story_engine():
    print("=== Story Engine Evaluation ===")
    
    # 1. Setup
    duration = 60.0
    beats = [i * 2.0 for i in range(1, 31)] # Cut every 2 seconds
    
    # Mock clips
    clips = [f"/data/input/clip_{i}.mp4" for i in range(20)]
    
    # Use dummy provider (deterministic random tensions)
    provider = TensionProvider(Path("/tmp"), allow_dummy=True)
    
    # Test different arcs
    presets = ["hero_journey", "mtv_energy", "slow_burn"]
    
    for preset in presets:
        print(f"\nTesting Preset: {preset}")
        arc = StoryArc.from_preset(preset)
        solver = StorySolver(arc, provider)
        
        timeline = solver.solve(clips, duration, beats)
        
        # Calculate MSE
        squared_errors = []
        print(f"{'Time':<6} | {'Target':<6} | {'Actual':<6} | {'Diff':<6}")
        print("-" * 30)
        
        for i, event in enumerate(timeline):
            # Calculate duration from next event
            start_time = event['time']
            if i < len(timeline) - 1:
                end_time = timeline[i+1]['time']
            else:
                end_time = duration
            
            event_duration = end_time - start_time
            midpoint = start_time + (event_duration / 2)
            progress = midpoint / duration
            
            target = arc.get_target_tension(progress)
            actual = event['clip_tension']
            diff = abs(target - actual)
            squared_errors.append(diff ** 2)
            
            if len(squared_errors) <= 5 or len(squared_errors) > len(timeline) - 5:
                 print(f"{midpoint:5.1f}  | {target:5.2f}  | {actual:5.2f}  | {diff:5.2f}")
        
        if not squared_errors:
            print("Error: No timeline generated.")
            continue
            
        mse = sum(squared_errors) / len(squared_errors)
        print(f"Mean Squared Error: {mse:.4f}")
        
        if mse < 0.15: # Allow some variance as we have limited clips
            print("✅ PASS")
        else:
            print("❌ FAIL (MSE too high)")

if __name__ == "__main__":
    evaluate_story_engine()
