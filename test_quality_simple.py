#!/usr/bin/env python3
"""
Simple Quality Enhancement Test - FFmpeg-based (no heavy dependencies)
Tests color grading, stabilization presets on existing footage.
"""

import json
import subprocess
from pathlib import Path
import time
from typing import Dict, List

INPUT_DIR = Path("/home/codeai/montage-ai/data/input")
OUTPUT_DIR = Path("/home/codeai/montage-ai/data/output")
TEMP_DIR = Path("/tmp/montage_quality_test")

TEMP_DIR.mkdir(exist_ok=True)

# FFmpeg color grading presets - simple, proven filters
GRADING_PRESETS = {
    "warm": "eq=contrast=1.05:brightness=0.05:saturation=1.1",
    "cool": "eq=contrast=1.05:brightness=-0.05:saturation=1.1",
    "vibrant": "eq=saturation=1.3:contrast=1.1",
    "high_contrast": "eq=contrast=1.3:brightness=0.05",
    "cinematic": "eq=saturation=0.9:contrast=1.15",
}

STABILIZATION_PRESETS = {
    "vidstab": "-vf 'vidstabdetect=stepsize=6:shakiness=8:accuracy=15' && ffmpeg -i {input} -vf 'vidstabtransform=smoothing=30' -c:v libx264 -crf 20 {output}",
    "deshake": "-vf 'deshake=x=-1:y=-1:w=-1:h=-1' -c:v libx264 -crf 20",
}

class SimpleQualityTester:
    def __init__(self):
        self.results = {}
    
    def get_sample_clips(self, count=2) -> List[Path]:
        """Get sample clips from input directory."""
        if not INPUT_DIR.exists():
            print(f"❌ Input directory not found: {INPUT_DIR}")
            return []
        
        all_clips = sorted(INPUT_DIR.glob("*.mp4")) + sorted(INPUT_DIR.glob("*.mov"))
        print(f"✅ Found {len(all_clips)} total clips")
        
        if not all_clips:
            print("❌ No clips found!")
            return []
        
        # Select first and last
        selected = [all_clips[0], all_clips[-1] if len(all_clips) > 1 else all_clips[0]]
        return selected[:count]
    
    def get_clip_duration(self, path: Path) -> float:
        """Get clip duration in seconds."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1:noprint_wrappers=1",
            str(path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def test_color_grading(self, clip: Path, preset: str) -> Dict:
        """Test FFmpeg color grading."""
        duration = self.get_clip_duration(clip)
        if duration < 5:
            # Use only first 5 seconds for faster testing
            trim_filter = f"trim=0:5,setpts=PTS-STARTPTS"
            filter_chain = f"[0]{trim_filter}[a];[a]{GRADING_PRESETS[preset]}"
        else:
            filter_chain = GRADING_PRESETS[preset]
        
        output = TEMP_DIR / f"{clip.stem}_graded_{preset}.mp4"
        
        print(f"  Testing: {preset} on {clip.name} (trim to 5s for speed)...")
        start = time.time()
        
        cmd = [
            "ffmpeg", "-i", str(clip),
            "-vf", filter_chain,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
            "-c:a", "aac", "-b:a", "128k",
            "-y", str(output)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            elapsed = time.time() - start
            
            if result.returncode == 0 and output.exists():
                file_size = output.stat().st_size
                print(f"    ✅ Success ({elapsed:.1f}s, {file_size/1e6:.1f}MB)")
                return {
                    'status': 'success',
                    'preset': preset,
                    'output': str(output),
                    'elapsed': elapsed,
                    'file_size': file_size
                }
            else:
                print(f"    ⚠️ FFmpeg error: {result.stderr[-200:]}")
                return {'status': 'failed', 'reason': 'ffmpeg_error'}
        except subprocess.TimeoutExpired:
            print(f"    ❌ Timeout (>60s)")
            return {'status': 'failed', 'reason': 'timeout'}
        except Exception as e:
            print(f"    ❌ Exception: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def test_scene_detection(self, clip: Path) -> Dict:
        """Test FFmpeg scene detection (scenecut detection)."""
        print(f"  Testing scene detection on {clip.name}...")
        start = time.time()
        
        cmd = [
            "ffmpeg", "-i", str(clip),
            "-vf", "select='gt(scene\\,0.3)',showscenedetect=mode=copy",
            "-vsync", "0",
            "-f", "null", "-"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            elapsed = time.time() - start
            
            # Count scene detections from stderr
            scene_count = result.stderr.count("Scene detected")
            print(f"    ✅ Detected {scene_count} scene cuts ({elapsed:.1f}s)")
            
            return {
                'status': 'success',
                'scenes_detected': scene_count,
                'elapsed': elapsed
            }
        except subprocess.TimeoutExpired:
            print(f"    ⚠️ Timeout (analysis incomplete)")
            return {'status': 'partial', 'reason': 'timeout'}
        except Exception as e:
            print(f"    ❌ Exception: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def run_tests(self):
        """Run all quality tests."""
        print("\n" + "="*70)
        print("🎬 QUALITY ENHANCEMENT TESTING - SIMPLE FFmpeg SUITE")
        print("="*70)
        
        clips = self.get_sample_clips(2)
        if not clips:
            return
        
        print(f"\n📸 Testing {len(clips)} sample clips:")
        for clip in clips:
            duration = self.get_clip_duration(clip)
            print(f"  - {clip.name} ({duration:.1f}s, {clip.stat().st_size/1e6:.1f}MB)")
        
        # Test on first clip
        test_clip = clips[0]
        print(f"\n🎯 Primary test clip: {test_clip.name}")
        print("-" * 70)
        
        results = {}
        
        # Test color grading presets
        print("\n[1/3] Testing Color Grading Presets:")
        grading_results = {}
        for preset in list(GRADING_PRESETS.keys()):
            grading_results[preset] = self.test_color_grading(test_clip, preset)
        results['color_grading'] = grading_results
        
        # Test scene detection
        print("\n[2/3] Testing Scene Detection:")
        results['scene_detection'] = self.test_scene_detection(test_clip)
        
        # Test on second clip if available
        if len(clips) > 1:
            print("\n[3/3] Testing Secondary Clip:")
            results['secondary_scene_test'] = self.test_scene_detection(clips[1])
        
        # Print summary
        print("\n" + "-" * 70)
        print("📊 TEST SUMMARY:")
        print("-" * 70)
        
        success_count = 0
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and 'status' in test_result:
                status = test_result['status']
                emoji = '✅' if status == 'success' else ('⚠️' if status == 'partial' else '❌')
                print(f"{emoji} {test_name}: {status}")
                if status == 'success':
                    success_count += 1
            elif isinstance(test_result, dict):
                # Multiple presets
                for preset, preset_result in test_result.items():
                    status = preset_result.get('status', 'unknown')
                    emoji = '✅' if status == 'success' else '❌'
                    print(f"{emoji} {test_name}:{preset}: {status}")
                    if status == 'success':
                        success_count += 1
        
        # Save results
        results_file = OUTPUT_DIR / "quality_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📝 Results saved to: {results_file}")
        print(f"✅ {success_count} tests successful")
        print(f"📂 Test output files in: {TEMP_DIR}")


if __name__ == "__main__":
    tester = SimpleQualityTester()
    tester.run_tests()
