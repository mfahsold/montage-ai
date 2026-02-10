#!/usr/bin/env python3
"""
Quality Enhancement Testing Suite
Tests all available quality improvements on existing footage.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import subprocess
import tempfile
import time

# Add src to path
sys.path.insert(0, '/home/codeai/montage-ai/src')

from montage_ai.config import get_settings
from montage_ai.logger import logger
from montage_ai.clip_enhancement import ClipEnhancer
from montage_ai.audio_enhancer import AudioEnhancer
from montage_ai.color_grading import ColorGradePreset
from montage_ai.scene_analysis import SceneDetector
from montage_ai.auto_reframe import AutoReframeEngine
from montage_ai.ffmpeg_utils import build_ffmpeg_cmd
from montage_ai.core.cmd_runner import run_command

settings = get_settings()
INPUT_DIR = Path(settings.paths.input_dir)
OUTPUT_DIR = Path(settings.paths.output_dir)

# Test configuration
TEST_SAMPLE_COUNT = 3  # Test with 3 representative clips
QUALITY_TESTS = {
    'stabilization': True,      # Motion stabilization
    'color_grading': True,       # Teal/Orange grading
    'upscaling': False,          # AI upscaling (expensive)
    'scene_detection': True,     # Scene breaks
    'audio_enhance': True,       # Voice polish
    'reframing': False,          # 16:9 → 9:16 (optional, needs MediaPipe)
}

class QualityEnhancementTester:
    """Test and evaluate quality improvements."""
    
    def __init__(self):
        self.results = {}
        self.enhancer = ClipEnhancer()
        self.audio_enhancer = AudioEnhancer()
        self.scene_detector = SceneDetector()
        self.test_clips: List[Path] = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix="montage_test_"))
        logger.info(f"Test temp dir: {self.temp_dir}")
    
    def get_sample_clips(self, count: int = 3) -> List[Path]:
        """Get representative sample clips from input directory."""
        if not INPUT_DIR.exists():
            logger.error(f"Input directory not found: {INPUT_DIR}")
            return []
        
        all_clips = sorted(INPUT_DIR.glob("*.mp4")) + sorted(INPUT_DIR.glob("*.mov"))
        logger.info(f"Found {len(all_clips)} total clips")
        
        if len(all_clips) < count:
            logger.warning(f"Fewer clips than requested ({len(all_clips)} < {count})")
            return all_clips
        
        # Select diverse samples: short, medium, long
        selected = [
            all_clips[0],                      # First (likely short intro)
            all_clips[len(all_clips)//2],     # Middle (representative)
            all_clips[-1],                     # Last (likely longer content)
        ]
        return selected[:count]
    
    def test_stabilization(self, clip_path: Path) -> Dict:
        """Test video stabilization."""
        logger.info(f"Testing stabilization: {clip_path.name}")
        start = time.time()
        
        output_path = self.temp_dir / f"{clip_path.stem}_stabilized.mp4"
        
        try:
            self.enhancer.stabilize(str(clip_path), str(output_path), fast_mode=True)
            
            if output_path.exists():
                elapsed = time.time() - start
                file_size = output_path.stat().st_size
                logger.info(f"  ✅ Stabilization complete ({elapsed:.1f}s, {file_size/1e6:.1f}MB)")
                return {
                    'status': 'success',
                    'output': str(output_path),
                    'elapsed': elapsed,
                    'file_size': file_size
                }
            else:
                logger.warning("  ⚠️ Output file not created")
                return {'status': 'failed', 'reason': 'no_output'}
        except Exception as e:
            logger.error(f"  ❌ Stabilization failed: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def test_color_grading(self, clip_path: Path) -> Dict:
        """Test color grading (Teal/Orange preset)."""
        logger.info(f"Testing color grading: {clip_path.name}")
        start = time.time()
        
        output_path = self.temp_dir / f"{clip_path.stem}_graded.mp4"
        
        try:
            self.enhancer.enhance(str(clip_path), str(output_path), color_grade="teal_orange")
            
            if output_path.exists():
                elapsed = time.time() - start
                file_size = output_path.stat().st_size
                logger.info(f"  ✅ Color grading complete ({elapsed:.1f}s, {file_size/1e6:.1f}MB)")
                return {
                    'status': 'success',
                    'output': str(output_path),
                    'preset': 'teal_orange',
                    'elapsed': elapsed,
                    'file_size': file_size
                }
            else:
                logger.warning("  ⚠️ Output file not created")
                return {'status': 'failed', 'reason': 'no_output'}
        except Exception as e:
            logger.error(f"  ❌ Color grading failed: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def test_scene_detection(self, clip_path: Path) -> Dict:
        """Test scene/cut detection."""
        logger.info(f"Testing scene detection: {clip_path.name}")
        start = time.time()
        
        try:
            scenes = self.scene_detector.detect(str(clip_path))
            elapsed = time.time() - start
            
            logger.info(f"  ✅ Scene detection complete ({elapsed:.1f}s)")
            logger.info(f"     Detected {len(scenes)} scenes")
            
            if scenes:
                for i, scene in enumerate(scenes[:3]):
                    logger.info(f"     Scene {i+1}: {scene.start:.2f}s - {scene.end:.2f}s ({scene.duration:.2f}s)")
            
            return {
                'status': 'success',
                'scenes_detected': len(scenes),
                'elapsed': elapsed,
                'sample_scenes': [
                    {'start': s.start, 'end': s.end, 'duration': s.duration}
                    for s in scenes[:3]
                ]
            }
        except Exception as e:
            logger.error(f"  ❌ Scene detection failed: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def test_audio_enhance(self, clip_path: Path) -> Dict:
        """Test audio enhancement."""
        logger.info(f"Testing audio enhancement: {clip_path.name}")
        start = time.time()
        
        output_path = self.temp_dir / f"{clip_path.stem}_audio_enhanced.mp4"
        
        try:
            # Extract audio, enhance, re-mux
            temp_audio = self.temp_dir / f"{clip_path.stem}_audio.aac"
            
            # Extract audio
            extract_cmd = build_ffmpeg_cmd([
                "-i", str(clip_path),
                "-q:a", "9",
                "-map", "a:0",
                str(temp_audio)
            ])
            run_command(extract_cmd)
            
            if temp_audio.exists():
                # Enhance audio
                enhanced_audio = self.temp_dir / f"{clip_path.stem}_audio_enhanced.aac"
                self.audio_enhancer.enhance_voice(str(temp_audio), str(enhanced_audio))
                
                # Re-mux
                if enhanced_audio.exists():
                    remux_cmd = build_ffmpeg_cmd([
                        "-i", str(clip_path),
                        "-i", str(enhanced_audio),
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        str(output_path)
                    ])
                    run_command(remux_cmd)
            
            if output_path.exists():
                elapsed = time.time() - start
                file_size = output_path.stat().st_size
                logger.info(f"  ✅ Audio enhancement complete ({elapsed:.1f}s, {file_size/1e6:.1f}MB)")
                return {
                    'status': 'success',
                    'output': str(output_path),
                    'elapsed': elapsed,
                    'file_size': file_size
                }
            else:
                logger.warning("  ⚠️ Output file not created")
                return {'status': 'failed', 'reason': 'no_output'}
        except Exception as e:
            logger.error(f"  ❌ Audio enhancement failed: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def run_all_tests(self):
        """Run all quality tests."""
        logger.info("="*70)
        logger.info("🎬 QUALITY ENHANCEMENT TESTING SUITE")
        logger.info("="*70)
        
        # Get sample clips
        self.test_clips = self.get_sample_clips(TEST_SAMPLE_COUNT)
        if not self.test_clips:
            logger.error("No test clips available!")
            return
        
        logger.info(f"\nTesting with {len(self.test_clips)} sample clips:")
        for clip in self.test_clips:
            logger.info(f"  - {clip.name} ({clip.stat().st_size/1e6:.1f}MB)")
        
        # Test first clip with all features
        test_clip = self.test_clips[0]
        logger.info(f"\n📸 Primary test clip: {test_clip.name}")
        logger.info("-" * 70)
        
        clip_results = {}
        
        if QUALITY_TESTS['stabilization']:
            logger.info("\n[1/5] Stabilization Test")
            clip_results['stabilization'] = self.test_stabilization(test_clip)
        
        if QUALITY_TESTS['color_grading']:
            logger.info("\n[2/5] Color Grading Test")
            clip_results['color_grading'] = self.test_color_grading(test_clip)
        
        if QUALITY_TESTS['scene_detection']:
            logger.info("\n[3/5] Scene Detection Test")
            clip_results['scene_detection'] = self.test_scene_detection(test_clip)
        
        if QUALITY_TESTS['audio_enhance']:
            logger.info("\n[4/5] Audio Enhancement Test")
            clip_results['audio_enhance'] = self.test_audio_enhance(test_clip)
        
        logger.info("\n[5/5] Summary")
        logger.info("-" * 70)
        
        self.results['clip_tests'] = clip_results
        
        # Print summary
        logger.info("\n📊 TEST RESULTS:")
        for test_name, result in clip_results.items():
            status = result.get('status', 'unknown')
            emoji = '✅' if status == 'success' else '❌'
            logger.info(f"{emoji} {test_name}: {status}")
        
        # Save results
        results_file = OUTPUT_DIR / "quality_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n📝 Results saved to: {results_file}")
    
    def cleanup(self):
        """Clean up test files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp dir: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temp dir: {e}")


if __name__ == "__main__":
    tester = QualityEnhancementTester()
    try:
        tester.run_all_tests()
    finally:
        # Optionally keep temp files for inspection
        logger.info(f"\nTemp files available at: {tester.temp_dir}")
        logger.info("(Will be auto-cleaned on next run)")
