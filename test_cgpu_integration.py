
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile

# Mock missing dependencies
sys.modules['librosa'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['moviepy'] = MagicMock()
sys.modules['moviepy.editor'] = MagicMock()
sys.modules['proglog'] = MagicMock()
sys.modules['scenedetect'] = MagicMock()
sys.modules['scenedetect.detectors'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['opentimelineio'] = MagicMock()
sys.modules['opentimelineio.schema'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['flask'] = MagicMock()
sys.modules['werkzeug'] = MagicMock()
sys.modules['werkzeug.utils'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['jsonschema'] = MagicMock()
sys.modules['color_matcher'] = MagicMock()
sys.modules['openai'] = MagicMock()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from montage_ai.cgpu_utils import CGPUConfig
from montage_ai.wan_vace import WanVACEService, WanConfig
from montage_ai.cgpu_upscaler import upscale_with_cgpu

class TestCGPUIntegration(unittest.TestCase):
    
    def setUp(self):
        # Set env var for cgpu
        os.environ['CGPU_GPU_ENABLED'] = 'true'
        
        self.mock_cgpu_available = patch('montage_ai.cgpu_utils.is_cgpu_available', return_value=True)
        self.mock_run_command = patch('montage_ai.cgpu_utils.run_cgpu_command')
        self.mock_copy = patch('montage_ai.cgpu_utils.cgpu_copy_to_remote', return_value=True)
        
        self.is_available = self.mock_cgpu_available.start()
        self.run_command = self.mock_run_command.start()
        self.copy_to_remote = self.mock_copy.start()
        
        # Default mock responses
        self.run_command.return_value = (True, "Success", "")

    def tearDown(self):
        self.mock_cgpu_available.stop()
        self.mock_run_command.stop()
        self.mock_copy.stop()
        if 'CGPU_GPU_ENABLED' in os.environ:
            del os.environ['CGPU_GPU_ENABLED']

    def test_wan_vace_generation(self):
        """Test Wan2.1 video generation command construction"""
        print("\nTesting Wan2.1 Video Generation...")
        
        # Force reload module to pick up env var
        import importlib
        import montage_ai.wan_vace
        importlib.reload(montage_ai.wan_vace)
        from montage_ai.wan_vace import WanVACEService
        
        service = WanVACEService()
        
        # Mock responses for run_cgpu_command
        # 1. Setup environment (must return "setup complete")
        # 2. Generation command (must return base64 output directly)
        self.run_command.side_effect = [
            (True, "setup complete", ""), 
            (True, "===VIDEO_BASE64_START===\nAAAA\n===VIDEO_BASE64_END===", "")
        ]
        
        # Mock cgpu_copy_to_remote (used by _run_cgpu)
        # We need to patch it where it is imported in wan_vace.py
        with patch('montage_ai.wan_vace.cgpu_copy_to_remote', return_value=True):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                result = service.generate_broll(
                    prompt="A cinematic drone shot of a futuristic city",
                    output_path=tmp.name
                )
                
                # Verify commands
                # Both setup and generation run via _run_cgpu which calls "python /tmp/wan_script.py"
                # We expect 2 calls to run_cgpu_command
                self.assertEqual(self.run_command.call_count, 2, "Should call run_cgpu_command twice (setup + generate)")
                
                # Verify the command used
                for args, _ in self.run_command.call_args_list:
                    self.assertIn("python /tmp/wan_script.py", args[0])
                
                print("✓ Wan2.1 generation flow verified")

    def test_upscale_with_cgpu(self):
        """Test cgpu upscaling flow"""
        print("\nTesting cgpu Upscaling...")
        
        # Force reload module to pick up env var
        import importlib
        import montage_ai.cgpu_upscaler
        importlib.reload(montage_ai.cgpu_upscaler)
        from montage_ai.cgpu_upscaler import upscale_with_cgpu
        
        # Mock local file operations and subprocess calls
        with patch('os.path.getsize', return_value=1024*1024*5), \
             patch('subprocess.run') as mock_subprocess:
            
            # Mock subprocess.run for local ffprobe/ffmpeg calls
            # 1. ffprobe (fps)
            # 2. ffmpeg (audio extract)
            mock_subprocess.return_value.stdout = "30.0"
            mock_subprocess.return_value.returncode = 0
            
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_in, \
                 tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_out:
                
                # Mock responses for run_cgpu_command
                # 1. Setup environment (must return "ENV_READY") - called because _colab_env_ready=False
                # 2. Create job dir (mkdir)
                # 3. Run upscale script (python3)
                # 4. Check output size (stat)
                # 5. Download output (cat base64)
                # 6. Cleanup (Success path)
                # 7. Cleanup (Finally path)
                self.run_command.side_effect = [
                    (True, "ENV_READY", ""),
                    (True, "", ""),
                    (True, "PIPELINE_SUCCESS", ""),
                    (True, "1024", ""),
                    (True, "AAAA", ""), # Valid base64
                    (True, "", ""),
                    (True, "", "")
                ]
                
                # Reset global state to force setup
                montage_ai.cgpu_upscaler._colab_env_ready = False
                
                result = upscale_with_cgpu(tmp_in.name, tmp_out.name, scale=2)
                
                # Verify unique job directory creation
                mkdir_called = any("mkdir -p" in args[0] and "/content/upscale_work/" in args[0] for args, _ in self.run_command.call_args_list)
                self.assertTrue(mkdir_called, "Unique job directory should be created")
                
                print("✓ Upscaling flow verified")

if __name__ == '__main__':
    unittest.main()
