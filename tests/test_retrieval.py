import unittest
from unittest.mock import patch, MagicMock
import sys
import subprocess
from pathlib import Path

# Add scripts dir to path to import the script
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

import retrieve_results

class TestRetrieval(unittest.TestCase):
    
    @patch('retrieve_results.run_kubectl')
    def test_find_helper_pod(self, mock_run):
        mock_run.return_value = "data-helper"
        pod = retrieve_results.find_helper_pod()
        self.assertEqual(pod, "data-helper")
        
    @patch('retrieve_results.run_kubectl')
    def test_list_remote_files(self, mock_run):
        # Mock ls output (mimic ls --full-time)
        mock_run.return_value = """total 100
drwxr-xr-x 2 root root 4096 2026-01-03 12:00:00 +0000 20260103_120000_v1_PROJECT
-rw-r--r-- 1 root root 1048576 2026-01-03 12:01:00 +0000 video.mp4
"""
        files = retrieve_results.list_remote_files("pod")
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0]['name'], "20260103_120000_v1_PROJECT")
        self.assertTrue(files[0]['is_dir'])
        self.assertEqual(files[1]['name'], "video.mp4")
        self.assertFalse(files[1]['is_dir'])
        self.assertEqual(files[1]['size'], 1048576)

    @patch('retrieve_results.run_kubectl')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('subprocess.run')
    @patch('subprocess.Popen')
    @patch('pathlib.Path.mkdir')
    def test_copy_file_base64(self, mock_mkdir, mock_popen, mock_run, mock_open, mock_kubectl):
        # Simulate kubectl cp failing
        mock_kubectl.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        # Setup mocks for Popen (used in tar and base64)
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.wait.return_value = 0
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        # Setup mocks for run (used in tar and base64)
        def run_side_effect(*args, **kwargs):
            cmd = args[0]
            # Fail tar
            if "tar" in cmd:
                raise subprocess.CalledProcessError(1, "tar")
            # Succeed base64
            if "base64" in cmd:
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)
            
        mock_run.side_effect = run_side_effect
        
        success = retrieve_results.copy_file("pod", "/remote/file", Path("/tmp/local/file"))
        
        # Should succeed via base64
        self.assertTrue(success)
        
        # Verify base64 was called
        # We check if Popen was called with base64
        calls = mock_popen.call_args_list
        base64_called = any("base64" in call[0][0] for call in calls)
        self.assertTrue(base64_called)

if __name__ == '__main__':
    unittest.main()
