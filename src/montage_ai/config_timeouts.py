"""
Centralized Timeout Configuration - Single Source of Truth

All subprocess, CGPU, HTTP, and probe timeouts defined here.
Supports env var overrides for different environments.

Usage:
    from montage_ai.config_timeouts import TimeoutConfig
    
    # Get timeout with defaults
    timeout = TimeoutConfig.probe_quick()  # 5 seconds
    
    # Or with env override
    timeout = TimeoutConfig.get("probe_quick")  # Checks TIMEOUT_PROBE_QUICK
    
    # Use in subprocess
    result = subprocess.run(cmd, timeout=TimeoutConfig.subprocess_default())

Environment Variables:
    TIMEOUT_PROBE_QUICK=5              # Hardware detection quick probes
    TIMEOUT_PROBE_K8S=10               # Kubernetes cluster detection
    TIMEOUT_SUBPROCESS_QUICK=2         # Quick subprocess ops (version, etc)
    TIMEOUT_SUBPROCESS_DEFAULT=30      # FFmpeg, standard operations
    TIMEOUT_SUBPROCESS_LONG=60         # Audio analysis, complex ops
    TIMEOUT_CGPU_QUICK=30              # Quick cloud operations
    TIMEOUT_CGPU_STANDARD=120          # Standard cloud jobs
    TIMEOUT_CGPU_UPLOAD=180            # File upload/download
    TIMEOUT_CGPU_ENCODING=3600         # Video encoding
    TIMEOUT_HTTP_DEFAULT=10            # HTTP requests
    TIMEOUT_HTTP_LONG=30               # Long HTTP operations
    TIMEOUT_ENCODING_LOCAL=3600        # Local video encoding
    TIMEOUT_NETWORK_CONNECT=5          # Network connection establishment

Rationale for Default Values:
    - PROBE_QUICK (5s): Should detect hardware quickly, fail fast if unavailable
    - SUBPROCESS_QUICK (2s): Version checks, quick info gathering
    - SUBPROCESS_DEFAULT (30s): FFmpeg typical operations
    - SUBPROCESS_LONG (60s): Complex audio/video processing
    - CGPU_QUICK (30s): Cloud service quick response
    - CGPU_STANDARD (120s): Typical cloud job setup/cleanup
    - CGPU_UPLOAD (180s): Large file transfers
    - CGPU_ENCODING (3600s): Video encoding (up to 1 hour)
    - HTTP_DEFAULT (10s): API calls, metadata fetch
    - HTTP_LONG (30s): Long-running API operations
"""

import os
from typing import Optional


class TimeoutConfig:
    """
    Centralized timeout configuration with environment variable support.
    
    All methods support env var overrides using TIMEOUT_<NAME> pattern.
    """
    
    # =========================================================================
    # Hardware Detection & Probing
    # =========================================================================
    
    @staticmethod
    def probe_quick() -> int:
        """
        Quick hardware/connectivity probes.
        
        Used for: nvidia-smi, vainfo, qsv detection, quick sanity checks
        Default: 5 seconds
        """
        return int(os.getenv("TIMEOUT_PROBE_QUICK", "5"))
    
    @staticmethod
    def probe_kubernetes() -> int:
        """
        Kubernetes cluster detection via kubectl.
        
        Used for: kubectl get nodes, cluster status checks
        Default: 10 seconds
        """
        return int(os.getenv("TIMEOUT_PROBE_K8S", "10"))
    
    @staticmethod
    def network_connect() -> int:
        """
        Network connection establishment timeout.
        
        Used for: HTTP connects, CGPU host connection
        Default: 5 seconds
        """
        return int(os.getenv("TIMEOUT_NETWORK_CONNECT", "5"))
    
    # =========================================================================
    # Subprocess Operations (Local FFmpeg, etc.)
    # =========================================================================
    
    @staticmethod
    def subprocess_quick() -> int:
        """
        Quick subprocess operations.
        
        Used for: --version, --help, quick metadata queries
        Default: 2 seconds
        """
        return int(os.getenv("TIMEOUT_SUBPROCESS_QUICK", "2"))
    
    @staticmethod
    def subprocess_default() -> int:
        """
        Standard subprocess operations.
        
        Used for: FFmpeg info, scene detection, audio analysis
        Default: 30 seconds
        """
        return int(os.getenv("TIMEOUT_SUBPROCESS_DEFAULT", "30"))
    
    @staticmethod
    def subprocess_long() -> int:
        """
        Long-running subprocess operations.
        
        Used for: Complex audio ducking, voice isolation, multi-pass encoding
        Default: 60 seconds
        """
        return int(os.getenv("TIMEOUT_SUBPROCESS_LONG", "60"))
    
    # =========================================================================
    # Cloud GPU (CGPU) Operations
    # =========================================================================
    
    @staticmethod
    def cgpu_quick() -> int:
        """
        Quick CGPU cloud operations.
        
        Used for: Connection checks, quick status, small commands
        Default: 30 seconds
        """
        return int(os.getenv("TIMEOUT_CGPU_QUICK", "30"))
    
    @staticmethod
    def cgpu_standard() -> int:
        """
        Standard CGPU cloud operations.
        
        Used for: Job setup, model loading, standard processing
        Default: 120 seconds (2 minutes)
        """
        return int(os.getenv("TIMEOUT_CGPU_STANDARD", "120"))
    
    @staticmethod
    def cgpu_upload() -> int:
        """
        CGPU file upload/download operations.
        
        Used for: Transferring files to/from cloud (via base64 or rsync)
        Default: 180 seconds (3 minutes)
        
        Note: Scales with file size in practice
        """
        return int(os.getenv("TIMEOUT_CGPU_UPLOAD", "180"))
    
    @staticmethod
    def cgpu_encoding() -> int:
        """
        CGPU video encoding operations.
        
        Used for: Remote encoding jobs (NVENC on Tesla T4)
        Default: 3600 seconds (1 hour)
        
        Note: May need increase for large files (10GB+)
        """
        return int(os.getenv("TIMEOUT_CGPU_ENCODING", "3600"))
    
    # =========================================================================
    # HTTP/Network Operations
    # =========================================================================
    
    @staticmethod
    def http_default() -> int:
        """
        Standard HTTP request timeout.
        
        Used for: API calls, metadata fetch, health checks
        Default: 10 seconds
        """
        return int(os.getenv("TIMEOUT_HTTP_DEFAULT", "10"))
    
    @staticmethod
    def http_long() -> int:
        """
        Long HTTP request timeout.
        
        Used for: Large file uploads, long-polling, streaming
        Default: 30 seconds
        """
        return int(os.getenv("TIMEOUT_HTTP_LONG", "30"))
    
    # =========================================================================
    # Video Encoding (Local)
    # =========================================================================
    
    @staticmethod
    def encoding_local() -> int:
        """
        Local video encoding timeout.
        
        Used for: ffmpeg final render, segment encoding
        Default: 3600 seconds (1 hour)
        
        Note: Typically completes in 5-15 min for 1080p, longer for 4K+
        """
        return int(os.getenv("TIMEOUT_ENCODING_LOCAL", "3600"))
    
    # =========================================================================
    # Generic Get Method
    # =========================================================================
    
    @staticmethod
    def get(operation: str, default: int) -> int:
        """
        Generic timeout getter with env override.
        
        Args:
            operation: Operation name (e.g., "probe_quick", "cgpu_encoding")
            default: Default timeout in seconds
        
        Returns:
            Timeout value from env var or default
        
        Example:
            timeout = TimeoutConfig.get("custom_op", 30)
            # Checks TIMEOUT_CUSTOM_OP env var
        """
        env_key = f"TIMEOUT_{operation.upper()}"
        return int(os.getenv(env_key, str(default)))
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @staticmethod
    def print_config() -> None:
        """Print current timeout configuration for debugging."""
        print("\n" + "=" * 70)
        print("TIMEOUT CONFIGURATION")
        print("=" * 70)
        
        config = {
            "Hardware Detection": {
                "probe_quick": TimeoutConfig.probe_quick(),
                "probe_kubernetes": TimeoutConfig.probe_kubernetes(),
                "network_connect": TimeoutConfig.network_connect(),
            },
            "Subprocess": {
                "quick": TimeoutConfig.subprocess_quick(),
                "default": TimeoutConfig.subprocess_default(),
                "long": TimeoutConfig.subprocess_long(),
            },
            "Cloud GPU (CGPU)": {
                "quick": TimeoutConfig.cgpu_quick(),
                "standard": TimeoutConfig.cgpu_standard(),
                "upload": TimeoutConfig.cgpu_upload(),
                "encoding": TimeoutConfig.cgpu_encoding(),
            },
            "HTTP/Network": {
                "default": TimeoutConfig.http_default(),
                "long": TimeoutConfig.http_long(),
            },
            "Encoding": {
                "local": TimeoutConfig.encoding_local(),
            },
        }
        
        for category, timeouts in config.items():
            print(f"\n{category}:")
            for name, value in timeouts.items():
                print(f"  {name:20} = {value:5}s")
        
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    TimeoutConfig.print_config()
