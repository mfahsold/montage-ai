"""
Centralized Pool/Worker Configuration - Hardware-Aware Defaults

Dynamically sizes thread pools, process pools, and HTTP connection pools
based on available CPU cores and environment configuration.

Usage:
    from montage_ai.config_pools import PoolConfig
    
    # Process pool for CPU-bound tasks
    num_workers = PoolConfig.process_workers()
    
    # Thread pool for I/O-bound tasks
    num_workers = PoolConfig.thread_workers()
    
    # HTTP adapter pool
    conn, maxsize = PoolConfig.http_pool_size()

Environment Variables:
    PROCESS_WORKERS=8              # CPU-bound task workers (default: cpu_count//2)
    THREAD_WORKERS=16              # I/O-bound task workers (default: cpu_count)
    HTTP_CONNECTIONS=32            # HTTP connection pool (default: min(cpu_count*2, 32))
    HTTP_MAXSIZE=64                # HTTP max pool size (default: min(cpu_count*4, 64))
    QUEUE_SIZE=20                  # Web UI queue (default: max(10, cpu_count*2))
    FFMPEG_THREADS=0               # FFmpeg encoder threads (0=auto)

Default Behavior:
    - Single core: 1 process, 1 thread, HTTP pool 2/4
    - Quad core: 2 process, 4 thread, HTTP pool 8/16
    - 16-core: 8 process, 16 thread, HTTP pool 32/64
    - 64-core: 32 process, 64 thread, HTTP pool 32/64 (capped)

Rationale:
    - Process workers: 50% of cores (reserve for system)
    - Thread workers: 100% of cores (I/O doesn't block)
    - HTTP connections: 2x cores but capped (too many = overhead)
    - HTTP maxsize: 4x cores but capped
"""

import os
from typing import Tuple


class PoolConfig:
    """
    Hardware-aware pool configuration with environment variable support.
    """
    
    @staticmethod
    def process_workers() -> int:
        """
        Number of workers for ProcessPoolExecutor (CPU-bound tasks).
        
        Use for: Scene detection, beat analysis, heavy computation
        
        Default: 50% of CPU cores (reserves other half for system)
        Range: [2, 32] (reasonable for most hardware)
        
        Returns:
            Number of worker processes
        """
        cpu_count = os.cpu_count() or 2
        default = max(2, cpu_count // 2)
        
        value = int(os.getenv("PROCESS_WORKERS", str(default)))
        
        # Sanity check: don't exceed CPU count
        return min(value, cpu_count)
    
    @staticmethod
    def thread_workers() -> int:
        """
        Number of workers for ThreadPoolExecutor (I/O-bound tasks).
        
        Use for: FFmpeg subprocess management, file I/O, network requests
        
        Default: 100% of CPU cores (I/O doesn't block cores)
        Range: [1, 128] (reasonable for most hardware)
        
        Returns:
            Number of worker threads
        """
        cpu_count = os.cpu_count() or 2
        default = cpu_count
        
        return int(os.getenv("THREAD_WORKERS", str(default)))
    
    @staticmethod
    def http_pool_size() -> Tuple[int, int]:
        """
        HTTP connection pool size for requests.adapters.HTTPAdapter.
        
        Use for: API calls, CGPU communication, metadata fetch
        
        Default:
            - connections: min(cpu_count * 2, 32)  # Concurrent connections
            - maxsize: min(cpu_count * 4, 64)      # Queue depth per connection
        
        Returns:
            (pool_connections, pool_maxsize) tuple
        """
        cpu_count = os.cpu_count() or 2
        
        # Default sizing
        default_conn = min(cpu_count * 2, 32)
        default_maxsize = min(cpu_count * 4, 64)
        
        # Allow env overrides
        conn = int(os.getenv("HTTP_CONNECTIONS", str(default_conn)))
        maxsize = int(os.getenv("HTTP_MAXSIZE", str(default_maxsize)))
        
        return conn, maxsize
    
    @staticmethod
    def queue_size() -> int:
        """
        Event queue size for web UI SSE (Server-Sent Events).
        
        Use for: Web UI job status updates, live progress
        
        Default: max(10, cpu_count * 2)
        
        Returns:
            Queue size
        """
        cpu_count = os.cpu_count() or 2
        default = max(10, cpu_count * 2)
        
        return int(os.getenv("QUEUE_SIZE", str(default)))
    
    @staticmethod
    def ffmpeg_threads() -> int:
        """
        FFmpeg encoder thread count.
        
        Use for: FFmpeg -threads parameter
        
        Default: 0 (auto-detect, FFmpeg uses available cores)
        
        Returns:
            Thread count (0 = auto)
        """
        return int(os.getenv("FFMPEG_THREADS", "0"))
    
    @staticmethod
    def print_config() -> None:
        """Print current pool configuration for debugging."""
        print("\n" + "=" * 70)
        print("POOL CONFIGURATION")
        print("=" * 70)
        
        cpu_count = os.cpu_count() or 2
        print(f"\nSystem CPU Cores: {cpu_count}")
        
        config = {
            "ProcessPool (CPU-bound)": {
                "workers": PoolConfig.process_workers(),
                "optimal_for": f"Scene detection, beat analysis"
            },
            "ThreadPool (I/O-bound)": {
                "workers": PoolConfig.thread_workers(),
                "optimal_for": "FFmpeg, file I/O, network requests"
            },
            "HTTP Adapter": {
                "connections": PoolConfig.http_pool_size()[0],
                "maxsize": PoolConfig.http_pool_size()[1],
                "optimal_for": "API calls, CGPU communication"
            },
            "Web UI Queue": {
                "size": PoolConfig.queue_size(),
                "optimal_for": "Live event streaming"
            },
            "FFmpeg": {
                "threads": PoolConfig.ffmpeg_threads(),
                "note": "0 = auto-detect"
            }
        }
        
        for category, config_items in config.items():
            print(f"\n{category}:")
            for key, value in config_items.items():
                if isinstance(value, int):
                    print(f"  {key:20} = {value}")
                else:
                    print(f"  {key:20} = {value}")
        
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    PoolConfig.print_config()
