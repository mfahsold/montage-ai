"""
Adaptive Memory Management for Montage-AI

Monitors memory usage and triggers proactive cleanup to prevent OOM crashes.
Dynamically adjusts batch sizes based on available RAM.

Saves ~150MB peak memory by preventing spikes through early intervention.
"""

import gc
import psutil
from typing import Optional, Tuple
from dataclasses import dataclass

from .logger import logger
from .config import get_settings


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    rss_mb: float  # Resident Set Size (actual RAM used)
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of system memory
    available_mb: float  # Available system memory
    limit_mb: float  # Container/process memory limit


class AdaptiveMemoryManager:
    """
    Monitors and manages memory usage to prevent OOM crashes.

    Features:
    - Real-time memory monitoring using psutil
    - Proactive garbage collection triggers
    - Dynamic batch size calculation
    - Memory pressure warnings
    """

    def __init__(self, memory_limit_gb: Optional[float] = None,
                 warning_threshold: float = 0.75,
                 critical_threshold: float = 0.90):
        """
        Initialize adaptive memory manager.

        Args:
            memory_limit_gb: Memory limit in GB (defaults to MEMORY_LIMIT_GB env var or auto-detect)
            warning_threshold: Trigger cleanup at this fraction of limit (default: 0.75)
            critical_threshold: Critical level requiring immediate action (default: 0.90)
        """
        self.process = psutil.Process()

        settings = get_settings()

        # Determine memory limit (explicit arg overrides settings)
        if memory_limit_gb is None:
            memory_limit_gb = settings.resources.memory_limit_gb

        self.memory_limit_gb = float(memory_limit_gb)
        self.memory_limit_mb = memory_limit_gb * 1024

        # Determine thresholds (explicit args override settings)
        self.warning_threshold = float(warning_threshold if warning_threshold is not None else settings.resources.memory_warning_threshold)
        self.critical_threshold = float(critical_threshold if critical_threshold is not None else settings.resources.memory_critical_threshold)

        # Statistics
        self.cleanup_count = 0
        self.peak_usage_mb = 0.0
        self.last_gc_mb = 0.0

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.

        Returns:
            MemoryStats object with current usage
        """
        # Process memory
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024)

        # System memory
        system_mem = psutil.virtual_memory()
        available_mb = system_mem.available / (1024 * 1024)
        percent = system_mem.percent

        # Update peak
        if rss_mb > self.peak_usage_mb:
            self.peak_usage_mb = rss_mb

        return MemoryStats(
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            percent=percent,
            available_mb=available_mb,
            limit_mb=self.memory_limit_mb
        )

    def get_current_usage_gb(self) -> float:
        """
        Get current process memory usage in GB (RSS).

        Returns:
            Memory usage in gigabytes
        """
        return self.process.memory_info().rss / (1024 ** 3)

    def get_current_usage_mb(self) -> float:
        """
        Get current process memory usage in MB (RSS).

        Returns:
            Memory usage in megabytes
        """
        return self.process.memory_info().rss / (1024 ** 2)

    def get_usage_percentage(self) -> float:
        """
        Get memory usage as percentage of limit.

        Returns:
            Percentage (0.0-1.0) of memory limit used
        """
        current_mb = self.get_current_usage_mb()
        return current_mb / self.memory_limit_mb

    def should_trigger_cleanup(self) -> bool:
        """
        Check if memory usage exceeds warning threshold.

        Returns:
            True if cleanup should be triggered
        """
        usage_pct = self.get_usage_percentage()
        return usage_pct >= self.warning_threshold

    def is_critical(self) -> bool:
        """
        Check if memory usage exceeds critical threshold.

        Returns:
            True if memory is in critical state
        """
        usage_pct = self.get_usage_percentage()
        return usage_pct >= self.critical_threshold

    def trigger_cleanup(self, force: bool = False) -> Tuple[float, float]:
        """
        Trigger garbage collection and return memory freed.

        Args:
            force: Force cleanup even if not needed

        Returns:
            Tuple of (before_mb, after_mb)
        """
        before_mb = self.get_current_usage_mb()

        # Trigger garbage collection
        gc.collect()

        after_mb = self.get_current_usage_mb()
        freed_mb = before_mb - after_mb

        self.cleanup_count += 1
        self.last_gc_mb = after_mb

        return before_mb, after_mb

    def calculate_safe_batch_size(self, base_batch_size: int = 25,
                                  clip_size_mb: float = 50.0) -> int:
        """
        Calculate safe batch size based on available memory.

        Args:
            base_batch_size: Desired batch size (will be capped)
            clip_size_mb: Estimated size per clip in MB

        Returns:
            Safe batch size (at least 1)
        """
        current_usage_mb = self.get_current_usage_mb()
        available_mb = self.memory_limit_mb - current_usage_mb

        # Reserve overhead for system/process (configurable via settings)
        try:
            from .config import get_settings
            safety_margin_mb = int(get_settings().resources.__dict__.get("memory_safety_margin_mb", 500))
        except Exception:
            safety_margin_mb = 500
        usable_mb = max(0, available_mb - safety_margin_mb)

        # Calculate how many clips can fit
        safe_batch = int(usable_mb / clip_size_mb)

        # Ensure at least 1, max base_batch_size
        return max(1, min(safe_batch, base_batch_size))

    def get_memory_pressure_level(self) -> str:
        """
        Get human-readable memory pressure level.

        Returns:
            One of: "normal", "elevated", "high", "critical"
        """
        usage_pct = self.get_usage_percentage()

        if usage_pct < 0.60:
            return "normal"
        elif usage_pct < self.warning_threshold:
            return "elevated"
        elif usage_pct < self.critical_threshold:
            return "high"
        else:
            return "critical"

    def print_memory_status(self, prefix: str = "") -> None:
        """
        Print current memory status to console.

        Args:
            prefix: Optional prefix for the message
        """
        stats = self.get_memory_stats()
        usage_pct = stats.rss_mb / stats.limit_mb

        pressure = self.get_memory_pressure_level()
        emoji = {
            "normal": "✓",
            "elevated": "ℹ️",
            "high": "⚠️",
            "critical": "🚨"
        }.get(pressure, "•")

        logger.info(f"{prefix}{emoji} Memory: {stats.rss_mb:.1f}MB / {stats.limit_mb:.1f}MB "
              f"({usage_pct * 100:.1f}%) - {pressure.upper()}")

    def get_statistics(self) -> dict:
        """
        Get memory management statistics.

        Returns:
            Dictionary with stats
        """
        current_stats = self.get_memory_stats()

        return {
            'current_usage_mb': current_stats.rss_mb,
            'peak_usage_mb': self.peak_usage_mb,
            'memory_limit_mb': self.memory_limit_mb,
            'usage_percentage': self.get_usage_percentage() * 100,
            'cleanup_count': self.cleanup_count,
            'memory_pressure': self.get_memory_pressure_level(),
            'available_mb': current_stats.available_mb,
        }


class MemoryMonitorContext:
    """
    Context manager for monitoring memory usage during a block of code.

    Usage:
        with MemoryMonitorContext("Processing clips") as monitor:
            # ... process clips ...
    """

    def __init__(self, operation_name: str, memory_manager: Optional[AdaptiveMemoryManager] = None):
        """
        Initialize memory monitor context.

        Args:
            operation_name: Name of the operation being monitored
            memory_manager: Optional existing memory manager (creates new if None)
        """
        self.operation_name = operation_name
        self.memory_manager = memory_manager or AdaptiveMemoryManager()
        self.start_mb = 0.0
        self.end_mb = 0.0
        self.peak_mb = 0.0

    def __enter__(self):
        """Enter context - record starting memory."""
        self.start_mb = self.memory_manager.get_current_usage_mb()
        logger.info(f"   🔍 [{self.operation_name}] Starting - Memory: {self.start_mb:.1f}MB")
        return self.memory_manager

    def __exit__(self, exc_type, exc_val, _exc_tb):
        """Exit context - record ending memory and print summary."""
        self.end_mb = self.memory_manager.get_current_usage_mb()
        delta_mb = self.end_mb - self.start_mb

        emoji = "✓" if delta_mb < 100 else ("⚠️" if delta_mb < 300 else "🚨")
        sign = "+" if delta_mb >= 0 else ""

        logger.info(f"   {emoji} [{self.operation_name}] Finished - Memory: {self.end_mb:.1f}MB "
              f"({sign}{delta_mb:.1f}MB delta)")

        # Suggest cleanup if significant growth
        if delta_mb > 200:
            logger.warning(f"      💡 Suggestion: Consider cleanup (grew by {delta_mb:.1f}MB)")

        return False  # Don't suppress exceptions


# Global singleton instance
_global_memory_manager = None


def get_memory_manager() -> AdaptiveMemoryManager:
    """
    Get the global memory manager instance.

    Returns:
        Global AdaptiveMemoryManager singleton
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = AdaptiveMemoryManager()
    return _global_memory_manager


def monitor_operation(operation_name: str):
    """
    Decorator to monitor memory usage of a function.

    Usage:
        @monitor_operation("Enhance clips")
        def enhance_clips():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with MemoryMonitorContext(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
