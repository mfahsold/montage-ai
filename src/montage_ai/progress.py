"""
Unified Progress Tracking System

Provides a consistent interface for progress updates across CLI, Web UI, and API.
Follows the Observer pattern with typed updates for type safety.

Usage:
    from montage_ai.progress import CLIProgress, ProgressUpdate

    callback = CLIProgress()
    callback(ProgressUpdate(percent=50, message="Processing...", current_item="video.mp4"))
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Any, Dict
import sys


@dataclass
class ProgressUpdate:
    """
    Typed progress update payload.

    All progress callbacks should accept this dataclass or a dict with these keys.
    This is the Single Source of Truth for progress data structure.
    """
    percent: int = 0
    message: str = ""
    current_item: Optional[str] = None  # e.g., "video_001.mp4", "clip_12"
    cpu_percent: Optional[float] = None  # Process CPU usage %
    memory_mb: Optional[float] = None    # Process memory (RSS) in MB
    gpu_util: Optional[str] = None       # GPU utilization string
    memory_pressure: Optional[str] = None  # "normal", "elevated", "high", "critical"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization or callback passing."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgressUpdate":
        """Create from dict (handles both old and new callback formats)."""
        return cls(
            percent=data.get("percent", 0),
            message=data.get("message", ""),
            current_item=data.get("current_item"),
            cpu_percent=data.get("cpu_percent"),
            memory_mb=data.get("memory_mb"),
            gpu_util=data.get("gpu_util"),
            memory_pressure=data.get("memory_pressure"),
        )


class ProgressCallback(ABC):
    """
    Abstract base class for progress callbacks.

    Implementations handle progress updates for different outputs:
    - CLIProgress: Terminal output with progress bar
    - WebProgress: SSE/WebSocket broadcast
    - LogProgress: File/logger output
    - NullProgress: No-op for testing
    """

    @abstractmethod
    def __call__(self, update: ProgressUpdate) -> None:
        """Handle a progress update."""
        pass

    def update(self, percent: int, message: str = "", **kwargs) -> None:
        """Convenience method for simple updates."""
        self(ProgressUpdate(percent=percent, message=message, **kwargs))


class CLIProgress(ProgressCallback):
    """
    CLI progress callback with live terminal output.

    Features:
    - Progress bar with percentage
    - Current item display
    - Resource metrics (CPU, RAM, GPU)
    - Memory pressure indicator
    """

    def __init__(self, bar_width: int = 20, line_width: int = 100):
        self.bar_width = bar_width
        self.line_width = line_width
        self._last_percent = -1

    def __call__(self, update: ProgressUpdate) -> None:
        """Render progress to stdout."""
        percent = update.percent

        # Build progress bar
        filled = int(self.bar_width * percent / 100)
        bar = "\u2588" * filled + "\u2591" * (self.bar_width - filled)

        # Build resource string
        resource_parts = []
        if update.cpu_percent is not None:
            resource_parts.append(f"CPU:{update.cpu_percent:.0f}%")
        if update.memory_mb is not None:
            resource_parts.append(f"RAM:{update.memory_mb:.0f}MB")
        if update.gpu_util:
            # Truncate GPU string if too long
            gpu_str = update.gpu_util[:20] if len(update.gpu_util) > 20 else update.gpu_util
            resource_parts.append(f"GPU:{gpu_str}")
        resource_str = " ".join(resource_parts)

        # Build output line
        item_str = f" [{update.current_item}]" if update.current_item else ""
        resource_suffix = f" | {resource_str}" if resource_str else ""

        line = f"\r   [{bar}] {percent:3d}%{item_str}{resource_suffix}"
        sys.stdout.write(line.ljust(self.line_width)[:self.line_width])
        sys.stdout.flush()

        # Newline when complete
        if percent >= 100 and self._last_percent < 100:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self._last_percent = percent


class LogProgress(ProgressCallback):
    """
    Logger-based progress callback.

    Logs progress updates at configurable intervals to avoid spam.
    """

    def __init__(self, logger=None, log_interval: int = 10):
        """
        Args:
            logger: Logger instance (uses module logger if None)
            log_interval: Only log every N percent change
        """
        self._logger = logger
        self._log_interval = log_interval
        self._last_logged = -log_interval

    def __call__(self, update: ProgressUpdate) -> None:
        """Log progress if interval threshold reached."""
        if update.percent - self._last_logged >= self._log_interval or update.percent >= 100:
            logger = self._logger
            if logger is None:
                from .logger import logger as default_logger
                logger = default_logger

            # Build log message
            parts = [f"{update.percent}%"]
            if update.message:
                parts.append(update.message)
            if update.current_item:
                parts.append(f"[{update.current_item}]")
            if update.cpu_percent is not None:
                parts.append(f"CPU:{update.cpu_percent:.0f}%")
            if update.memory_mb is not None:
                parts.append(f"RAM:{update.memory_mb:.0f}MB")

            logger.info(f"   📊 Progress: {' | '.join(parts)}")
            self._last_logged = update.percent


class NullProgress(ProgressCallback):
    """No-op progress callback for testing or silent operation."""

    def __call__(self, update: ProgressUpdate) -> None:
        """Do nothing."""
        pass


class CompositeProgress(ProgressCallback):
    """
    Broadcasts progress to multiple callbacks.

    Example:
        composite = CompositeProgress([CLIProgress(), LogProgress()])
        composite(update)  # Sends to both CLI and log
    """

    def __init__(self, callbacks: list):
        self._callbacks = callbacks

    def __call__(self, update: ProgressUpdate) -> None:
        """Broadcast to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(update)
            except Exception:
                pass  # Don't let one callback failure break others


# =============================================================================
# Legacy Compatibility Wrapper
# =============================================================================

def normalize_callback(callback: Callable) -> ProgressCallback:
    """
    Wrap legacy callbacks (int, str) -> None to use ProgressUpdate.

    This allows gradual migration from old signature to new.
    """
    import inspect

    if isinstance(callback, ProgressCallback):
        return callback

    # Check if it's old-style (2 positional args)
    try:
        sig = inspect.signature(callback)
        params = list(sig.parameters.values())
        # Old style: (percent: int, message: str)
        if len(params) >= 2 and params[0].annotation in (int, inspect.Parameter.empty):

            class LegacyWrapper(ProgressCallback):
                def __init__(self, fn):
                    self._fn = fn

                def __call__(self, update: ProgressUpdate) -> None:
                    self._fn(update.percent, update.message)

            return LegacyWrapper(callback)
    except (ValueError, TypeError):
        pass

    # Assume it accepts dict or ProgressUpdate
    class DictWrapper(ProgressCallback):
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, update: ProgressUpdate) -> None:
            self._fn(update.to_dict())

    return DictWrapper(callback)


# =============================================================================
# Factory Functions
# =============================================================================

def create_progress_callback(
    mode: str = "cli",
    logger=None,
    callbacks: list = None,
) -> ProgressCallback:
    """
    Factory function to create appropriate progress callback.

    Args:
        mode: "cli", "log", "null", "composite"
        logger: Logger instance for log mode
        callbacks: List of callbacks for composite mode

    Returns:
        ProgressCallback instance
    """
    if mode == "cli":
        return CLIProgress()
    elif mode == "log":
        return LogProgress(logger=logger)
    elif mode == "null":
        return NullProgress()
    elif mode == "composite":
        return CompositeProgress(callbacks or [])
    else:
        raise ValueError(f"Unknown progress mode: {mode}")
