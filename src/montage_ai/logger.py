"""
Centralized Logging for Montage AI

Nightfly Studio - Montage AI Post-Production Assistant

This module provides a configured logger with:
- Console output with emoji support for user-facing messages
- File logging for debugging
- Configurable log levels via LOG_LEVEL environment variable

Usage:
    from montage_ai.logger import logger

    logger.info("Processing video...")
    logger.debug("Detailed debug info")
    logger.warning("Something unexpected")
    logger.error("Something failed")

    # For user-facing output with emojis:
    logger.info("   ✅ Video processed successfully")
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# =============================================================================
# Branding
# =============================================================================
STUDIO_NAME = "Nightfly Studio"
APP_NAME = "Montage AI"
APP_VERSION = "1.0.0"


# =============================================================================
# Log Level Configuration
# =============================================================================
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def get_log_level() -> int:
    """Get log level from environment variable."""
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    return LOG_LEVEL_MAP.get(level_str, logging.INFO)


# =============================================================================
# Custom Formatter
# =============================================================================
class MontageFormatter(logging.Formatter):
    """
    Custom formatter that preserves emoji output and adds context.

    For user-facing messages (INFO level), output is clean.
    For debug/warning/error, includes timestamp and level.
    """

    FORMATS = {
        logging.DEBUG: "%(asctime)s [DEBUG] %(name)s: %(message)s",
        logging.INFO: "%(message)s",  # Clean output for user-facing
        logging.WARNING: "%(asctime)s [WARN] %(message)s",
        logging.ERROR: "%(asctime)s [ERROR] %(message)s",
        logging.CRITICAL: "%(asctime)s [CRITICAL] %(message)s",
    }

    def __init__(self):
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


class FileFormatter(logging.Formatter):
    """Detailed formatter for file logging."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


# =============================================================================
# Logger Setup
# =============================================================================
def setup_logger(
    name: str = "montage_ai",
    log_file: Optional[Path] = None,
    level: Optional[int] = None
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (default: montage_ai)
        log_file: Optional path to log file
        level: Log level (default: from LOG_LEVEL env var)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    log_level = level or get_log_level()
    logger.setLevel(log_level)

    # Console handler with emoji-friendly formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(MontageFormatter())
    logger.addHandler(console_handler)

    # File handler if path provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "montage_ai") -> logging.Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (default: montage_ai)

    Returns:
        Logger instance
    """
    return setup_logger(name)


def configure_file_logging(output_dir: Path, job_id: str) -> None:
    """
    Enable file logging for a specific job.

    Args:
        output_dir: Output directory for log file
        job_id: Job identifier for log filename
    """
    log_file = output_dir / f"render_{job_id}.log"

    # Get the root logger and add file handler
    root_logger = logging.getLogger("montage_ai")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FileFormatter())
    root_logger.addHandler(file_handler)


# =============================================================================
# Global Logger Instance
# =============================================================================
logger = setup_logger()


# =============================================================================
# Convenience Functions
# =============================================================================
def print_banner() -> None:
    """Print application startup banner."""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║              {STUDIO_NAME} presents                          ║
║                                                              ║
║      ███╗   ███╗ ██████╗ ███╗   ██╗████████╗ █████╗  ██████╗ ███████╗    ║
║      ████╗ ████║██╔═══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝    ║
║      ██╔████╔██║██║   ██║██╔██╗ ██║   ██║   ███████║██║  ███╗█████╗      ║
║      ██║╚██╔╝██║██║   ██║██║╚██╗██║   ██║   ██╔══██║██║   ██║██╔══╝      ║
║      ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║   ██║   ██║  ██║╚██████╔╝███████╗    ║
║      ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ║
║                                                              ║
║                    AI Post-Production Assistant              ║
║                        v{APP_VERSION}                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    logger.info(banner)


def log_phase(phase: str) -> None:
    """Log a major phase transition with visual separator."""
    separator = "═" * 60
    logger.info(f"\n{separator}")
    logger.info(f"  {phase}")
    logger.info(f"{separator}\n")


def log_step(step: str, emoji: str = "▶") -> None:
    """Log a processing step."""
    logger.info(f"{emoji} {step}")


def log_success(message: str) -> None:
    """Log a success message with checkmark."""
    logger.info(f"   ✅ {message}")


def log_error(message: str) -> None:
    """Log an error message with X mark."""
    logger.error(f"   ❌ {message}")


def log_warning(message: str) -> None:
    """Log a warning message."""
    logger.warning(f"   ⚠️  {message}")


def log_debug(message: str) -> None:
    """Log debug information."""
    logger.debug(message)
