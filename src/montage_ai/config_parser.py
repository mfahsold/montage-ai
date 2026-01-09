"""
ConfigParser - Unified Environment Variable Parsing for Montage AI

This module provides centralized parsing utilities for environment variables.
Following DRY principles to avoid repeated lambda patterns in dataclass fields.

Usage:
    from montage_ai.config_parser import ConfigParser
    
    # In dataclass fields:
    my_int: int = field(default_factory=ConfigParser.make_int_parser("MY_VAR", 42))
    my_bool: bool = field(default_factory=ConfigParser.make_bool_parser("MY_FLAG", False))
    my_float: float = field(default_factory=ConfigParser.make_float_parser("MY_RATE", 1.5))
    my_path: Path = field(default_factory=ConfigParser.make_path_parser("MY_PATH", "/data/input"))
    
Example:
    >>> int_value = ConfigParser.parse_int("SCENE_MIN_LENGTH", 15)
    >>> bool_value = ConfigParser.parse_bool("STABILIZE", False)
    >>> float_value = ConfigParser.parse_float("THRESHOLD", 30.0)
"""

import os
from pathlib import Path
from typing import Any, Callable, Optional, Union


class ConfigParser:
    """Unified environment variable parsing utilities."""
    
    # =========================================================================
    # Direct Parsing Functions (for immediate value access)
    # =========================================================================
    
    @staticmethod
    def parse_int(key: str, default: int) -> int:
        """Parse an integer from environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not set or invalid
            
        Returns:
            Parsed integer value
        """
        try:
            return int(os.environ.get(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def parse_float(key: str, default: float) -> float:
        """Parse a float from environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not set or invalid
            
        Returns:
            Parsed float value
        """
        try:
            return float(os.environ.get(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def parse_bool(key: str, default: bool) -> bool:
        """Parse a boolean from environment variable.
        
        Treats 'true', 'True', 'TRUE', '1', 'yes' as True.
        All other values are False.
        
        Args:
            key: Environment variable name
            default: Default value if not set
            
        Returns:
            Parsed boolean value
        """
        value = os.environ.get(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        else:
            return default
    
    @staticmethod
    def parse_str(key: str, default: str) -> str:
        """Parse a string from environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not set
            
        Returns:
            String value
        """
        return os.environ.get(key, default)
    
    @staticmethod
    def parse_path(key: str, default: Union[str, Path]) -> Path:
        """Parse a Path from environment variable.
        
        Args:
            key: Environment variable name
            default: Default path if not set
            
        Returns:
            Path object
        """
        env_value = os.environ.get(key)
        if env_value:
            return Path(env_value)
        return Path(default)
    
    # =========================================================================
    # Factory Functions (for use with dataclass field default_factory)
    # =========================================================================
    # These return lambdas suitable for use with @dataclass field(default_factory=...)
    
    @staticmethod
    def make_int_parser(key: str, default: int) -> Callable[[], int]:
        """Create a parser function for integer config values.
        
        Usage in dataclass:
            my_field: int = field(default_factory=ConfigParser.make_int_parser("MY_VAR", 42))
        """
        return lambda: ConfigParser.parse_int(key, default)
    
    @staticmethod
    def make_float_parser(key: str, default: float) -> Callable[[], float]:
        """Create a parser function for float config values.
        
        Usage in dataclass:
            my_field: float = field(default_factory=ConfigParser.make_float_parser("MY_VAR", 1.5))
        """
        return lambda: ConfigParser.parse_float(key, default)
    
    @staticmethod
    def make_bool_parser(key: str, default: bool) -> Callable[[], bool]:
        """Create a parser function for boolean config values.
        
        Usage in dataclass:
            my_field: bool = field(default_factory=ConfigParser.make_bool_parser("MY_FLAG", False))
        """
        return lambda: ConfigParser.parse_bool(key, default)
    
    @staticmethod
    def make_str_parser(key: str, default: str) -> Callable[[], str]:
        """Create a parser function for string config values.
        
        Usage in dataclass:
            my_field: str = field(default_factory=ConfigParser.make_str_parser("MY_VAR", "default"))
        """
        return lambda: ConfigParser.parse_str(key, default)
    
    @staticmethod
    def make_path_parser(key: str, default: Union[str, Path]) -> Callable[[], Path]:
        """Create a parser function for path config values.
        
        Usage in dataclass:
            my_field: Path = field(default_factory=ConfigParser.make_path_parser("MY_PATH", "/data/input"))
        """
        return lambda: ConfigParser.parse_path(key, default)
    
    # =========================================================================
    # Batch Parsing (for advanced use cases)
    # =========================================================================
    
    @staticmethod
    def parse_dict(key: str, default: Optional[dict] = None) -> dict:
        """Parse a JSON dict from environment variable.
        
        Args:
            key: Environment variable name
            default: Default dict if not set or invalid
            
        Returns:
            Parsed dict
        """
        import json
        
        value = os.environ.get(key, "")
        if not value:
            return default or {}
        
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default or {}


__all__ = [
    "ConfigParser",
]
