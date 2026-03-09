"""Compatibility wrapper for the legacy `config.py` public API.

This package path (`montage_ai.config`) currently shadows the historical
`src/montage_ai/config.py` module that most of the codebase depends on.
To preserve runtime and test compatibility, we load the legacy module from
disk and forward attribute access to it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_legacy_config_module() -> ModuleType:
    """Load legacy `config.py` as an internal module exactly once."""
    module_name = "src.montage_ai._legacy_config"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    legacy_path = Path(__file__).resolve().parent.parent / "config.py"
    spec = importlib.util.spec_from_file_location(module_name, legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy config module from {legacy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_config_module()

# Frequently used symbols are bound explicitly for better editor/type support.
Settings = _legacy.Settings
PathConfig = _legacy.PathConfig
FeatureConfig = _legacy.FeatureConfig
LLMConfig = _legacy.LLMConfig
FileTypeConfig = _legacy.FileTypeConfig
get_settings = _legacy.get_settings
reload_settings = _legacy.reload_settings
get_effective_cpu_count = _legacy.get_effective_cpu_count


def __getattr__(name: str) -> Any:
    """Delegate unresolved attributes to the legacy config module."""
    try:
        return getattr(_legacy, name)
    except AttributeError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc


def __dir__() -> list[str]:
    """Expose delegated attributes for introspection and completion."""
    return sorted(set(globals().keys()) | set(dir(_legacy)))


__all__ = [name for name in dir(_legacy) if not name.startswith("_")]
