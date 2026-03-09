"""
Proxy Configuration Module

Proxy generation and caching settings.
"""

from dataclasses import dataclass, field

from . import BaseConfig, _env_override


@dataclass
class ProxyConfig(BaseConfig):
    """Proxy generation configuration."""

    # Caching
    cache_proxies: bool = field(
        default_factory=lambda: _env_override("proxy_cache_enabled", True)
    )
    proxy_cache_ttl_seconds: int = field(
        default_factory=lambda: _env_override("proxy_cache_ttl", 86400)
    )
    proxy_lock_timeout_seconds: int = field(
        default_factory=lambda: _env_override("proxy_lock_timeout", 30)
    )

    # Generation settings
    proxy_resolution: str = field(
        default_factory=lambda: _env_override("proxy_resolution", "480p")
    )
    proxy_codec: str = field(
        default_factory=lambda: _env_override("proxy_codec", "h264")
    )
    proxy_preset: str = field(
        default_factory=lambda: _env_override("proxy_preset", "fast")
    )
