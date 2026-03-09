"""
LLM Configuration Module

Language model backends and settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from . import BaseConfig, _env_override


@dataclass
class LLMConfig(BaseConfig):
    """LLM backend configuration."""

    # Backend selection (priority order)
    use_openai_api: bool = field(
        default_factory=lambda: _env_override("llm_use_openai", False)
    )
    use_google_ai: bool = field(
        default_factory=lambda: _env_override("llm_use_google", False)
    )
    use_cgpu: bool = field(default_factory=lambda: _env_override("llm_use_cgpu", False))
    use_ollama: bool = field(
        default_factory=lambda: _env_override("llm_use_ollama", True)
    )

    # OpenAI-compatible API
    openai_api_base: Optional[str] = field(
        default_factory=lambda: _env_override("openai_api_base", None)
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: _env_override("openai_api_key", None)
    )
    openai_model: str = field(
        default_factory=lambda: _env_override("openai_model", "gpt-4")
    )

    # Google AI
    google_api_key: Optional[str] = field(
        default_factory=lambda: _env_override("google_api_key", None)
    )
    google_model: str = field(
        default_factory=lambda: _env_override("google_model", "gemini-2.0-flash")
    )

    # Ollama
    ollama_host: str = field(
        default_factory=lambda: _env_override(
            "ollama_host", "http://host.docker.internal:11434"
        )
    )
    ollama_model: str = field(
        default_factory=lambda: _env_override("ollama_model", "llama3.1:70b")
    )

    # CGPU
    cgpu_endpoint: Optional[str] = field(
        default_factory=lambda: _env_override("cgpu_endpoint", None)
    )
    cgpu_enabled: bool = field(
        default_factory=lambda: _env_override("cgpu_enabled", False)
    )
    cgpu_gpu_enabled: bool = field(
        default_factory=lambda: _env_override("cgpu_gpu_enabled", False)
    )
    cgpu_timeout: int = field(
        default_factory=lambda: _env_override("cgpu_timeout", 300)
    )
    cgpu_host: str = field(
        default_factory=lambda: _env_override("cgpu_host", "localhost")
    )
    cgpu_port: int = field(default_factory=lambda: _env_override("cgpu_port", 8000))
    cgpu_model: str = field(
        default_factory=lambda: _env_override("cgpu_model", "gemini-2.0-flash")
    )
    cgpu_max_concurrency: int = field(
        default_factory=lambda: _env_override("cgpu_max_concurrency", 4)
    )

    # Common settings
    timeout: int = field(default_factory=lambda: _env_override("llm_timeout", 60))
    max_retries: int = field(
        default_factory=lambda: _env_override("llm_max_retries", 3)
    )
    temperature: float = field(
        default_factory=lambda: _env_override("llm_temperature", 0.7)
    )

    def validate(self) -> list:
        """Validate LLM configuration."""
        errors = []

        # Check at least one backend is enabled
        if not any(
            [self.use_openai_api, self.use_google_ai, self.use_cgpu, self.use_ollama]
        ):
            errors.append("At least one LLM backend must be enabled")

        # Validate backend-specific settings
        if self.use_openai_api and not self.openai_api_base:
            errors.append("OPENAI_API_BASE required when use_openai_api is enabled")

        if self.use_google_ai and not self.google_api_key:
            errors.append("GOOGLE_API_KEY required when use_google_ai is enabled")

        return errors
