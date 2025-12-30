"""
Style template loader (config-first, no hardcoded presets).

Features
- Ships with JSON presets in `montage_ai/styles/*.json`.
- Overrides via env vars: `STYLE_PRESET_PATH` (file) or
  `STYLE_PRESET_DIR` / `STYLE_TEMPLATES_DIR` (directory of *.json).
- Later files override earlier ones (defaults < user overrides).
- Lightweight validation to catch malformed presets.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

from jsonschema import ValidationError, validate

from .config import get_settings

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

DEFAULT_STYLE_DIR = Path(__file__).resolve().parent / "styles"


# ---------------------------------------------------------------------------
# Validation schema (minimal on purpose)
# ---------------------------------------------------------------------------

STYLE_SCHEMA = {
    "type": "object",
    "required": ["id", "name", "description", "params"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "params": {"type": "object"},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_files() -> List[Path]:
    """Gather style preset files with precedence: defaults then overrides."""
    settings = get_settings()
    candidates: List[Path] = []

    if DEFAULT_STYLE_DIR.exists():
        candidates.extend(sorted(DEFAULT_STYLE_DIR.glob("*.json")))

    if settings.paths.style_preset_dir:
        env_dir = settings.paths.style_preset_dir
        if env_dir.is_dir():
            candidates.extend(sorted(env_dir.glob("*.json")))
        elif env_dir.is_file() and env_dir.suffix.lower() == ".json":
            candidates.append(env_dir)

    if settings.paths.style_preset_path:
        env_file = settings.paths.style_preset_path
        if env_file.is_file() and env_file.suffix.lower() == ".json":
            candidates.append(env_file)

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: List[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _extract_templates(raw: object, source: Path) -> Iterable[dict]:
    """Normalize raw JSON content into an iterable of template dicts."""

    if isinstance(raw, dict) and "params" in raw:
        yield raw
        return

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                value.setdefault("id", key)
                yield value
        return

    raise TypeError(f"Style file {source} must contain an object or array of objects")


def _normalize_template(raw: dict, source: Path) -> dict:
    """Ensure required keys exist and align identifiers."""

    template_id = (
        raw.get("id")
        or raw.get("key")
        or raw.get("slug")
        or raw.get("style", {}).get("name")
        or raw.get("params", {}).get("style", {}).get("name")
        or source.stem
    )

    params = raw.get("params", {}) or {}
    style_block = params.get("style", {}) or {}
    style_block.setdefault("name", str(template_id))
    params["style"] = style_block

    normalized = {
        "id": str(template_id).lower(),
        "name": raw.get("name") or style_block.get("label") or str(template_id),
        "description": raw.get("description", ""),
        "params": params,
    }

    validate(instance=normalized, schema=STYLE_SCHEMA)
    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_style_templates() -> Dict[str, dict]:
    """Load and cache style templates from JSON files."""

    templates: Dict[str, dict] = {}
    files = _style_files()

    if not files:
        raise RuntimeError(
            "No style preset files found. Set STYLE_PRESET_DIR/STYLE_PRESET_PATH or keep defaults."
        )

    for path in files:
        try:
            raw_content = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Skipping style file {path}: {exc}")
            continue

        try:
            candidate_templates = list(_extract_templates(raw_content, path))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Skipping style file {path}: {exc}")
            continue

        for entry in candidate_templates:
            try:
                normalized = _normalize_template(entry, path)
            except ValidationError as exc:
                print(f"⚠️  Invalid style in {path}: {exc.message}")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Skipping style in {path}: {exc}")
                continue

            templates[normalized["id"]] = normalized  # later files override earlier ones

    if not templates:
        raise RuntimeError("Failed to load any style templates; check preset files")

    return templates


def reload_style_templates() -> None:
    """Clear cache so new/changed presets are picked up."""

    load_style_templates.cache_clear()


def get_style_template(style_name: str) -> dict:
    """Get a style template by id (case-insensitive)."""

    templates = load_style_templates()
    key = style_name.lower()
    if key not in templates:
        available = ", ".join(sorted(templates.keys()))
        raise KeyError(f"Unknown style '{style_name}'. Available: {available}")
    return templates[key]


def list_available_styles() -> List[str]:
    """List available style ids (sorted)."""

    return sorted(load_style_templates().keys())


def get_style_description(style_name: str) -> str:
    """Get the human-friendly description for a style."""

    return get_style_template(style_name)["description"]

