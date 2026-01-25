from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

try:
    from werkzeug.utils import secure_filename as _secure_filename
except Exception:
    _secure_filename = None

from .media_files import normalize_extensions


def safe_filename(filename: str) -> str:
    """Return a sanitized filename that is safe for local filesystem writes."""
    raw = (filename or "").strip()
    if not raw:
        return ""
    if _secure_filename:
        return _secure_filename(raw)
    name = Path(raw).name
    return name.replace("\x00", "")


def build_safe_path(base_dir: Path, filename: str) -> Optional[Path]:
    """Return a safe, resolved path inside base_dir or None if invalid."""
    safe_name = safe_filename(filename)
    if not safe_name:
        return None
    base_dir = base_dir.resolve()
    candidate = (base_dir / safe_name).resolve()
    if not str(candidate).startswith(str(base_dir)):
        return None
    return candidate


def format_extensions(extensions: Iterable[str]) -> str:
    """Format extensions for error messaging (dot-prefixed, sorted)."""
    normalized = normalize_extensions(extensions)
    return ", ".join(normalized)
