from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .logger import logger


def normalize_extensions(extensions: Iterable[str]) -> Tuple[str, ...]:
    """Normalize extension strings to lowercase dot-prefixed suffixes."""
    normalized: List[str] = []
    for ext in extensions:
        if not ext:
            continue
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return tuple(sorted(set(normalized)))


def list_media_files(directory: Path, extensions: Iterable[str], recursive: bool = False) -> List[Path]:
    """List files in a directory that match the provided extensions."""
    if not directory.exists():
        return []

    suffixes = normalize_extensions(extensions)
    if not suffixes:
        return []

    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files: List[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def parse_inventory_descriptions(inventory_path: Path) -> Dict[str, str]:
    """Parse FOOTAGE_INVENTORY.md into filename -> description mapping."""
    if not inventory_path.exists():
        return {}

    try:
        text = inventory_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("Failed to read inventory file %s: %s", inventory_path, exc)
        return {}

    pattern = re.compile(r"^###\s+\d+\.\s+\*\*([^*]+)\*\*", re.IGNORECASE)
    descriptions: Dict[str, str] = {}
    current: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal current, buffer
        if current and buffer:
            descriptions[current] = "\n".join(buffer).strip()
        current = None
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = pattern.match(line)
        if match:
            flush()
            current = match.group(1).strip()
            continue
        if current:
            if line.startswith("###"):
                flush()
                continue
            if line.startswith("- "):
                buffer.append(line[2:].strip())
            elif line:
                buffer.append(line)

    flush()
    return descriptions


def read_sidecar_description(file_path: Path, max_chars: int = 4000) -> Optional[str]:
    """Read a sidecar .md description for a media file."""
    sidecar = file_path.with_suffix(".md")
    if not sidecar.exists():
        return None

    try:
        text = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError as exc:
        logger.warning("Failed to read description file %s: %s", sidecar, exc)
        return None

    if not text:
        return None
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


__all__ = [
    "normalize_extensions",
    "list_media_files",
    "parse_inventory_descriptions",
    "read_sidecar_description",
]
