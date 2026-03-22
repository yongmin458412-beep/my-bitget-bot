"""General-purpose helpers shared by multiple modules."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar


T = TypeVar("T")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs."""

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path, default: Any | None = None) -> Any:
    """Read JSON from disk, returning a default value when missing."""

    file_path = Path(path)
    if not file_path.exists():
        return {} if default is None else default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: Any) -> None:
    """Write JSON to disk with stable formatting."""

    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)


def hash_text(text: str) -> str:
    """Create a stable SHA-256 hash for deduplication."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, limit: int = 3500) -> list[str]:
    """Split long text into chunks suitable for Telegram messages."""

    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = []
    current_size = 0
    for line in text.splitlines(keepends=True):
        if current_size + len(line) > limit and current:
            chunks.append("".join(current))
            current = [line]
            current_size = len(line)
        else:
            current.append(line)
            current_size += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def round_to_step(value: float, step: float, mode: str = "nearest") -> float:
    """Round a numeric value using a symbol-specific step size."""

    if step <= 0:
        return value

    factor = value / step
    if mode == "down":
        rounded = math.floor(factor) * step
    elif mode == "up":
        rounded = math.ceil(factor) * step
    else:
        rounded = round(factor) * step
    decimals = max(0, len(f"{step:.16f}".rstrip("0").split(".")[-1]))
    return round(rounded, decimals)


def as_float(value: Any, default: float = 0.0) -> float:
    """Safely cast a value to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    """Safely cast a value to int."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_client_order_id(prefix: str, *parts: str) -> str:
    """Create a deterministic yet collision-resistant client order id."""

    clean_prefix = prefix[:8].lower()
    normalized = "-".join(part.replace("/", "_").replace(" ", "_") for part in parts if part)
    suffix = uuid.uuid4().hex[:10]
    return f"{clean_prefix}-{normalized[:24]}-{suffix}"[:64]


def chunks(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield slices of a sequence."""

    if size <= 0:
        raise ValueError("size must be positive")
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """Return unique items while preserving their original order."""

    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output
