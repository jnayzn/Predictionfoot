from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any


def apply_changes(obj: object, values: Mapping[str, Any]) -> list[str]:
    """Set attributes that differ; return the names of changed fields."""
    changed: list[str] = []
    for name, value in values.items():
        if getattr(obj, name) != value:
            setattr(obj, name, value)
            changed.append(name)
    return changed


def chunks[T](items: Sequence[T], size: int = 500) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
