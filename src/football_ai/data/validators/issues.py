"""Data-quality issue representation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    WARNING = "warning"  # kept, but flagged
    ERROR = "error"  # record rejected from normalized output (raw JSON is still archived)


@dataclass(frozen=True, slots=True)
class DataQualityIssue:
    entity: str
    entity_id: str
    field: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.entity}#{self.entity_id}.{self.field}: {self.message}"


def has_errors(issues: list[DataQualityIssue]) -> bool:
    return any(i.severity is Severity.ERROR for i in issues)
