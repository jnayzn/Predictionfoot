"""Versioned feature-set storage (Parquet + metadata JSON) for reproducibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from football_ai.core.io import atomic_write_json, read_json
from football_ai.features.builder import (
    FEATURE_SET_VERSION,
    FeatureConfig,
    feature_columns,
    feature_group,
)


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Hash of the targets/metadata: changes whenever the underlying data changes."""
    cols = ["match_id", "kickoff_utc", "status_group", "home_goals", "away_goals"]
    payload = pd.util.hash_pandas_object(frame[cols], index=False).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def save_feature_set(frame: pd.DataFrame, config: FeatureConfig, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"features_{config.fingerprint()}_{data_fingerprint(frame)}"
    path = directory / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    cols = feature_columns(frame)
    atomic_write_json(
        directory / f"{name}.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "feature_set_version": FEATURE_SET_VERSION,
            "config": asdict(config),
            "config_fingerprint": config.fingerprint(),
            "data_fingerprint": data_fingerprint(frame),
            "rows": len(frame),
            "played_rows": int(frame["outcome"].notna().sum()),
            "features": {c: feature_group(c) for c in cols},
            "missing_rate": {c: round(float(frame[c].isna().mean()), 4) for c in cols},
        },
    )
    (directory / "latest.txt").write_text(path.name, encoding="utf-8")
    return path


def load_feature_set(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(path)
    meta = read_json(path.with_suffix(".json"))
    return frame, meta


def latest_feature_set(directory: Path) -> Path | None:
    marker = directory / "latest.txt"
    if not marker.is_file():
        return None
    path = directory / marker.read_text(encoding="utf-8").strip()
    return path if path.is_file() else None


def describe(frame: pd.DataFrame) -> dict[str, Any]:
    cols = feature_columns(frame)
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(feature_group(c), []).append(c)
    return {
        "rows": len(frame),
        "played": int(frame["outcome"].notna().sum()),
        "features": len(cols),
        "groups": {
            g: {"count": len(c), "missing_rate": round(float(frame[c].isna().mean().mean()), 3)}
            for g, c in groups.items()
        },
    }


def to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, default=str)
