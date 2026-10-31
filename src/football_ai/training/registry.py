"""Model registry: versioned artifacts on disk + ``model_versions`` rows."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.core.io import atomic_write_json
from football_ai.database import models as db


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    name: str
    version: str
    path: Path


def make_version(params: dict[str, Any], data_fingerprint: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha256(
        json.dumps(
            {"params": params, "data": data_fingerprint}, sort_keys=True, default=str
        ).encode()
    ).hexdigest()[:8]
    # random suffix: two trainings in the same second must never share a version
    return f"{stamp}-{digest}-{uuid.uuid4().hex[:6]}"


def save_model(
    model: Any,
    *,
    name: str,
    algorithm: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    data_fingerprint: str,
    training_start: datetime | None,
    training_end: datetime | None,
    root: Path,
    session: Session | None = None,
    mlflow_run_id: str | None = None,
) -> RegisteredModel:
    version = make_version(params, data_fingerprint)
    directory = root / name / version
    directory.mkdir(parents=True, exist_ok=False)  # never overwrite an artifact
    artifact = directory / "model.joblib"
    joblib.dump(model, artifact)
    atomic_write_json(
        directory / "metadata.json",
        {
            "name": name,
            "version": version,
            "algorithm": algorithm,
            "params": params,
            "metrics": metrics,
            "data_fingerprint": data_fingerprint,
            "training_start": training_start.isoformat() if training_start else None,
            "training_end": training_end.isoformat() if training_end else None,
            "mlflow_run_id": mlflow_run_id,
        },
    )
    if session is not None:
        session.add(
            db.ModelVersion(
                name=name,
                version=version,
                algorithm=algorithm,
                params=json.loads(json.dumps(params, default=str)),
                metrics=json.loads(json.dumps(metrics, default=str)),
                training_data_start=training_start,
                training_data_end=training_end,
                artifact_uri=str(artifact),
                mlflow_run_id=mlflow_run_id,
            )
        )
    return RegisteredModel(name, version, artifact)


def load_model(path: Path) -> Any:
    return joblib.load(path)


def latest_version(session: Session, name: str) -> db.ModelVersion | None:
    return session.scalar(
        select(db.ModelVersion)
        .where(db.ModelVersion.name == name)
        .order_by(db.ModelVersion.trained_at.desc())
        .limit(1)
    )
