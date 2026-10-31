"""Experiment tracking: MLflow when available, JSON files otherwise.

Every evaluation run logs its parameters, metrics and artifacts so results
are reproducible and comparable over time.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_ai.core.io import atomic_write_json

logger = logging.getLogger(__name__)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _flat(prefix: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, bool | int | float):
            out[name] = float(value)
    return out


@dataclass(slots=True)
class RunHandle:
    run_id: str
    backend: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    _mlflow: Any = None

    def log_metrics(self, metrics: Mapping[str, Any], prefix: str = "") -> None:
        flat = _flat(prefix, metrics)
        self.metrics.update(flat)
        if self._mlflow is not None and flat:
            self._mlflow.log_metrics(flat)

    def log_artifact(self, path: Path) -> None:
        self.artifacts.append(str(path))
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))


class ExperimentTracker:
    def __init__(self, root: Path, experiment: str, tracking_uri: str | None = None) -> None:
        self.root = root
        self.experiment = experiment
        self.tracking_uri = tracking_uri or f"sqlite:///{(root / 'mlflow.db').resolve()}"
        self._mlflow: Any = None
        try:
            import mlflow

            root.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
        except Exception as exc:  # optional dependency / backend unavailable
            logger.warning("mlflow_unavailable", extra={"error": str(exc)})

    @property
    def backend(self) -> str:
        return "mlflow" if self._mlflow is not None else "json"

    @contextmanager
    def run(self, name: str, params: Mapping[str, Any]) -> Iterator[RunHandle]:
        clean = {k: json.dumps(v, default=str) if not isinstance(v, str) else v
                 for k, v in params.items()}  # fmt: skip
        if self._mlflow is not None:
            with self._mlflow.start_run(run_name=name) as active:
                self._mlflow.log_params(clean)
                handle = RunHandle(active.info.run_id, "mlflow", dict(params), _mlflow=self._mlflow)
                yield handle
        else:
            handle = RunHandle(uuid.uuid4().hex, "json", dict(params))
            yield handle
        # JSON record is always written (also with MLflow) for offline inspection
        atomic_write_json(
            self.root / "runs" / self.experiment / f"{name}_{handle.run_id}.json",
            {
                "run_id": handle.run_id,
                "backend": handle.backend,
                "name": name,
                "created_at": datetime.now(UTC).isoformat(),
                "params": handle.params,
                "metrics": handle.metrics,
                "artifacts": handle.artifacts,
            },
        )
